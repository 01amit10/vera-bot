"""
composer.py — Vera's deterministic message composition engine.

Given 4 contexts (category, merchant, trigger, customer?), produces
a composed WhatsApp message with body, cta, send_as, suppression_key,
and rationale.

Every decision is grounded in the context provided — no fabrication.
"""

import json
import os
import re
from typing import Optional
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Module-level pattern constants (importable without instantiating any client)
# ---------------------------------------------------------------------------

AUTO_REPLY_PATTERNS = [
    "thank you for contacting",
    "our team will respond",
    "i am an automated",
    "automated message",
    "bahut-bahut shukriya",
    "aapki jaankari ke liye",
    "we will get back",
    "will contact you shortly",
    "out of office",
    "automated assistant",
]

HOSTILE_PATTERNS = [
    "stop messaging",
    "spam",
    "useless",
    "leave me alone",
    "don't contact",
    "mat karo",
    "band karo",
    "not interested",
    "remove me",
    "unsubscribe",
    "why are you bothering",
    "stop sending",
    "this is useless",
]

INTENT_COMMIT_PATTERNS = [
    "let's do it", "lets do it",
    "ok go ahead", "yes go ahead",
    "confirm", "confirmed",
    "send it", "send the",
    "proceed", "chaliye",
    "haan karo", "kar do",
    "shuru karo",
    "what's next", "whats next",
    "next step",
    "ok let",
]

OUT_OF_SCOPE_PATTERNS = [
    "gst", "income tax", "tax filing",
    "legal advice", "court", "lawsuit",
    "insurance claim", "loan application",
    "property dispute", "bank account",
]

# ---------------------------------------------------------------------------
# System prompt — the core Vera persona and scoring rubric
# ---------------------------------------------------------------------------
VERA_SYSTEM = """You are Vera, magicpin's AI assistant for merchant growth. You compose WhatsApp messages on behalf of Vera (merchant-facing) or on behalf of the merchant (customer-facing).

SCORING CRITERIA (each 0-10, you must maximize all 5):

1. DECISION QUALITY — pick the single best signal from trigger + merchant state + category, then decide whether to send at all.

2. SPECIFICITY — anchor on verifiable facts: real numbers, dates, citations, prices, names. Never invent data. If a data point isn't in the context, don't use it.

3. CATEGORY FIT — dentists: peer/clinical, no hype; salons: warm/practical; restaurants: operator-to-operator; gyms: coaching/motivational; pharmacies: trustworthy/precise.

4. MERCHANT FIT — personalize to this merchant's numbers, signals, offers, conversation history, and language preference. Use owner first name. Honor language pref (hi-en mix = code-mix).

5. ENGAGEMENT COMPULSION — one clear reason to reply NOW: loss aversion, curiosity, social proof, effort externalization, or single binary CTA. Never multi-CTA.

HARD RULES:
- Single CTA per message (binary YES/STOP for action triggers; open_ended for info triggers; none for pure info)
- No fabricated data — if not in context, omit it
- No "guaranteed", "100% safe", "miracle", "best in city"
- No URLs in the message body
- No long preambles ("I hope you're doing well...")
- Keep messages concise, max ~200 chars for simple nudges, ~400 chars for complex ones
- Hindi-English code-mix when merchant.identity.languages includes "hi"
- For customer-facing messages: no medical claims, warm tone"""

# ---------------------------------------------------------------------------
# Trigger-kind to composition strategy mapping
# ---------------------------------------------------------------------------
TRIGGER_STRATEGIES = {
    "research_digest": "Share a specific research finding from the digest relevant to this merchant's cohort. Use clinical language for dentists. Offer to draft a patient-facing WhatsApp. CTA: open_ended.",
    "regulation_change": "Alert the merchant to a compliance requirement with a specific deadline. Use official source name. Offer to help audit. CTA: binary_yes_no.",
    "festival_upcoming": "Frame the festival as a growth window with specific days. Reference merchant's active offer. Offer to draft a GBP post. CTA: binary_yes_no.",
    "category_seasonal": "Use specific trend percentages from trigger payload. Recommend concrete service changes. CTA: binary_yes_no.",
    "competitor_opened": "Name the competitor and distance. Recommend a specific counter-move using existing offers. CTA: binary_yes_no.",
    "category_trend_movement": "Quote the specific trend search delta. Tie to merchant's existing services. Suggest GBP post. CTA: binary_yes_no.",
    "cde_opportunity": "Mention specific webinar title, date, credits, fee. CTA: binary_yes_no.",
    "perf_spike": "Celebrate the specific metric spike with the actual number. Suggest amplifying momentum. CTA: binary_yes_no.",
    "perf_dip": "State the specific metric and drop percentage. Offer specific corrective action. CTA: binary_yes_no.",
    "seasonal_perf_dip": "Reframe the dip as normal seasonal pattern with industry benchmark. Recommend retention focus. CTA: binary_yes_no.",
    "milestone_reached": "Name the exact metric and milestone value. Offer to help hit it. CTA: binary_yes_no.",
    "dormant_with_vera": "Re-engage warmly with a specific hook. Don't reference dormancy. CTA: open_ended.",
    "review_theme_emerged": "Quote the specific theme and occurrence count. Offer a concrete resolution action. CTA: binary_yes_no.",
    "renewal_due": "State the exact days remaining and plan name. Show a specific metric that would be lost. CTA: binary_yes_no.",
    "stale_posts": "Mention days since last post. Offer to draft 2-3 posts immediately. CTA: binary_yes_no.",
    "recall_due": "Name the specific service due and elapsed time. Offer 2 specific time slots. Include the price. send_as: merchant_on_behalf. CTA: multi_choice_slot.",
    "customer_lapsed_hard": "Warm winback, no shame. Reference previous service. Announce new offering. send_as: merchant_on_behalf. CTA: binary_yes_no.",
    "trial_followup": "Reference the specific trial date. Offer next session with date/time. send_as: merchant_on_behalf. CTA: binary_yes_no.",
    "supply_alert": "Name batch numbers and molecule. State risk level. Offer complete workflow. CTA: binary_yes_no.",
    "curious_ask_due": "Ask the merchant a single, low-friction curiosity question. Offer to turn their answer into a deliverable. CTA: open_ended.",
    "scheduled_recurring": "Pick the most interesting category-relevant hook from digest. Frame as peer intelligence. CTA: open_ended.",
}


def _get_strategy(trigger_kind: str) -> str:
    return TRIGGER_STRATEGIES.get(trigger_kind,
        "Compose a relevant, specific message tied to the trigger context. Use real data from contexts.")


def _build_context_summary(category: dict, merchant: dict, trigger: dict,
                            customer: Optional[dict] = None) -> str:
    """Build a structured context summary for the LLM prompt."""

    # ---- Category ----
    cat_slug = category.get("slug", "unknown")
    voice = category.get("voice", {})
    taboos = voice.get("vocab_taboo", [])
    digest_items = category.get("digest", [])
    peer_stats = category.get("peer_stats", {})
    seasonal_beats = category.get("seasonal_beats", [])
    offer_catalog = category.get("offer_catalog", [])

    # ---- Merchant ----
    identity = merchant.get("identity", {})
    merchant_name = identity.get("name", "")
    owner_name = identity.get("owner_first_name", "")
    city = identity.get("city", "")
    locality = identity.get("locality", "")
    languages = identity.get("languages", ["en"])

    subscription = merchant.get("subscription", {})
    sub_status = subscription.get("status", "")
    sub_days = subscription.get("days_remaining", "?")

    perf = merchant.get("performance", {})
    offers = merchant.get("offers", [])
    active_offers = [o for o in offers if o.get("status") == "active"]
    signals = merchant.get("signals", [])
    conv_history = merchant.get("conversation_history", [])
    cust_agg = merchant.get("customer_aggregate", {})
    review_themes = merchant.get("review_themes", [])

    merchant_ctr = perf.get("ctr", 0)
    peer_ctr = peer_stats.get("avg_ctr", 0)
    ctr_vs_peer = "below_peer" if merchant_ctr < peer_ctr else "above_peer" if merchant_ctr > peer_ctr else "at_peer"

    # ---- Trigger ----
    trigger_kind = trigger.get("kind", "")
    trigger_scope = trigger.get("scope", "merchant")
    trigger_payload = trigger.get("payload", {})
    trigger_urgency = trigger.get("urgency", 3)
    suppression_key = trigger.get("suppression_key", f"{trigger_kind}:{merchant.get('merchant_id','')}")

    # Find relevant digest item
    relevant_digest = None
    item_id = trigger_payload.get("top_item_id") or trigger_payload.get("digest_item_id")
    if item_id:
        relevant_digest = next((d for d in digest_items if d.get("id") == item_id), None)

    # ---- Customer ----
    customer_section = ""
    if customer:
        c_identity = customer.get("identity", {})
        c_rel = customer.get("relationship", {})
        c_prefs = customer.get("preferences", {})
        customer_section = f"""
CUSTOMER CONTEXT (message is TO this customer FROM the merchant):
  Name: {c_identity.get('name', '')}
  Language pref: {c_identity.get('language_pref', 'english')}
  Age band: {c_identity.get('age_band', '')}
  First visit: {c_rel.get('first_visit', '')}
  Last visit: {c_rel.get('last_visit', '')}
  Services received: {c_rel.get('services_received', [])}
  Preferred slots: {c_prefs.get('preferred_slots', '')}
  Senior citizen: {c_identity.get('senior_citizen', False)}"""

    summary = f"""=== CONTEXT ===

CATEGORY: {cat_slug}
  Voice/tone: {voice.get('tone', '')}
  Taboo words (NEVER use): {taboos}
  Peer stats: avg_views_30d={peer_stats.get('avg_views_30d','?')}, avg_ctr={peer_ctr}, avg_calls_30d={peer_stats.get('avg_calls_30d','?')}
  Seasonal beats: {json.dumps(seasonal_beats[:2])}
  Category offer catalog: {json.dumps(offer_catalog[:4])}

RELEVANT DIGEST ITEM:
{json.dumps(relevant_digest, indent=2) if relevant_digest else '  (none directly linked — use most relevant from digest below)'}

ALL DIGEST ITEMS:
{json.dumps(digest_items, indent=2)}

MERCHANT: {merchant_name} ({cat_slug}, {locality}, {city})
  Owner first name: {owner_name}
  Languages: {languages}
  Subscription: {sub_status}, {sub_days} days remaining
  Performance (30d): views={perf.get('views','?')}, calls={perf.get('calls','?')}, ctr={merchant_ctr}
  CTR vs peer: {ctr_vs_peer} (merchant: {merchant_ctr}, peer median: {peer_ctr})
  7d deltas: {json.dumps(perf.get('delta_7d', {}))}
  Active offers: {json.dumps(active_offers)}
  Signals: {signals}
  Customer aggregate: {json.dumps(cust_agg)}
  Review themes: {json.dumps(review_themes)}
  Recent conversation history: {json.dumps(conv_history[-3:] if conv_history else [])}
{customer_section}

TRIGGER:
  Kind: {trigger_kind}
  Scope: {trigger_scope}
  Urgency: {trigger_urgency}/5
  Payload: {json.dumps(trigger_payload, indent=2)}
  Suppression key: {suppression_key}

=== COMPOSITION STRATEGY ===
{_get_strategy(trigger_kind)}
"""
    return summary


def _build_compose_prompt(context_summary: str, customer_facing: bool) -> str:
    send_as_note = ("send_as MUST be 'merchant_on_behalf' since there is a customer context."
                    if customer_facing else "send_as MUST be 'vera' since this is a merchant-facing message.")

    return f"""{context_summary}

=== YOUR TASK ===
Compose the next WhatsApp message. {send_as_note}

RESPOND WITH ONLY THIS JSON (no markdown, no extra text):
{{
  "body": "<the WhatsApp message body — concise, specific, no URLs, use Rs. for rupees>",
  "cta": "<one of: open_ended | binary_yes_no | binary_confirm_cancel | multi_choice_slot | none>",
  "send_as": "<vera | merchant_on_behalf>",
  "suppression_key": "<copy from trigger suppression_key>",
  "rationale": "<2-3 sentences: which signal triggered this, what merchant-specific fact anchors the message, what compulsion lever is used>"
}}

CHECKLIST before responding:
- Does the body use at least 2 specific numbers/names/dates from the context?
- Is the tone correct for the category?
- Is the owner name used?
- Is the CTA the last sentence?
- No URLs in body?
- No fabricated data?
- Under 400 characters?"""


class GeminiClient:
    """Thin wrapper around Google Gemini API (new google-genai SDK)."""

    def __init__(self, api_key: str, model: str = "gemini-3.8-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model

    def complete(self, prompt: str) -> str:
        """Call Gemini and return text response."""
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=VERA_SYSTEM,
                max_output_tokens=900,
            )
        )
        return response.text.strip()


class VeraComposer:
    """Core composition engine for Vera using Google Gemini."""

    def __init__(self, api_key: str, model: str = "gemini-3.8-flash"):
        self.llm = GeminiClient(api_key, model)

    def compose(
        self,
        category: dict,
        merchant: dict,
        trigger: dict,
        customer: Optional[dict] = None
    ) -> dict:
        """
        Core compose function.
        Returns: {body, cta, send_as, suppression_key, rationale}
        """
        customer_facing = customer is not None
        context_summary = _build_context_summary(category, merchant, trigger, customer)
        prompt = _build_compose_prompt(context_summary, customer_facing)

        raw = self.llm.complete(prompt)
        return self._parse_response(raw, trigger, merchant, customer_facing)

    def _parse_response(self, raw: str, trigger: dict, merchant: dict,
                        customer_facing: bool) -> dict:
        # Strip markdown code fences if present
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'```\s*$', '', raw, flags=re.MULTILINE)

        match = re.search(r'\{[\s\S]*\}', raw)
        if not match:
            return self._fallback(trigger, merchant, customer_facing)

        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return self._fallback(trigger, merchant, customer_facing)

        body = re.sub(r'https?://\S+', '', data.get("body", "")).strip()
        cta = data.get("cta", "open_ended")
        send_as = data.get("send_as", "merchant_on_behalf" if customer_facing else "vera")
        suppression_key = data.get("suppression_key", trigger.get("suppression_key", ""))
        rationale = data.get("rationale", "")

        valid_ctas = {"open_ended", "binary_yes_no", "binary_confirm_cancel",
                      "multi_choice_slot", "none", "binary_yes_stop"}
        if cta not in valid_ctas:
            cta = "open_ended"

        return {
            "body": body,
            "cta": cta,
            "send_as": send_as,
            "suppression_key": suppression_key,
            "rationale": rationale
        }

    def _fallback(self, trigger: dict, merchant: dict, customer_facing: bool) -> dict:
        identity = merchant.get("identity", {})
        name = identity.get("owner_first_name") or identity.get("name", "")
        return {
            "body": f"Hi {name}, Vera here — a quick update worth your attention. Reply YES to see what's new.",
            "cta": "binary_yes_no",
            "send_as": "merchant_on_behalf" if customer_facing else "vera",
            "suppression_key": trigger.get("suppression_key", "fallback"),
            "rationale": "Fallback message — LLM parse failed"
        }


class ReplyComposer:
    """Handles multi-turn conversation replies using Gemini."""

    def __init__(self, api_key: str, model: str = "gemini-3.8-flash"):
        self.llm = GeminiClient(api_key, model)

    def compose_reply(
        self,
        conversation_id: str,
        message: str,
        conversation_history: list,
        merchant: dict,
        category: dict,
        trigger: Optional[dict] = None,
        customer: Optional[dict] = None,
        turn_number: int = 2
    ) -> dict:
        """
        Compose a reply — fast-path detection runs in bot.py before this is called.
        This handles: intent-commit and general conversational replies.
        """
        # Intent commit fast-path
        msg_lower = message.lower()
        if any(p in msg_lower for p in INTENT_COMMIT_PATTERNS):
            return self._compose_action_reply(message, merchant, category, trigger, conversation_history)

        # General reply via LLM
        return self._compose_general_reply(message, merchant, category, trigger, conversation_history, turn_number)

    def _compose_action_reply(self, message: str, merchant: dict, category: dict,
                               trigger: Optional[dict], conversation_history: list) -> dict:
        identity = merchant.get("identity", {})
        owner = identity.get("owner_first_name", "")
        trigger_kind = trigger.get("kind", "") if trigger else ""
        active_offers = [o for o in merchant.get("offers", []) if o.get("status") == "active"]

        prompt = f"""Merchant just committed: "{message}"

Merchant: {identity.get('name')} (owner: {owner})
Category: {category.get('slug')}
Languages: {identity.get('languages', ['en'])}
Trigger was: {trigger_kind}
Active offers: {json.dumps(active_offers)}
Previous conversation: {json.dumps(conversation_history[-3:])}

Compose an ACTION-MODE response (NOT a qualifying question). You are NOW doing the work.
Confirm the action and state the concrete next step.

RESPOND WITH ONLY THIS JSON:
{{"body": "<action message>", "cta": "<binary_confirm_cancel|binary_yes_no|none>", "rationale": "<reasoning>"}}"""

        try:
            raw = self.llm.complete(prompt)
            raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
            raw = re.sub(r'```\s*$', '', raw, flags=re.MULTILINE)
            match = re.search(r'\{[\s\S]*\}', raw)
            if match:
                data = json.loads(match.group())
                body = re.sub(r'https?://\S+', '', data.get("body", "")).strip()
                return {
                    "action": "send",
                    "body": body,
                    "cta": data.get("cta", "none"),
                    "rationale": data.get("rationale", "Merchant committed; switching to action mode.")
                }
        except Exception:
            pass

        return {
            "action": "send",
            "body": f"On it! Drafting this for you right now. Reply CONFIRM when ready to send.",
            "cta": "binary_confirm_cancel",
            "rationale": "Merchant committed to action; executing immediately."
        }

    def _compose_general_reply(self, message: str, merchant: dict, category: dict,
                                trigger: Optional[dict], conversation_history: list, turn_number: int) -> dict:
        identity = merchant.get("identity", {})
        owner = identity.get("owner_first_name", "")
        trigger_kind = trigger.get("kind", "") if trigger else ""
        active_offers = [o for o in merchant.get("offers", []) if o.get("status") == "active"]

        prompt = f"""You are Vera continuing a conversation with a merchant.

Merchant message: "{message}"
Turn: {turn_number}
Owner: {owner}
Category: {category.get('slug')}
Languages: {identity.get('languages', ['en'])}
Trigger was: {trigger_kind}
Active offers: {json.dumps(active_offers)}
Customer aggregate: {json.dumps(merchant.get('customer_aggregate', {}))}
Conversation so far: {json.dumps(conversation_history[-4:])}

Compose a helpful, specific, category-appropriate reply. End with one clear next step.

RESPOND WITH ONLY THIS JSON:
{{"body": "<reply body>", "cta": "<open_ended|binary_yes_no|none>", "rationale": "<reasoning>"}}"""

        try:
            raw = self.llm.complete(prompt)
            raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
            raw = re.sub(r'```\s*$', '', raw, flags=re.MULTILINE)
            match = re.search(r'\{[\s\S]*\}', raw)
            if match:
                data = json.loads(match.group())
                body = re.sub(r'https?://\S+', '', data.get("body", "")).strip()
                return {
                    "action": "send",
                    "body": body,
                    "cta": data.get("cta", "open_ended"),
                    "rationale": data.get("rationale", "General contextual reply")
                }
        except Exception:
            pass

        return {
            "action": "send",
            "body": "Got it — let me look into that and follow up shortly.",
            "cta": "open_ended",
            "rationale": "General reply fallback"
        }


def _get_prev_topic(conversation_history: list) -> str:
    if not conversation_history:
        return "your listing"
    last_vera = next(
        (h for h in reversed(conversation_history) if h.get("from_role") == "vera"),
        None
    )
    if last_vera:
        body = last_vera.get("body", last_vera.get("msg", ""))
        return body[:60] + "..." if len(body) > 60 else body
    return "the campaign"

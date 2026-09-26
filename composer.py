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
import time
from typing import Optional
from anthropic import Anthropic

# ---------------------------------------------------------------------------
# Module-level pattern constants (importable without instantiating Anthropic)
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
- No re-introducing Vera after the first message
- Keep messages concise, max ~200 chars for simple nudges, ~400 chars for complex ones
- Hindi-English code-mix when merchant.identity.languages includes "hi" and language_pref is "hi-en mix"
- For customer-facing messages: no medical claims, warm tone, honor language preference

AUTO-REPLY DETECTION PATTERNS:
- "Thank you for contacting", "Our team will respond", "automated response", "bahut-bahut shukriya" (canned)
- If same text verbatim 2+ times in conversation history → auto-reply

INTENT TRANSITION:
- When merchant says "ok let's do it", "yes go ahead", "confirm", "send it", "let's start" → SWITCH to action mode immediately, never ask another qualifying question.

CONVERSATION ENDING:
- If merchant says "stop", "not interested", "spam", abuse → action: "end"
- If out-of-scope ask (GST, legal, etc.) → politely decline, redirect to original topic
"""

# ---------------------------------------------------------------------------
# Trigger-kind to composition strategy mapping
# ---------------------------------------------------------------------------
TRIGGER_STRATEGIES = {
    # External triggers
    "research_digest": "Share a specific research finding from the digest that's directly relevant to this merchant's patient/customer cohort. Use clinical language for dentists. Offer to draft a patient-facing WhatsApp the merchant can share. CTA: open_ended.",
    "regulation_change": "Alert the merchant to a compliance requirement with a specific deadline. Use official source name. Offer to help audit or document. CTA: binary_yes_no.",
    "festival_upcoming": "Frame the festival as a growth window with a specific number of days. Reference the merchant's active offer or the category offer catalog. Offer to draft a GBP post or campaign. CTA: binary_yes_no.",
    "category_seasonal": "Use specific trend percentages from the trigger payload. Recommend concrete shelf/service changes. Tie to merchant's current offers. CTA: binary_yes_no.",
    "competitor_opened": "Name the competitor and distance (both in trigger payload). Recommend a specific counter-move using merchant's existing offers. Don't disparage — focus on differentiation. CTA: binary_yes_no.",
    "category_trend_movement": "Quote the specific trend search delta from category data. Tie to merchant's existing services or offer catalog. Suggest GBP post or new offer. CTA: binary_yes_no.",
    "cde_opportunity": "Mention the specific webinar/CDE title, date, credits, and fee. Reference source. CTA: binary_yes_no.",
    "ipl_match_today": "Name the match and venue. If Saturday (not weeknight), warn that Saturday IPL = home viewing = restaurant footfall dip; recommend delivery pivot instead. If weeknight, recommend a match-night special. CTA: binary_yes_no.",
    "weather_heatwave": "Use the specific temperature. Recommend category-appropriate seasonal action (pharmacy: ORS/sunscreen; restaurant: cold drinks/delivery; gym: early morning timing). CTA: binary_yes_no.",
    "local_news_event": "Name the specific event from trigger payload. Recommend timing or logistics adjustment. CTA: open_ended.",
    # Internal triggers
    "perf_spike": "Celebrate the specific metric spike with the actual number. Explain likely driver if in payload. Suggest amplifying the momentum. CTA: binary_yes_no.",
    "perf_dip": "State the specific metric and drop percentage. Distinguish seasonal vs. unexpected. Offer specific corrective action. CTA: binary_yes_no.",
    "seasonal_perf_dip": "Reframe the dip as normal seasonal pattern with industry benchmark range. Recommend retention focus instead of acquisition. Offer specific retention campaign. CTA: binary_yes_no.",
    "milestone_reached": "Name the exact metric and milestone value. Show how close they are. Offer to help hit it with a specific action. CTA: binary_yes_no.",
    "dormant_with_vera": "Re-engage warmly with a specific hook relevant to their category or recent signals. Don't reference the dormancy period explicitly. CTA: open_ended.",
    "winback_eligible": "For expired subscription merchants: acknowledge the gap, show a specific metric consequence (lapsed customers, views dip), offer a concrete restart path. CTA: binary_yes_no.",
    "review_theme_emerged": "Quote the specific theme and occurrence count from the trigger. Offer a concrete resolution action (response template, process change). CTA: binary_yes_no.",
    "renewal_due": "State the exact days remaining and plan name. Show a specific metric that would be lost without renewal. CTA: binary_yes_no.",
    "gbp_unverified": "State the specific estimated uplift from verification. Explain the verification paths. Offer to guide the process. CTA: binary_yes_no.",
    "stale_posts": "Mention the number of days since last post (from signals). Offer to draft 2-3 posts immediately. CTA: binary_yes_no.",
    # Customer-triggered
    "recall_due": "For customer-facing: name the specific service due and elapsed time. Offer 2 specific time slots from trigger payload. Include the price from merchant's active offers. send_as: merchant_on_behalf. CTA: multi_choice_slot.",
    "chronic_refill_due": "For pharmacy customer: name all molecule names. State exact run-out date. Confirm same dose/brand. Show total with senior discount if applicable. Offer home delivery to saved address. send_as: merchant_on_behalf. CTA: binary_confirm_cancel.",
    "customer_lapsed_hard": "Warm winback, no shame. Reference their previous focus/service. Announce a specific new offering that matches. Offer a no-commitment trial. send_as: merchant_on_behalf. CTA: binary_yes_no.",
    "trial_followup": "Reference the specific trial date. Offer next session with date/time. Show the enrollment offer price. send_as: merchant_on_behalf. CTA: binary_yes_no.",
    "wedding_package_followup": "Count days to wedding. Reference the completed trial. Name the specific next-step program with price. Offer to block preferred slot. send_as: merchant_on_behalf. CTA: binary_yes_no.",
    "supply_alert": "Name batch numbers and molecule. State risk level (sub-potency, not safety). Derive affected customer count from merchant's chronic_rx_count if possible. Offer complete workflow. CTA: binary_yes_no.",
    "active_planning_intent": "Bot received merchant's explicit planning question → give a complete draft answer NOW. Don't ask qualifying questions. Show the plan in structured form. CTA: open_ended.",
    "curious_ask_due": "Ask the merchant a single, low-friction curiosity question about their business. Offer to turn their answer into a concrete deliverable (GBP post, WhatsApp reply template). CTA: open_ended.",
    "scheduled_recurring": "Pick the most interesting category-relevant hook from digest or trend signals. Frame as peer intelligence. CTA: open_ended.",
}


def _get_strategy(trigger_kind: str) -> str:
    """Get composition strategy for trigger kind."""
    return TRIGGER_STRATEGIES.get(trigger_kind, 
        "Compose a relevant, specific message tied to the trigger context. Use real data from the contexts.")


def _build_context_summary(category: dict, merchant: dict, trigger: dict, 
                            customer: Optional[dict] = None) -> str:
    """Build a structured context summary for the LLM prompt."""
    
    # ---- Category ----
    cat_slug = category.get("slug", "unknown")
    voice = category.get("voice", {})
    voice_tone = voice.get("tone", "")
    taboos = voice.get("vocab_taboo", [])
    
    # Get relevant digest items
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
    
    # Compute CTR vs peer
    merchant_ctr = perf.get("ctr", 0)
    peer_ctr = peer_stats.get("avg_ctr", 0)
    ctr_vs_peer = "below_peer" if merchant_ctr < peer_ctr else "above_peer" if merchant_ctr > peer_ctr else "at_peer"
    
    # ---- Trigger ----
    trigger_kind = trigger.get("kind", "")
    trigger_scope = trigger.get("scope", "merchant")
    trigger_payload = trigger.get("payload", {})
    trigger_urgency = trigger.get("urgency", 3)
    suppression_key = trigger.get("suppression_key", f"{trigger_kind}:{merchant.get('merchant_id','')}")
    
    # Find relevant digest item for trigger
    relevant_digest = None
    if "top_item_id" in trigger_payload or "digest_item_id" in trigger_payload:
        item_id = trigger_payload.get("top_item_id") or trigger_payload.get("digest_item_id")
        relevant_digest = next((d for d in digest_items if d.get("id") == item_id), None)
    
    # ---- Customer ----
    customer_section = ""
    if customer:
        c_identity = customer.get("identity", {})
        c_rel = customer.get("relationship", {})
        c_state = customer.get("state", "")
        c_prefs = customer.get("preferences", {})
        c_consent = customer.get("consent", {})
        customer_section = f"""
CUSTOMER CONTEXT (message is TO this customer FROM the merchant):
  Name: {c_identity.get('name', '')}
  Language pref: {c_identity.get('language_pref', 'english')}
  Age band: {c_identity.get('age_band', '')}
  State: {c_state}
  First visit: {c_rel.get('first_visit', '')}
  Last visit: {c_rel.get('last_visit', '')}
  Visits total: {c_rel.get('visits_total', 0)}
  Services received: {c_rel.get('services_received', [])}
  Preferred slots: {c_prefs.get('preferred_slots', '')}
  Consent scope: {c_consent.get('scope', [])}
  Senior citizen: {c_identity.get('senior_citizen', False)}
  Channel: {c_prefs.get('channel', 'whatsapp')}"""
    
    summary = f"""=== CONTEXT ===

CATEGORY: {cat_slug}
  Voice/tone: {voice_tone}
  Taboo words (NEVER use): {taboos}
  Peer stats: avg_views_30d={peer_stats.get('avg_views_30d', '?')}, avg_ctr={peer_ctr}, avg_calls_30d={peer_stats.get('avg_calls_30d', '?')}
  Seasonal beats: {json.dumps(seasonal_beats[:2])}
  Category offer catalog (use these templates if merchant has no active offers): {json.dumps(offer_catalog[:4])}
  
RELEVANT DIGEST ITEM (from trigger):
{json.dumps(relevant_digest, indent=2) if relevant_digest else '  (none directly linked — use most relevant from digest below)'}

ALL DIGEST ITEMS:
{json.dumps(digest_items, indent=2)}

MERCHANT: {merchant_name} ({cat_slug}, {locality}, {city})
  Owner first name: {owner_name}
  Languages: {languages}
  Subscription: {sub_status}, {sub_days} days remaining
  Performance (30d): views={perf.get('views','?')}, calls={perf.get('calls','?')}, directions={perf.get('directions','?')}, ctr={merchant_ctr}
  CTR vs peer: {ctr_vs_peer} (merchant: {merchant_ctr}, peer median: {peer_ctr})
  7d deltas: {json.dumps(perf.get('delta_7d', {}))}
  Active offers: {json.dumps(active_offers)}
  All signals: {signals}
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
    """Build the final compose prompt."""
    send_as_note = ("send_as MUST be 'merchant_on_behalf' since there is a customer context." 
                    if customer_facing else "send_as MUST be 'vera' since this is a merchant-facing message.")
    
    return f"""{context_summary}

=== YOUR TASK ===
Compose the next WhatsApp message. {send_as_note}

RESPOND WITH ONLY THIS JSON (no markdown, no explanation outside the JSON):
{{
  "body": "<the WhatsApp message body — concise, specific, no URLs, use ₹ for rupees>",
  "cta": "<one of: open_ended | binary_yes_no | binary_confirm_cancel | multi_choice_slot | none>",
  "send_as": "<vera | merchant_on_behalf>",
  "suppression_key": "<copy from trigger suppression_key>",
  "rationale": "<2-3 sentences explaining: which signal triggered this, what merchant-specific fact anchors the message, and what compulsion lever is used>"
}}

CHECKLIST before responding:
✓ Does the body use at least 2 specific numbers/names/dates from the context?
✓ Is the tone correct for the category?
✓ Is the owner name used (not just the clinic name)?
✓ Is the CTA the last sentence?
✓ No URLs in body?
✓ No fabricated data (every claim is in the context above)?
✓ Is it under 400 characters (aim for concise)?"""


class VeraComposer:
    """
    The core composition engine for Vera.
    Deterministic at temperature=0; uses Claude claude-3-5-haiku for speed/cost balance.
    """
    
    def __init__(self, api_key: str, model: str = "claude-3-5-haiku-20241022"):
        self.client = Anthropic(api_key=api_key)
        self.model = model
    
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
        
        # Call Claude with low effort for fast, consistent output
        response = self.client.messages.create(
            model=self.model,
            max_tokens=800,
            system=VERA_SYSTEM,
            messages=[{"role": "user", "content": prompt}]
        )
        
        raw = response.content[0].text.strip()
        
        # Parse JSON response
        result = self._parse_response(raw, trigger, merchant, customer_facing)
        return result
    
    def _parse_response(self, raw: str, trigger: dict, merchant: dict, 
                        customer_facing: bool) -> dict:
        """Parse and validate LLM response."""
        # Try to extract JSON
        match = re.search(r'\{[\s\S]*\}', raw)
        if not match:
            return self._fallback(trigger, merchant, customer_facing)
        
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return self._fallback(trigger, merchant, customer_facing)
        
        # Validate and sanitize
        body = data.get("body", "").strip()
        cta = data.get("cta", "open_ended")
        send_as = data.get("send_as", "merchant_on_behalf" if customer_facing else "vera")
        suppression_key = data.get("suppression_key", trigger.get("suppression_key", ""))
        rationale = data.get("rationale", "")
        
        # Safety: strip URLs from body
        body = re.sub(r'https?://\S+', '', body).strip()
        
        # Validate cta values
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
        """Minimal fallback if LLM fails."""
        identity = merchant.get("identity", {})
        name = identity.get("owner_first_name") or identity.get("name", "")
        return {
            "body": f"Hi {name}, quick update from Vera — check your magicpin dashboard for the latest insights.",
            "cta": "open_ended",
            "send_as": "merchant_on_behalf" if customer_facing else "vera",
            "suppression_key": trigger.get("suppression_key", "fallback"),
            "rationale": "Fallback message — LLM parse failed"
        }


class ReplyComposer:
    """
    Handles multi-turn conversation replies.
    Detects auto-replies, intent transitions, hostile messages.
    """
    
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
        "stop",
    ]
    
    INTENT_COMMIT_PATTERNS = [
        "let's do it", "lets do it",
        "ok go ahead", "yes go ahead",
        "confirm", "confirmed",
        "send it", "send the",
        "proceed", "chaliye",
        "haan karo", "kar do",
        "karo", "shuru karo",
        "what's next", "whats next",
        "next step",
    ]
    
    def __init__(self, api_key: str, model: str = "claude-3-5-haiku-20241022"):
        self.client = Anthropic(api_key=api_key)
        self.model = model
    
    def is_auto_reply(self, message: str) -> bool:
        """Detect WhatsApp Business canned auto-replies."""
        msg_lower = message.lower()
        return any(pattern in msg_lower for pattern in self.AUTO_REPLY_PATTERNS)
    
    def is_hostile(self, message: str) -> bool:
        """Detect explicit opt-out or hostile messages."""
        msg_lower = message.lower()
        return any(pattern in msg_lower for pattern in self.HOSTILE_PATTERNS)
    
    def is_intent_commit(self, message: str) -> bool:
        """Detect explicit commitment/action intent."""
        msg_lower = message.lower()
        return any(pattern in msg_lower for pattern in self.INTENT_COMMIT_PATTERNS)
    
    def is_out_of_scope(self, message: str) -> bool:
        """Detect out-of-scope requests."""
        oos_patterns = ["gst", "tax", "legal", "court", "lawsuit", "insurance claim",
                        "government", "loan", "bank", "property dispute"]
        msg_lower = message.lower()
        return any(p in msg_lower for p in oos_patterns)
    
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
        Compose a reply to a merchant/customer message.
        Returns: {action: "send"|"wait"|"end", body?, cta?, rationale}
        """
        
        # --- Auto-reply detection ---
        auto_reply_count = sum(
            1 for h in conversation_history 
            if h.get("from_role") == "merchant" and self.is_auto_reply(h.get("message", ""))
        )
        # Also check current message
        if self.is_auto_reply(message):
            auto_reply_count += 1
        
        if auto_reply_count >= 3:
            return {
                "action": "end",
                "rationale": f"Auto-reply detected {auto_reply_count}× in a row — no real owner engagement. Closing conversation."
            }
        elif auto_reply_count == 2:
            return {
                "action": "wait",
                "wait_seconds": 86400,
                "rationale": "Same auto-reply twice — owner not at phone. Waiting 24h before retry."
            }
        elif auto_reply_count == 1:
            owner = merchant.get("identity", {}).get("owner_first_name", "")
            return {
                "action": "send",
                "body": f"Looks like an auto-reply 😊 When {owner or 'the owner'} sees this, just reply YES to continue.",
                "cta": "binary_yes_no",
                "rationale": "Detected WhatsApp Business auto-reply; leaving a minimal prompt for the owner to pick up."
            }
        
        # --- Hostile detection ---
        if self.is_hostile(message):
            return {
                "action": "end",
                "rationale": "Merchant expressed opt-out or hostility. Closing conversation gracefully; suppressing for 30 days."
            }
        
        # --- Out-of-scope detection ---
        if self.is_out_of_scope(message):
            # Redirect politely
            identity = merchant.get("identity", {})
            owner = identity.get("owner_first_name", "")
            prev_topic = _get_prev_topic(conversation_history)
            return {
                "action": "send",
                "body": f"That's outside what I can help with directly — you'd need your CA or a specialist for that. Coming back to {prev_topic or 'what we were discussing'} — want me to proceed?",
                "cta": "binary_yes_no",
                "rationale": "Out-of-scope request politely declined; redirecting to original topic."
            }
        
        # --- Intent commit detection ---
        if self.is_intent_commit(message):
            return self._compose_action_reply(message, merchant, category, trigger, customer, conversation_history)
        
        # --- General conversational reply ---
        return self._compose_general_reply(message, merchant, category, trigger, customer, conversation_history, turn_number)
    
    def _compose_action_reply(self, message: str, merchant: dict, category: dict,
                               trigger: Optional[dict], customer: Optional[dict],
                               conversation_history: list) -> dict:
        """Compose action-mode reply when merchant commits."""
        identity = merchant.get("identity", {})
        owner = identity.get("owner_first_name", "")
        languages = identity.get("languages", ["en"])
        
        # Build action context
        trigger_kind = trigger.get("kind", "") if trigger else ""
        active_offers = [o for o in merchant.get("offers", []) if o.get("status") == "active"]
        cust_agg = merchant.get("customer_aggregate", {})
        
        prompt = f"""The merchant just committed with: "{message}"
        
Merchant: {identity.get('name')} (owner: {owner})
Category: {category.get('slug')}
Languages: {languages}
Trigger kind was: {trigger_kind}
Active offers: {json.dumps(active_offers)}
Customer aggregate: {json.dumps(cust_agg)}
Previous conversation: {json.dumps(conversation_history[-3:])}

Compose an ACTION-MODE response (NOT a qualifying question). You are NOW doing the work:
- Confirm you are starting/doing the action
- State what specific thing you are doing (drafting a post, sending a list, booking a slot, etc.)
- Give a concrete deliverable or next step
- End with a single binary confirmation if needed

RESPOND WITH ONLY THIS JSON:
{{"body": "<action message>", "cta": "<binary_confirm_cancel|binary_yes_no|none>", "rationale": "<why this action>"}}"""
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=400,
                system=VERA_SYSTEM,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = response.content[0].text.strip()
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
            "body": f"Great — starting on this right now. I'll have a draft ready in 2 minutes. Reply CONFIRM to send it to your list.",
            "cta": "binary_confirm_cancel",
            "rationale": "Merchant committed to action; executing immediately."
        }
    
    def _compose_general_reply(self, message: str, merchant: dict, category: dict,
                                trigger: Optional[dict], customer: Optional[dict],
                                conversation_history: list, turn_number: int) -> dict:
        """Compose a general contextual reply."""
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

Compose a helpful, specific, category-appropriate reply that:
- Directly addresses what the merchant said
- Advances the conversation toward a useful outcome
- Uses real data from context (not fabricated)
- Ends with one clear next step

RESPOND WITH ONLY THIS JSON:
{{"body": "<reply body>", "cta": "<open_ended|binary_yes_no|none>", "rationale": "<reasoning>"}}"""
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=400,
                temperature=0,
                system=VERA_SYSTEM,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = response.content[0].text.strip()
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
            "body": "Got it — let me work on that and get back to you shortly.",
            "cta": "open_ended",
            "rationale": "General reply fallback"
        }


def _get_prev_topic(conversation_history: list) -> str:
    """Extract the main topic from recent conversation history."""
    if not conversation_history:
        return "your listing"
    
    last_vera = next(
        (h for h in reversed(conversation_history) if h.get("from_role") == "vera" or h.get("from") == "vera"),
        None
    )
    if last_vera:
        body = last_vera.get("body", last_vera.get("msg", ""))
        # Extract first ~50 chars as topic hint
        return body[:60] + "..." if len(body) > 60 else body
    return "the campaign"

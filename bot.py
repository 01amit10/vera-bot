"""
bot.py — Vera's FastAPI HTTP server.

Exposes the 5 required endpoints:
  GET  /v1/healthz
  GET  /v1/metadata
  POST /v1/context
  POST /v1/tick
  POST /v1/reply
"""

import os
import time
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from composer import VeraComposer, ReplyComposer, AUTO_REPLY_PATTERNS, HOSTILE_PATTERNS, INTENT_COMMIT_PATTERNS, OUT_OF_SCOPE_PATTERNS

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("vera")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
COMPOSER_MODEL = os.getenv("COMPOSER_MODEL", "claude-3-5-haiku-20241022")
TEAM_NAME = os.getenv("TEAM_NAME", "Vera-Alpha")
TEAM_MEMBERS = os.getenv("TEAM_MEMBERS", "Amit").split(",")
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "amit@example.com")
BOT_VERSION = "1.0.0"
SUBMITTED_AT = "2026-09-26T12:00:00Z"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Vera Bot", version=BOT_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------
# contexts[(scope, context_id)] = {version: int, payload: dict}
contexts: dict[tuple[str, str], dict] = {}

# conversations[conversation_id] = {merchant_id, customer_id, turns: list, suppressed: bool}
conversations: dict[str, dict] = {}

# suppressed suppression_keys: set of keys → skip on next tick
suppressed_keys: set[str] = set()

# ---------------------------------------------------------------------------
# Lazy-init composers
# ---------------------------------------------------------------------------
_composer: Optional[VeraComposer] = None
_reply_composer: Optional[ReplyComposer] = None


def get_composer() -> VeraComposer:
    global _composer
    if _composer is None:
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not set")
        _composer = VeraComposer(ANTHROPIC_API_KEY, COMPOSER_MODEL)
    return _composer


def get_reply_composer() -> ReplyComposer:
    global _reply_composer
    if _reply_composer is None:
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not set")
        _reply_composer = ReplyComposer(ANTHROPIC_API_KEY, COMPOSER_MODEL)
    return _reply_composer


# ---------------------------------------------------------------------------
# Helper: resolve contexts
# ---------------------------------------------------------------------------

def get_context(scope: str, context_id: str) -> Optional[dict]:
    """Retrieve payload for a (scope, context_id)."""
    entry = contexts.get((scope, context_id))
    return entry["payload"] if entry else None


def resolve_trigger_contexts(trigger: dict) -> tuple[Optional[dict], Optional[dict], Optional[dict], Optional[dict]]:
    """Given a trigger payload, return (category, merchant, trigger_payload, customer)."""
    merchant_id = trigger.get("merchant_id") or trigger.get("payload", {}).get("merchant_id")
    customer_id = trigger.get("customer_id") or trigger.get("payload", {}).get("customer_id")

    merchant = get_context("merchant", merchant_id) if merchant_id else None
    category = None
    if merchant:
        cat_slug = merchant.get("category_slug", "")
        category = get_context("category", cat_slug)

    customer = get_context("customer", customer_id) if customer_id else None
    return category, merchant, trigger, customer


def context_counts() -> dict:
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _) in contexts:
        if scope in counts:
            counts[scope] += 1
    return counts


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ContextBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int = 2


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/v1/healthz")
async def healthz():
    counts = context_counts()
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": counts,
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": TEAM_NAME,
        "team_members": TEAM_MEMBERS,
        "model": COMPOSER_MODEL,
        "approach": (
            "Trigger-kind routing + 4-context structured prompt + Claude at temperature=0. "
            "Auto-reply detection, intent-commit detection, and hostile/out-of-scope routing in reply handler. "
            "Suppression key dedup prevents re-sending same message."
        ),
        "contact_email": CONTACT_EMAIL,
        "version": BOT_VERSION,
        "submitted_at": SUBMITTED_AT,
    }


@app.post("/v1/context")
async def push_context(body: ContextBody):
    valid_scopes = {"category", "merchant", "customer", "trigger"}
    if body.scope not in valid_scopes:
        return JSONResponse(
            status_code=400,
            content={"accepted": False, "reason": "invalid_scope", "details": f"scope must be one of {valid_scopes}"}
        )

    key = (body.scope, body.context_id)
    current = contexts.get(key)

    if current and current["version"] >= body.version:
        # Idempotent: same version is a no-op; higher version we already have → stale
        if current["version"] == body.version:
            # Pure idempotent re-post
            return {
                "accepted": False,
                "reason": "stale_version",
                "current_version": current["version"]
            }
        else:
            return JSONResponse(
                status_code=409,
                content={"accepted": False, "reason": "stale_version", "current_version": current["version"]}
            )

    contexts[key] = {"version": body.version, "payload": body.payload}
    stored_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    logger.info(f"Context stored: {body.scope}/{body.context_id} v{body.version}")

    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": stored_at,
    }


@app.post("/v1/tick")
async def tick(body: TickBody):
    """
    Periodic wake-up. Bot inspects available_triggers and decides what to send.
    Cap: 20 actions per tick.
    """
    actions = []

    for trg_id in body.available_triggers:
        if len(actions) >= 20:
            break

        trigger_entry = contexts.get(("trigger", trg_id))
        if not trigger_entry:
            continue

        trigger = trigger_entry["payload"]

        # Check suppression
        sup_key = trigger.get("suppression_key", "")
        if sup_key and sup_key in suppressed_keys:
            logger.info(f"Suppressed trigger {trg_id} (key={sup_key})")
            continue

        # Check expiry using simulated time from tick request
        expires_at = trigger.get("expires_at")
        if expires_at:
            try:
                exp_dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                # Use the tick's 'now' as simulated time (judge uses historical timestamps)
                tick_now_str = body.now.replace("Z", "+00:00")
                tick_now = datetime.fromisoformat(tick_now_str)
                if tick_now > exp_dt:
                    logger.info(f"Trigger {trg_id} expired (simulated time)")
                    continue
            except Exception:
                pass

        # Resolve all 4 contexts
        category, merchant, _, customer = resolve_trigger_contexts(trigger)

        if not merchant:
            logger.warning(f"No merchant for trigger {trg_id}")
            continue
        if not category:
            logger.warning(f"No category for trigger {trg_id}")
            continue

        # Check if we already have an open conversation for this merchant + trigger
        conv_id = _find_existing_conv(trigger.get("merchant_id", ""), trg_id)
        if conv_id and conversations.get(conv_id, {}).get("suppressed"):
            continue
        
        # Fresh conversation
        conv_id = f"conv_{trigger.get('merchant_id', 'unk')}_{trg_id}_{uuid.uuid4().hex[:8]}"

        try:
            composer = get_composer()
            result = composer.compose(category, merchant, trigger, customer)
        except Exception as e:
            logger.error(f"Compose failed for {trg_id}: {e}")
            continue

        body_text = result.get("body", "")
        if not body_text:
            continue

        # Mark suppression key
        if sup_key:
            suppressed_keys.add(sup_key)

        # Store conversation
        merchant_id = trigger.get("merchant_id", "")
        customer_id = trigger.get("customer_id")
        conversations[conv_id] = {
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "trigger_id": trg_id,
            "turns": [{"from_role": "vera", "message": body_text}],
            "suppressed": False,
        }

        action = {
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": result.get("send_as", "vera"),
            "trigger_id": trg_id,
            "template_name": f"vera_{trigger.get('kind', 'generic')}_v1",
            "template_params": _extract_template_params(result.get("body", ""), merchant),
            "body": body_text,
            "cta": result.get("cta", "open_ended"),
            "suppression_key": result.get("suppression_key", sup_key),
            "rationale": result.get("rationale", ""),
        }
        actions.append(action)
        logger.info(f"Action composed: {conv_id} ({trigger.get('kind')}) → {len(body_text)} chars")

    return {"actions": actions}


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    """
    Handle a reply from the simulated merchant/customer.
    Returns: {action: send|wait|end, body?, cta?, rationale}
    """
    conv_id = body.conversation_id
    message = body.message
    msg_lower = message.lower()

    # Get or create conversation state
    conv = conversations.get(conv_id, {
        "merchant_id": body.merchant_id,
        "customer_id": body.customer_id,
        "trigger_id": None,
        "turns": [],
        "suppressed": False,
    })

    if conv_id not in conversations:
        conversations[conv_id] = conv

    # Check if conversation is suppressed
    if conv.get("suppressed"):
        return {"action": "end", "rationale": "Conversation suppressed; no further messages."}

    # ----------------------------------------------------------------
    # FAST-PATH DETECTION (pure Python, no LLM needed)
    # ----------------------------------------------------------------

    # 1. Hostile / opt-out detection
    if any(p in msg_lower for p in HOSTILE_PATTERNS):
        conv["suppressed"] = True
        if body.merchant_id:
            suppressed_keys.add(f"conv_ended:{body.merchant_id}")
        return {
            "action": "end",
            "rationale": "Merchant expressed opt-out or hostility. Closing conversation; suppressing for 30 days."
        }

    # 2. Auto-reply detection
    is_auto = any(p in msg_lower for p in AUTO_REPLY_PATTERNS)
    auto_count_in_history = sum(
        1 for t in conv["turns"]
        if t.get("from_role") == "merchant" and
        any(p in t.get("message", "").lower() for p in AUTO_REPLY_PATTERNS)
    )
    total_auto = (1 if is_auto else 0) + auto_count_in_history

    if total_auto >= 3:
        conv["suppressed"] = True
        return {
            "action": "end",
            "rationale": f"Auto-reply detected {total_auto}× — no real owner engagement. Closing conversation."
        }
    elif total_auto == 2:
        conv["turns"].append({"from_role": body.from_role, "message": message})
        return {
            "action": "wait",
            "wait_seconds": 86400,
            "rationale": "Auto-reply 2× in a row — owner not at phone. Waiting 24h before retry."
        }
    elif total_auto == 1:
        merchant_id_fast = body.merchant_id or conv.get("merchant_id", "")
        merchant_fast = get_context("merchant", merchant_id_fast) or {}
        owner = merchant_fast.get("identity", {}).get("owner_first_name", "the owner")
        conv["turns"].append({"from_role": body.from_role, "message": message})
        return {
            "action": "send",
            "body": f"Looks like an auto-reply 😊 When {owner} sees this, just reply YES to continue.",
            "cta": "binary_yes_no",
            "rationale": "Detected WhatsApp Business auto-reply; one prompt for the owner to pick up."
        }

    # 3. Out-of-scope detection
    if any(p in msg_lower for p in OUT_OF_SCOPE_PATTERNS):
        conv["turns"].append({"from_role": body.from_role, "message": message})
        prev_topic = _get_prev_topic_from_conv(conv["turns"])
        return {
            "action": "send",
            "body": f"That's outside what I can help with directly — you'd need your CA or specialist for that. Coming back to {prev_topic} — want me to proceed?",
            "cta": "binary_yes_no",
            "rationale": "Out-of-scope request politely declined; redirected to original topic."
        }

    # ----------------------------------------------------------------
    # Add this turn to history (after fast-path checks)
    # ----------------------------------------------------------------
    conv["turns"].append({"from_role": body.from_role, "message": message})

    # Resolve merchant and category
    merchant_id = body.merchant_id or conv.get("merchant_id")
    customer_id = body.customer_id or conv.get("customer_id")
    trigger_id = conv.get("trigger_id")

    merchant = get_context("merchant", merchant_id) if merchant_id else {}
    category = {}
    if merchant:
        cat_slug = merchant.get("category_slug", "")
        category = get_context("category", cat_slug) or {}

    trigger = None
    if trigger_id:
        trigger_entry = contexts.get(("trigger", trigger_id))
        trigger = trigger_entry["payload"] if trigger_entry else None

    customer = get_context("customer", customer_id) if customer_id else None

    try:
        reply_composer = get_reply_composer()
        result = reply_composer.compose_reply(
            conversation_id=conv_id,
            message=message,
            conversation_history=conv["turns"],
            merchant=merchant or {},
            category=category or {},
            trigger=trigger,
            customer=customer,
            turn_number=body.turn_number,
        )
    except Exception as e:
        logger.error(f"Reply compose failed for {conv_id}: {e}")
        result = {
            "action": "send",
            "body": "Got it — give me a moment and I'll follow up shortly.",
            "cta": "open_ended",
            "rationale": "Fallback reply"
        }

    # If ending, mark conversation suppressed
    if result.get("action") == "end":
        conv["suppressed"] = True
        # Mark merchant-level suppression for 30 days if hostile
        if merchant_id:
            suppressed_keys.add(f"conv_ended:{merchant_id}")

    # Add bot reply to history
    if result.get("action") == "send":
        conv["turns"].append({"from_role": "vera", "message": result.get("body", "")})

    logger.info(f"Reply for {conv_id}: action={result.get('action')} turn={body.turn_number}")
    return result


# ---------------------------------------------------------------------------
# Optional teardown (clean state at end of test)
# ---------------------------------------------------------------------------

@app.post("/v1/teardown")
async def teardown():
    global contexts, conversations, suppressed_keys
    contexts.clear()
    conversations.clear()
    suppressed_keys.clear()
    logger.info("State wiped via teardown")
    return {"status": "ok", "message": "State cleared"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_existing_conv(merchant_id: str, trigger_id: str) -> Optional[str]:
    """Find if there's an existing open conversation for this merchant+trigger."""
    for conv_id, conv in conversations.items():
        if (conv.get("merchant_id") == merchant_id and 
            conv.get("trigger_id") == trigger_id and 
            not conv.get("suppressed")):
            return conv_id
    return None


def _get_prev_topic_from_conv(turns: list) -> str:
    """Extract the main topic from recent conversation history."""
    if not turns:
        return "your listing"
    last_vera = next(
        (t for t in reversed(turns) if t.get("from_role") == "vera"),
        None
    )
    if last_vera:
        body = last_vera.get("message", "")
        return (body[:60] + "...") if len(body) > 60 else body
    return "the previous topic"


def _extract_template_params(body: str, merchant: dict) -> list:
    """Extract up to 3 template params from message body for WA template registration."""
    identity = merchant.get("identity", {})
    name = identity.get("owner_first_name") or identity.get("name", "")
    # Split body into ~3 parts as template params
    words = body.split()
    if len(words) <= 10:
        return [name, body, ""]
    mid = len(words) // 2
    p1 = " ".join(words[:mid])
    p2 = " ".join(words[mid:])
    return [name, p1[:100], p2[:100]]


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

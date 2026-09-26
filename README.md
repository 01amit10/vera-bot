# Vera Bot — magicpin AI Challenge

## What this is
A production-quality implementation of **Vera**, magicpin's AI assistant for merchant growth. Built for the [magicpin AI Challenge](https://partners.magicpin.com/vera/ai-challenge/).

## Architecture

```
vera_bot/
├── bot.py          # FastAPI server — all 5 HTTP endpoints
├── composer.py     # Core message composition engine
└── requirements.txt
```

### Core compose flow
```
compose(category, merchant, trigger, customer?) → ComposedMessage
```

1. **Trigger routing** — 25+ trigger kinds mapped to specific composition strategies (strategy dict in `composer.py`). Each strategy tells the LLM *what angle to use*, not *what words to say*.
2. **4-context injection** — Category (voice, digest, peer stats), Merchant (identity, performance, signals, offers, conv history), Trigger (payload, urgency), Customer (relationship, preferences, consent) are all structured into a dense prompt.
3. **Claude at temperature=0** — `claude-3-5-haiku-20241022` for fast, deterministic composition. Same input → same output.
4. **Post-LLM validation** — URL stripping, CTA normalization, JSON parse with fallback.

### Reply handler
Handles multi-turn conversations with 4 detection layers:
- **Auto-reply detection** (15+ canned patterns) → wait → end progression
- **Intent-commit detection** → immediate action mode (never re-qualifies)
- **Hostile/opt-out detection** → graceful end with 30-day suppression
- **Out-of-scope detection** → polite redirect back to original topic

### State management
- In-memory `contexts` dict — idempotent by `(scope, context_id, version)`
- `conversations` dict — tracks turn history per conversation
- `suppressed_keys` set — prevents re-sending same trigger to same merchant

## Model choice
**`claude-3-5-haiku-20241022`** — chosen for:
- Fast P50 latency (~1.5s) comfortably within the 30s timeout
- Strong instruction-following at temperature=0
- Native Hindi-English understanding for code-mix output

## What the judge rewards (and how we chase it)

| Dimension | Our approach |
|---|---|
| Decision quality | Trigger-kind routing strategy; bot chooses not to send if trigger is stale/suppressed |
| Specificity | Prompt explicitly demands ≥2 verifiable numbers/names/dates; checklist in prompt |
| Category fit | Per-category voice rules injected; taboo words listed; tone examples provided |
| Merchant fit | Owner first name, CTR vs peer, active offer titles, customer aggregate, signals all in prompt |
| Engagement compulsion | Strategy dict prescribes which Cialdini lever to use per trigger kind |

## Tradeoffs
- **In-memory state**: Simple but restarts wipe state. Railway keeps the process alive.
- **Single LLM call per compose**: Could add retrieval over digest items for better recall on large catalogs — skipped for latency.
- **No database**: Context is held in RAM. For production Vera, Redis would be better.

## Additional context that would help
- Real merchant conversation history beyond the last 3 turns
- Merchant's WhatsApp session state (24h window open/closed)
- Historical suppression keys across test runs
- Real-time slot availability for booking flows

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
export TEAM_NAME="Your Name"
export CONTACT_EMAIL="you@example.com"
python bot.py
```

## Local test
```bash
# Run judge simulator (after setting keys in judge_simulator.py)
python judge_simulator.py
```

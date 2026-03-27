"""assistant/claude_api.py — Purchasing intelligence assistant powered by Claude.

Serves the partswatch-ai purchasing chat interface.  On every user message:
  1. Fresh live context is pulled from assistant/context_builder.py.
  2. The static business system prompt is combined with the live context.
  3. The full conversation history is sent to Claude.
  4. Claude's response is returned and appended to the in-session history.

Multi-turn conversations are supported for the lifetime of a PurchasingAssistant
instance (one instance per buyer session).  History is in-memory only — it is
not persisted between process restarts.

Usage
-----
    from assistant.claude_api import PurchasingAssistant
    from db.connection import get_client

    assistant = PurchasingAssistant(get_client())
    reply = assistant.chat("What are the most critical stockout risks right now?")
    print(reply)

    # Multi-turn
    reply2 = assistant.chat("Which supplier is the biggest risk for those SKUs?")
    print(reply2)

CLI test
--------
    python -m assistant.claude_api
"""

from __future__ import annotations

import sys
import time
from typing import Any

import anthropic

import config
from assistant.context_builder import build_context
from db.connection import get_client
from utils.logging_config import get_logger, setup_logging

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Model + generation constants
# ---------------------------------------------------------------------------

CLAUDE_MODEL: str = "claude-opus-4-5"

# Maximum tokens Claude may generate per response.
# Purchasing answers are concise operational decisions — 1 024 is generous.
MAX_TOKENS: int = 1_024

# Maximum turns to retain in history before pruning oldest exchanges.
# Each turn = one user + one assistant message = 2 entries.
MAX_HISTORY_TURNS: int = 20


# ---------------------------------------------------------------------------
# Static system prompt
# ---------------------------------------------------------------------------

_STATIC_SYSTEM_PROMPT: str = """\
You are PartsWatch AI, a purchasing intelligence assistant for a \
$100M automotive aftermarket parts distributor with 23 locations across \
Northeast Ohio and approximately 200,000 active SKUs.

Your role is to help purchasing managers and buyers make fast, confident \
decisions. You have access to live data injected below — use it to give \
specific, actionable answers grounded in the numbers.

OPERATIONAL CONTEXT
-------------------
- Two-step distribution model: you supply independent repair shops and \
  fleet accounts, not end consumers.
- SKU tiers: A-class (top 10K by velocity), B-class (next 30K), \
  C-class (~160K remaining). Each tier uses a different forecast model.
- Locations are classified Tier 1 (first-call), Tier 2 (second-call), \
  or Tier 3 (third-call / third-call-store). Third-call locations have \
  lower demand quality — their signals are weighted toward network averages.
- Suppliers are scored green / amber / red. Red means elevated risk; \
  open POs with red-flag suppliers require immediate attention.
- Freeze events (below 20°F) drive spikes in batteries, antifreeze, \
  and cooling system parts. Check weather before any winter stocking decision.
- Reorder types: "po" = new purchase order from supplier; \
  "transfer" = move excess stock from another location internally.

HOW TO RESPOND
--------------
- Be direct and specific. Buyers are busy — lead with the answer, \
  then the supporting data.
- Quote SKU IDs, location IDs, supplier IDs, and numbers from the \
  live context rather than speaking in generalities.
- When data supports a clear recommendation, make it. Do not hedge \
  unnecessarily.
- If a question is outside the live data window or cannot be answered \
  with confidence, say so clearly and suggest what additional data \
  would help.
- Never fabricate SKU details, prices, or supplier information that \
  is not in the live context or established business rules.
- Keep responses concise. Bullet points are preferred for multi-item answers.

LIVE DATA (refreshed on every message)
---------------------------------------
"""


# ---------------------------------------------------------------------------
# PurchasingAssistant
# ---------------------------------------------------------------------------

class PurchasingAssistant:
    """Session-scoped purchasing intelligence assistant.

    One instance per buyer session.  Maintains conversation history for
    the lifetime of the instance.  Context is rebuilt fresh on every call
    to `chat()` so Claude always sees the current database state.

    Args:
        db_client: Active Supabase client from db.connection.get_client().
        model:     Claude model identifier (default: CLAUDE_MODEL).
    """

    def __init__(
        self,
        db_client: Any,
        model: str = CLAUDE_MODEL,
    ) -> None:
        self._db = db_client
        self._model = model
        self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self._history: list[dict[str, str]] = []
        log.info(
            "PurchasingAssistant initialised  model=%s  max_tokens=%d",
            self._model, MAX_TOKENS,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """Send a message and get a response, maintaining conversation history.

        Rebuilds the live Supabase context on every call so Claude sees
        current data regardless of how long the session has been open.

        Args:
            user_message: The buyer's question or request.

        Returns:
            Claude's response as a plain string.

        Raises:
            anthropic.APIError: Propagated from the Anthropic SDK on API failure.
        """
        user_message = user_message.strip()
        if not user_message:
            return "Please enter a question or request."

        t0 = time.perf_counter()

        # Build fresh live context
        log.debug("Building live context …")
        context_str = build_context(self._db)

        # Full system prompt = static business context + live data
        system_prompt = _STATIC_SYSTEM_PROMPT + context_str

        # Append user turn to history
        self._history.append({"role": "user", "content": user_message})

        # Prune history if it exceeds MAX_HISTORY_TURNS (keep most recent)
        max_entries = MAX_HISTORY_TURNS * 2  # user + assistant per turn
        if len(self._history) > max_entries:
            self._history = self._history[-max_entries:]
            log.debug("History pruned to %d entries.", len(self._history))

        # Call Claude
        log.debug(
            "Calling Claude  model=%s  history_turns=%d  context_chars=%d",
            self._model, len(self._history) // 2, len(context_str),
        )
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=MAX_TOKENS,
                system=system_prompt,
                messages=self._history,
            )
        except anthropic.APIError:
            # Remove the user message we just appended so history stays clean
            self._history.pop()
            log.exception("Claude API call failed.")
            raise

        reply = response.content[0].text if response.content else ""

        # Append assistant turn
        self._history.append({"role": "assistant", "content": reply})

        elapsed_ms = (time.perf_counter() - t0) * 1000
        log.info(
            "Chat response  %.0fms  in_tokens=%d  out_tokens=%d  "
            "history_turns=%d",
            elapsed_ms,
            response.usage.input_tokens,
            response.usage.output_tokens,
            len(self._history) // 2,
        )

        return reply

    def reset(self) -> None:
        """Clear conversation history to start a fresh session."""
        self._history.clear()
        log.info("Conversation history cleared.")

    @property
    def turn_count(self) -> int:
        """Number of completed conversation turns (user + assistant pairs)."""
        return len(self._history) // 2


# ---------------------------------------------------------------------------
# CLI entry point — for direct testing from the terminal
# ---------------------------------------------------------------------------

def run_cli() -> int:
    """Interactive REPL for testing the assistant from the terminal.

    Exits on EOF (Ctrl-D) or the commands 'exit' / 'quit'.
    Type 'reset' to clear conversation history mid-session.

    Returns:
        Exit code 0.
    """
    setup_logging()
    log.info("Connecting to Supabase …")
    try:
        db = get_client()
    except Exception:
        log.exception("Failed to connect to Supabase.")
        return 1

    assistant = PurchasingAssistant(db)

    print("\n" + "=" * 60)
    print("  PartsWatch AI — Purchasing Intelligence Assistant")
    print("  Model:", CLAUDE_MODEL)
    print("  Type 'reset' to clear history | 'exit' to quit")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if user_input.lower() == "reset":
            assistant.reset()
            print("[History cleared — starting fresh session.]\n")
            continue

        try:
            reply = assistant.chat(user_input)
        except anthropic.APIError as exc:
            print(f"[API error: {exc}]\n")
            continue

        print(f"\nAssistant: {reply}\n")
        print(f"[Turn {assistant.turn_count}]\n")

    return 0


if __name__ == "__main__":
    sys.exit(run_cli())

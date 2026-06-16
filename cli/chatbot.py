"""Phase 3 — minimal command-line chatbot (with Phase 4 conversation history).

Run with:
    python cli/chatbot.py

Behavior
--------
- On start, loads config (which loads .env). If the API key is missing it
  crashes immediately with a clear message.
- Maintains an in-memory conversation history (system prompt + prior turns).
  Each new user message is sent together with the full history, and the
  assistant's reply is appended to history so the model "remembers" it.
- Supports `/clear` to reset history without restarting the program.
- Loops until the user types `/exit`, `/quit`, or hits Ctrl+C.

This is intentionally the *minimum* working version. No real system prompt
(Phase 5), no UI (Phase 6). Folder renamed from `phase3_cli/` → `cli/`
in Phase 7; no logic changes.
"""

from __future__ import annotations

import sys
from typing import List

from app.config import OPENROUTER_MODEL, iter_api_keys, iter_models
from app.openrouter import OpenRouterError, chat
from app.prompts import DEFAULT_SYSTEM_PROMPT
from app.router import (
    AllSlotsExhaustedError,
    ModelRouter,
    NoFreeModelConfiguredError,
    build_from_config,
)

# Commands the user can type to control the loop without sending a question.
EXIT_COMMANDS = {"/exit", "/quit", ":q"}
# Command the user can type to wipe conversation history mid-session.
CLEAR_COMMANDS = {"/clear", ":c"}


def _print_banner(router: ModelRouter) -> None:
    """Print a friendly welcome message."""
    print("=" * 60)
    print("  AI Security Chatbot — CLI (with multi-key router)")
    print(f"  Default model: {OPENROUTER_MODEL}")
    print(
        f"  Router pool: {router.healthy_slot_count()} slot(s) — "
        f"{len(list(iter_api_keys()))} key(s) × "
        f"{len(router.slot_labels()) // max(len(list(iter_api_keys())), 1)} model(s)"
    )
    print("  Type your cybersecurity question and press Enter.")
    print("  Commands: /clear to reset history, /exit to leave.")
    print("=" * 60)


def _read_user_input() -> str | None:
    """Read one line of input from the user.

    Returns None if the input was an exit command (caller should break).
    Returns the cleaned user message otherwise.
    Returns "" for empty input (caller should skip and loop again).
    """
    try:
        raw = input("\nYou> ").strip()
    except (EOFError, KeyboardInterrupt):
        # Ctrl+Z / Ctrl+C -> treat as exit.
        print()  # newline after the ^C echo
        return None

    if raw.lower() in EXIT_COMMANDS:
        return None
    return raw


def _build_messages(history: List[dict[str, str]], user_input: str) -> list[dict[str, str]]:
    """Build the message list for the next API call.

    The model has no memory of its own — "remembering" the conversation is
    just re-sending the prior turns inside the `messages` list. The caller
    is responsible for keeping `history` up to date.

    The system prompt is always pinned at the front of every call so the
    model's behavior stays consistent across turns.
    """
    return history + [{"role": "user", "content": user_input}]


def _run_loop() -> int:
    """Main interaction loop. Returns a process exit code."""
    # Build the multi-key, multi-model router once for the lifetime of
    # the CLI process. Slot health (disabled, last_error) persists
    # across turns so a 429 on key A doesn't get retried on key A
    # again on the next user message — the cursor advances.
    try:
        router = build_from_config(list(iter_api_keys()), list(iter_models()))
    except (NoFreeModelConfiguredError, ValueError) as cfg_exc:
        print(
            "Fatal: router is not configured. Set OPENROUTER_API_KEY "
            "and optionally OPENROUTER_API_KEY_2..5 and OPENROUTER_MODELS "
            "in your environment or .env file.",
            file=sys.stderr,
        )
        print(f"Underlying error: {cfg_exc}", file=sys.stderr)
        return 2

    _print_banner(router)

    # History starts with the system prompt. We keep it as a plain list and
    # mutate it in place; that's the simplest model for a single-user CLI.
    history: list[dict[str, str]] = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
    ]

    while True:
        user_input = _read_user_input()
        if user_input is None:        # exit command or Ctrl+C
            print("Goodbye.")
            return 0
        if not user_input:             # empty line — just prompt again
            continue
        if user_input.lower() in CLEAR_COMMANDS:
            history = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
            print("[history cleared]")
            continue

        messages = _build_messages(history, user_input)

        try:
            reply = router.chat(messages)
        except AllSlotsExhaustedError as exc:
            # Every slot in the pool failed in turn — distinct from a
            # single-model error. The user should know the whole pool
            # is dead, not just one slot, and the loop should NOT keep
            # hammering a dead pool on the next turn. Bail with a
            # non-zero exit so an outer script can detect it.
            print(
                "\n[error] All router slots are exhausted. Every "
                "(key, model) pair failed — most likely the daily "
                "per-account free-tier cap has been hit on every account.",
                file=sys.stderr,
            )
            print(f"Tried: {', '.join(router.slot_labels())}", file=sys.stderr)
            return 3
        except OpenRouterError as exc:
            # A single-slot error bubbled out of router.chat (the
            # router normally handles these internally; reaching here
            # means something unexpected). Don't crash the whole loop
            # on a single failed call. Log, continue.
            print(f"\n[error] {exc}", file=sys.stderr)
            continue

        # Persist this exchange into history BEFORE printing, so the next
        # turn sees it.
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})

        print(f"\nAssistant> {reply}")
        print(f"  [history: {len(history)} messages, "
              f"~{sum(len(m['content']) for m in history)} chars]")

    # Unreachable, but keeps the type-checker happy.
    return 0


def main() -> int:
    """Entry point. Wraps the loop with a friendly top-level error net."""
    try:
        return _run_loop()
    except OpenRouterError as exc:
        # Configuration errors (e.g. missing API key) land here.
        print(f"Fatal: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

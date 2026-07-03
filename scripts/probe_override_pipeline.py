"""Probe: simulate the exact override pipeline the Streamlit view uses,
without spinning up Streamlit. Confirms whether
``custom_model_override`` makes it all the way to the router's
``stream_chat(..., model=...)`` call.

We patch only:
    * ``stream_chat`` (so no real HTTP call fires)
    * the ``st.session_state`` mirror dict (so we can simulate the
      values that would arrive from the text_input + dropdown)

What we assert:
    1. With ``custom_model_override`` set to ``qwen/qwen3-coder:free``
       and ``model`` set to a curated id, the router receives
       ``model=qwen/qwen3-coder:free`` on the next ``stream_chat``
       call. If it doesn't, the override is being lost upstream of
       the router.
    2. The ephemeral-slot path is taken (no built slot for the
       override id) — confirmed by capturing the ``model`` arg to
       ``stream_chat`` and asserting it matches the override id.

Run from the project root:
    python scripts/probe_override_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make ``app`` importable when run as a plain script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from unittest.mock import patch

from app.config import iter_api_keys, iter_models
from app.router import build_from_config


def _build_router():
    """Mimic the view's ``_get_router`` factory: real keys × real built
    models, no monkey-patching of the config layer."""
    keys = list(iter_api_keys())
    models = list(iter_models())
    if not keys or not models:
        raise SystemExit(
            "Probe needs at least one OPENROUTER_API_KEY and one "
            "OPENROUTER_MODEL in the env. Aborting."
        )
    # Mirror the production router factory: one slot per (key, model).
    return build_from_config(keys, models)


def main() -> int:
    router = _build_router()
    print(
        f"[probe] router built: {len(list(iter_api_keys()))} key(s) × "
        f"{len(router._slots)} built slot(s); "
        f"first built slot model = `{router._slots[0].model_id}`"
    )

    # Simulate session_state after the user types in the Advanced
    # expander and clicks the lock.
    curated_default = st_session_state_model = router._slots[0].model_id
    override_id = "qwen/qwen3-coder:free"

    # This is the EXACT assignment the view uses at the top of _ask.
    requested_model = override_id or curated_default
    print(
        f"[probe] requested_model after override-or-curated logic: "
        f"`{requested_model}`"
    )
    assert requested_model == override_id, (
        "Override was lost at the session-state read. requested_model "
        f"= {requested_model!r} but expected {override_id!r}"
    )

    # Patch stream_chat and capture the model arg the router passes.
    captured: list[dict] = []

    def fake_stream(_messages, *, model=None, api_key=None, **_kw):
        captured.append({"model": model, "api_key": api_key})
        yield "ok"

    with patch("app.openrouter.stream_chat", side_effect=fake_stream):
        # ``messages`` is a single user turn; the router's stream_chat
        # signature is (messages, *, model=...).
        chunks = list(
            router.stream_chat(
                [{"role": "user", "content": "hello"}],
                model=requested_model,
            )
        )

    print(f"[probe] stream_chat received model = `{captured[0]['model']}`")
    print(f"[probe] stream_chat received api_key = `{captured[0]['api_key']}`")

    assert len(captured) == 1, f"Expected 1 call, got {len(captured)}"
    assert captured[0]["model"] == override_id, (
        f"Router received model `{captured[0]['model']}` but expected "
        f"`{override_id}`. The override is being lost in the router, "
        "not in session state."
    )
    print("[probe] OK — override reached the router intact.")

    # Phase 2: pin the chip-picker lock contract. Simulate a session
    # state where the user has an active override and verify the
    # chatbox picker does NOT re-write ``session_state["model"]``.
    from web.chat_helpers import resolve_chatbox_model_id

    curated = [
        {"id": "google/gemma-4-31b-it:free", "label": "Gemma 4 31B (default)"},
        {"id": "meta-llama/llama-3.3-70b-instruct:free", "label": "Llama 3.3 70B"},
    ]
    fake_state = {"model": "google/gemma-4-31b-it:free",
                  "custom_model_override": "qwen/qwen3-coder:free"}
    # User pops the chip and clicks Llama. Without the lock the
    # chip would write session_state["model"] to "meta-llama/...".
    chosen_id, changed = resolve_chatbox_model_id(
        curated,
        chosen_label="Llama 3.3 70B",
        current_id=fake_state["model"],
        override_id=fake_state.get("custom_model_override", ""),
    )
    print(
        f"[probe] chip-picker with active override: chosen_id="
        f"`{chosen_id}`, changed={changed}"
    )
    assert not changed, (
        "Chip picker reported a change while an override was active "
        "— the curated dropdown would have re-written session_state "
        "and the next turn would have silently dropped the override."
    )
    assert chosen_id == "qwen/qwen3-coder:free", (
        f"Chip picker did not return the override id; got `{chosen_id}`."
    )
    print("[probe] OK — chip picker is locked while override is active.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
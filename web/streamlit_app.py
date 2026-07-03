"""Web UI for the AI Security Chatbot (was Phase 6, settled in Phase 7).

The architecture is a thin shell:

    user -> this file (view) -> web.chat_helpers (UI logic)
                                       -> app.openrouter.chat (engine)
                                                -> OpenRouter -> LLM

The view file does three things:
    1. Render: header, sidebar, chat bubbles, controls.
    2. State:  manage st.session_state["messages"] and settings.
    3. Wire:   call the pure helpers and the engine at the right moments.

Everything that can be tested without a browser lives in chat_helpers.py.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# --- sys.path bootstrap -------------------------------------------------------
# Streamlit's `streamlit run web/streamlit_app.py` only prepends the SCRIPT's
# directory (here: `web/`) to `sys.path[0]`. It does NOT prepend the project
# root. So `from app.config import ...` will raise `ModuleNotFoundError: No
# module named 'app'` whenever the worker's cwd is not the project root, or
# whenever the empty string (cwd sentinel) has been removed from sys.path.
# We compute the project root from `__file__` and add it to sys.path
# defensively. This is a no-op when the project root is already on path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from app.config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    OPENROUTER_MODEL,
    iter_api_keys,
    iter_models,
    validate_free_model_id,
    InvalidFreeModelIdError,
)
from app.openrouter import (
    OpenRouterError,
    OpenRouterRateLimitError,
    chat,
    stream_chat,
)
from app.router import (
    AllSlotsExhaustedError,
    ModelRouter,
    NoFreeModelConfiguredError,
    RouterError,
    build_from_config,
)

# Absolute import (not `from .chat_helpers import ...`) because Streamlit runs
# this file as a top-level module under `streamlit run web/streamlit_app.py`,
# which means there is no parent package and a relative import would raise
# `ImportError: attempted relative import with no known parent package`.
# The CLI and tests use the same form (`from web.chat_helpers import ...`)
# and the helper module is exported through `web/__init__.py`.
from web.chat_helpers import (
    DEFAULT_MAX_HISTORY_MESSAGES,
    ChatMessage,
    _active_system_prompt,
    _bubble_alignment,
    _build_messages,
    _coerce_message_text,
    _copy_button_html,
    _copy_button_html_for_bubble,
    _count_chars,
    _format_chat_timestamp,
    _friendly_error_message,
    _render_copy_button_for_bubble,
    _serialize_for_download,
    _truncate_history,
    build_user_turn_content,
    build_user_turn_text,
    consume_stop_flag,
    parse_recon_command,
    resolve_chatbox_model_id,
    select_model_for_request,
    stream_vision_turn_with_fallback,
    vision_timeout_seconds,
)
from app import file_processor as _file_processor
from app import storage as _storage
from app.storage import (
    append_message as _append_message,
    create_chat as _create_chat,
    get_chat as _get_chat,
    init_db as _init_db,
    list_chats as _list_chats,
    load_messages as _load_messages,
    soft_delete_chat as _soft_delete_chat,
)
from app.config import model_supports_vision
from app.recon.orchestrator import (
    TargetBlockedError,
    run_recon,
    stream_recon,
)
from app.recon.report import (
    render_report_json,
    render_report_markdown,
)

# --- Media-file storage safety net -------------------------------------------
# Streamlit's ``st.chat_input(accept_file="multiple")`` registers every
# uploaded file in the per-session ``MemoryMediaFileStorage`` and ships
# the file's media ID to the browser. The browser then echoes that ID
# back on every subsequent rerun so the input can re-display the file
# the user previously attached. The catch: ``MemoryMediaFileStorage`` is
# an in-memory dict on the *server* process. If the server restarts
# (a Ctrl-C + relaunch, a code change that triggers ``runOnSave``, a
# Streamlit hot-reload, etc.) and the user keeps the tab open, the
# browser still holds the old media IDs, the chat input tries to
# resolve them against the new server's empty store, and Streamlit
# raises ``MediaFileStorageError`` from inside the widget's render
# call. The user-facing symptom is a server-side traceback ("Bad
# filename '...txt'. (No media file with id '...')") and a blank page
# on every reload.
#
# The structural fix is a one-shot guard around the chat-input render:
# if the underlying storage is missing the file, we (a) show a friendly
# banner, (b) clear any pending input state so the next render starts
# from a clean slate, and (c) fall back to a plain text ``st.chat_input``
# for the rest of the session. The guard is *only* triggered when the
# server has lost the file — a normal in-process rerun still uses the
# full file-attached chat input with no behavioural change.
try:
    from streamlit.runtime.memory_media_file_storage import (
        MediaFileStorageError as _MediaFileStorageError,
    )
except ImportError:  # pragma: no cover - older Streamlit shape
    class _MediaFileStorageError(Exception):  # type: ignore[no-redef]
        """Fallback for Streamlit versions that don't expose the class."""


#: Per-session flag set by the media-file guard when a stale file ID
#: is detected. The next render downgrades the chat input to text-only
#: so the same exception does not fire on every rerun. Reset by the
#: "Reset attachments" button below.
_SESSION_STATE_MEDIA_ERROR = "_media_file_storage_error"


# --- Page configuration -------------------------------------------------------

st.set_page_config(
    page_title="SecMentor — AI-Powered Cybersecurity Platform",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Chat-history helpers (PR-C sidebar) -------------------------------------
# These three top-level callables are the only entry points the sidebar
# widgets use to mutate the chat-history state. They are intentionally
# module-level (not nested) so ``st.button(..., on_click=_new_chat)`` can
# resolve them by name. All three are best-effort: if SQLite is unwritable
# they surface a one-shot ``st.warning`` rather than raising, so a wedged
# DB never blocks the user from chatting.
def _new_chat() -> None:
    """Start a brand-new chat.

    Clears ``active_chat_id`` (forcing pass 1 to call ``_create_chat``
    on the next user turn) and invalidates the sidebar cache so the
    list refreshes on the next render.
    """
    st.session_state["active_chat_id"] = None
    st.session_state["chats"] = _list_chats(limit=20)


def _open_chat(chat_id: str) -> None:
    """Switch the active chat to ``chat_id`` and replay its history.

    Replay happens by re-binding ``st.session_state["messages"]`` from
    the storage layer — the history-render loop at the top of the
    script then paints every turn.
    """
    st.session_state["active_chat_id"] = chat_id
    try:
        _messages = _load_messages(chat_id, limit=200)
    except Exception as exc:  # noqa: BLE001
        st.warning(
            f"Could not load that chat's history: {exc}", icon="⚠️"
        )
        return
    # Reset the in-memory transcript to a single system turn followed
    # by the persisted turns. The system turn is supplied by the
    # existing ``_build_messages`` call inside ``_ask`` — here we just
    # seed the user-visible list with the persisted turns.
    st.session_state["messages"] = [
        {"role": "system", "content": "loaded from disk"}
    ] + [
        {"role": m["role"], "content": m["content"]}
        for m in _messages
    ]


def _soft_delete_chat(chat_id: str) -> None:
    """Soft-delete ``chat_id`` and clear the active pointer if needed.

    Soft-delete preserves the row in the ``chats`` table (sets
    ``deleted_at``) so an admin tool could undelete. The sidebar's
    next ``_list_chats`` call already filters out ``deleted_at IS NOT
    NULL`` rows, so the deleted chat vanishes from the UI without a
    full page reload.

    Note: the storage function is reached through the ``_storage``
    module alias (``from app import storage as _storage``) — the
    ``from app.storage import soft_delete_chat as _soft_delete_chat``
    binding is intentionally shadowed by this helper, which is the
    one Streamlit's ``on_click`` resolves by name.
    """
    try:
        _storage.soft_delete_chat(chat_id)
    except Exception as exc:  # noqa: BLE001
        st.warning(
            f"Could not delete that chat: {exc}", icon="⚠️"
        )
        return
    if st.session_state.get("active_chat_id") == chat_id:
        st.session_state["active_chat_id"] = None
        st.session_state["messages"] = []
    # Invalidate the sidebar cache.
    st.session_state["chats"] = _list_chats(limit=20)


# --- Custom CSS for a production-grade cybersecurity platform ----------------
# The look-and-feel is inspired by Microsoft Security Copilot, CrowdStrike
# Falcon, Palo Alto Cortex, Datadog, and GitHub Enterprise: deep navy/slate
# surfaces, restrained blue accents, no neon, no glow, no ChatGPT-isms.
# Layout density is driven by a CSS class on the root container
# (`.layout-compact` / `.layout-standard` / `.layout-wide` / `.layout-full`)
# toggled from the sidebar.
#
# The stylesheet lives in ``web/styles.css`` (single source of truth) so
# designers can edit CSS without touching Python. We load it lazily, wrap
# the contents in a single ``<style>`` tag, and inject it through
# ``st.markdown(..., unsafe_allow_html=True)`` so the rules apply to the
# entire page (Streamlit hoists the ``<style>`` element out of the
# markdown container at render time).
#
# ``_STYLESHEET_PATH`` is resolved relative to this file so it works
# whether the app is launched with ``streamlit run`` from the project
# root or from ``web/``.

_STYLESHEET_PATH = Path(__file__).resolve().parent / "styles.css"


@st.cache_data(show_spinner=False)
def _load_stylesheet() -> str:
    """Read ``web/styles.css`` and wrap it in a ``<style>`` tag.

    Returns the payload as a single string. The result is cached for
    the life of the Streamlit process — the file is read once, on the
    first rerun, and never touched again. This keeps the cost of the
    injection at roughly the cost of a single ``open()`` call.
    """
    raw = _STYLESHEET_PATH.read_text(encoding="utf-8").strip()
    # If the file already starts with ``<style>``, return as-is. This
    # keeps a hand-edited header (e.g. ``<!-- @import ... -->``) working.
    if raw.lower().startswith("<style"):
        return raw
    return f"<style>\n{raw}\n</style>"


st.markdown(_load_stylesheet(), unsafe_allow_html=True)

# Per-bubble copy buttons are rendered inline by
# ``_render_copy_button_for_bubble`` — one ``st.components.v1.html`` call
# per assistant message. Each iframe contains the reply text and a
# self-contained "📋 Copy" button, so no shared init script or delegated
# listener is needed (and the cross-origin sandbox on Streamlit's
# component iframe cannot interfere with the click handler).


# --- Constants ---------------------------------------------------------------

# Curated set of free OpenRouter models. The `id` is what we send to the
# API; the rest is just metadata for the sidebar dropdown. Keeping this
# list small and hand-picked means a broken/paid model cannot leak in
# by accident. If a `id` here stops working, just delete the row.
#
# Roles are loose categories so the UI can group them. They are not
# enforced by the engine — pick whichever model you want for any prompt.
FREE_MODEL_CHOICES: list[dict[str, str]] = [
    {
        "id": "google/gemma-4-31b-it:free",
        "label": "Gemma 4 31B (default)",
        "role": "Balanced",
        "blurb": "Strong reasoning, refusal-aware. 262K context.",
    },
    {
        "id": "nvidia/nemotron-nano-9b-v2:free",
        "label": "Nemotron Nano 9B",
        "role": "Quick",
        "blurb": "Small and fast. Best for short factual Q&A.",
    },
    {
        "id": "openai/gpt-oss-20b:free",
        "label": "GPT-OSS 20B",
        "role": "Reasoning",
        "blurb": "OpenAI's open-weight MoE. Well-rounded answers.",
    },
    {
        "id": "qwen/qwen3-coder:free",
        "label": "Qwen3 Coder 480B",
        "role": "Coder",
        "blurb": "Specialist for code snippets and config audits.",
    },
    {
        "id": "meta-llama/llama-3.3-70b-instruct:free",
        "label": "Llama 3.3 70B",
        "role": "Backup",
        "blurb": "Permissive fallback when others refuse. 131K context.",
    },
    # --- Vision-capable models -----------------------------------------
    # As of mid-2026, the only free-tier model on OpenRouter that
    # accepts image inputs and returns 200 is nemotron-nano-12b-v2-vl.
    # Every other vision candidate (llama-3.2-vision, qwen2.5-vl,
    # gemma-3, mistral-small-3.1, gemini-2.0-flash-exp) returns 404 on
    # the free tier. The auto-swap in ``select_model_for_request``
    # uses the *first* id in this list, so as long as there is exactly
    # one row, image uploads route to it deterministically. If more
    # free vision models come online, append them below and the
    # allow-list test in tests/test_files.py will need a matching
    # update.
    {
        "id": "nvidia/nemotron-nano-12b-v2-vl:free",
        "label": "Nemotron Nano 12B VL (vision)",
        "role": "Vision",
        "blurb": "Free vision model. Reads screenshots, diagrams, photos. 128K context.",
    },
    # Extra rotation fodder — the router cycles through every (key, model)
    # pair in the pool, so adding more ids is the cheap way to stretch the
    # per-account daily cap. Hand-picked to stay :free-only; if a row stops
    # working, just delete it.
    {
        "id": "mistralai/mistral-small-3.2-24b-instruct:free",
        "label": "Mistral Small 3.2 24B",
        "role": "Balanced",
        "blurb": "Solid generalist with a calmer refusal curve.",
    },
    {
        "id": "deepseek/deepseek-chat-v3.1:free",
        "label": "DeepSeek Chat V3.1",
        "role": "Reasoning",
        "blurb": "Strong step-by-step reasoning. 128K context.",
    },
    {
        "id": "z-ai/glm-4.5-air:free",
        "label": "GLM 4.5 Air",
        "role": "Quick",
        "blurb": "Lightweight, low-latency answers. Good for short drills.",
    },
]

# The model selected in the sidebar on a fresh session. Must be an id
# present in FREE_MODEL_CHOICES (the session_state init also defensively
# re-binds it if not).
DEFAULT_SELECTED_MODEL_INDEX: int = 0


EXAMPLE_PROMPTS: list[str] = [
    "Explain the structure of a SQL injection attack and what defeats it.",
    "What are the structural red flags of SSRF in a Python request handler?",
    "Walk me through the OWASP Top 10 categories and how they map to defenses.",
    "Why is putting secrets in environment variables only half the answer?",
    "Compare SAST vs DAST in the secure SDLC and where each catches what.",
    "What is prompt injection, in structural terms, and how do I detect it in logs?",
]


# --- Router (multi-key, multi-model) ----------------------------------------
# The router is built ONCE per Streamlit process (via @st.cache_resource) and
# reused across every rerun. That's the right scope for a long-lived pool
# of (key, model) slots with their own per-slot health state: rebuilding on
# every rerun would re-enable disabled slots and reset backoff counters.
# `show_spinner=False` keeps the cached call quiet — the spinner is shown
# at the call site, not at the factory.

@st.cache_resource(show_spinner=False)
def _get_router() -> ModelRouter:
    """Build the multi-key, multi-model router from the current env config.

    The Cartesian product of `iter_api_keys()` × `iter_models()` becomes the
    router's slot pool. If either iterator is empty, this raises
    `NoFreeModelConfiguredError` so the caller can surface a friendly
    message instead of silently falling back to a paid model.
    """
    return build_from_config(list(iter_api_keys()), list(iter_models()))


# --- Session state initialization -------------------------------------------

def _init_state() -> None:
    """Seed session state on first run."""
    # The web UI default teaching mode. The CTF / Lab mentor (SecMentor)
    # prompt is the default for the web UI so a learner landing on the
    # page gets the wider, lab-scoped teaching persona from the first
    # turn. The CLI (`cli/chatbot.py`) keeps importing
    # `app.prompts.DEFAULT_SYSTEM_PROMPT`, which still points at the
    # conservative four-pillar defensive prompt — a deliberate split,
    # see the docstring at the top of `app/prompts.py`.
    _DEFAULT_TEACHING_MODE: str = "mentor"
    if "messages" not in st.session_state:
        # The active system prompt is chosen by `_active_system_prompt`
        # from the teaching_mode below, so a fresh session starts with
        # the prompt that matches the toggle.
        st.session_state["messages"] = [
            {
                "role": "system",
                "content": _active_system_prompt(
                    {"teaching_mode": _DEFAULT_TEACHING_MODE}
                ),
            }
        ]
    # Teaching mode toggle (defensive vs. CTF/lab mentor). The helper
    # `_active_system_prompt` looks up the matching prompt constant;
    # unknown / missing values always fall back to the defensive one
    # (fail-closed safety property, see tests/test_smoke.py).
    if "teaching_mode" not in st.session_state:
        st.session_state["teaching_mode"] = _DEFAULT_TEACHING_MODE
    # The sidebar swap block tracks the *previous* teaching mode in a
    # separate key so the live `key="teaching_mode"` widget doesn't
    # poison the comparison. Seed it here so a fresh session starts
    # with previous == current and no spurious swap fires.
    if "teaching_mode_previous" not in st.session_state:
        st.session_state["teaching_mode_previous"] = st.session_state["teaching_mode"]
    if "model" not in st.session_state:
        # Prefer the curated list's first entry as the UI default; fall
        # back to whatever the engine is configured with if the curated
        # list is somehow empty.
        st.session_state["model"] = (
            FREE_MODEL_CHOICES[DEFAULT_SELECTED_MODEL_INDEX]["id"]
            if FREE_MODEL_CHOICES
            else OPENROUTER_MODEL
        )
    if "temperature" not in st.session_state:
        st.session_state["temperature"] = DEFAULT_TEMPERATURE
    if "max_tokens" not in st.session_state:
        # Cap is already 1024 in app.config; the slider below lets the user
        # raise it if they want a longer answer.
        st.session_state["max_tokens"] = DEFAULT_MAX_TOKENS
    if "max_history" not in st.session_state:
        st.session_state["max_history"] = DEFAULT_MAX_HISTORY_MESSAGES
    # Concise mode prepends a one-line instruction to the system prompt
    # so the model gives shorter answers. Big win for perceived latency
    # on free-tier models.
    if "concise" not in st.session_state:
        st.session_state["concise"] = True
    # In-session response cache: (model, temperature, max_tokens, prompt)
    # -> reply. Re-asking the same question is instant, no network call.
    if "response_cache" not in st.session_state:
        st.session_state["response_cache"] = {}

    # --- Chat history (persistent across reloads via SQLite) -------------
    # ``active_chat_id`` is the FK into the ``chats`` table for the chat
    # the user is currently viewing. ``None`` means "no chat yet — create
    # one on the first user message". ``chats`` is the sidebar's cached
    # list (refreshed lazily when the user opens the sidebar widget).
    # ``recon_scope_token`` is an optional override of
    # ``DEFAULT_RECON_SCOPE_TOKEN``; ``None`` means "use the default".
    # We intentionally do NOT eagerly create a chat here — the first
    # user message of a session will create one and store the user turn.
    if "active_chat_id" not in st.session_state:
        st.session_state["active_chat_id"] = None
    if "chats" not in st.session_state:
        st.session_state["chats"] = []
    if "recon_scope_token" not in st.session_state:
        st.session_state["recon_scope_token"] = None
    # Initialise the SQLite schema once per session. The call is cheap
    # (``CREATE TABLE IF NOT EXISTS``), idempotent, and swallows errors
    # so a corrupt DB does not crash the whole page on import. The chat
    # UI degrades gracefully — the user just loses history — if init
    # fails; we surface a one-time warning so they know why.
    if "db_initialised" not in st.session_state:
        try:
            _storage.init_db()
            st.session_state["db_initialised"] = True
            st.session_state.setdefault("db_init_warning", None)
        except Exception as exc:  # noqa: BLE001
            st.session_state["db_initialised"] = False
            st.session_state["db_init_warning"] = (
                f"Chat history disabled — storage init failed: {exc}"
            )
    # Last call's elapsed seconds, shown in the status line.
    if "last_elapsed" not in st.session_state:
        st.session_state["last_elapsed"] = None
    # Show a small "You" / "SecMentor" label above each bubble so the user
    # can tell at a glance which side said what. ON by default — that's
    # what every modern chat UI does.
    if "show_role_labels" not in st.session_state:
        st.session_state["show_role_labels"] = True
    # Layout density mode — drives a CSS class on <body> via a small
    # st.markdown in the sidebar block. See ``web/styles.css`` for token
    # overrides. The default ("standard") matches the original 920px
    # max-width so existing layouts look identical on first load.
    if "layout_mode" not in st.session_state:
        st.session_state["layout_mode"] = "standard"


_init_state()


# --- Sidebar -----------------------------------------------------------------
# All sidebar widgets live in ``_render_sidebar`` so main() stays linear and
# the layout is easy to reason about. The new layout is card-based (CSS in
# ``web/styles.css``: .sm-card / .sm-card-title / .sm-pill / .sm-chat-row)
# and orders sections by user value:
#
#   1. Brand block
#   2. CORE PICKS  — Layout, Teaching mode, Model, Display, Advanced
#   3. Try a question
#   3. RECON HELP   (always visible — discoverable, not buried)
#   4. CONVERSATION — New chat + Download transcript
#   5. CHAT HISTORY (always open, no expander)
#   7. OVERVIEW     — four-pillar summary and where-to-practice legally
def _render_sidebar() -> None:
    """Render the full sidebar — brand + cards + chat history + recon help."""

    # --- Brand -----------------------------------------------------------------
    st.markdown(
        """
        <div class="sm-brand">
          <span class="logo">🛡</span>
          <div>
            <div class="name">SecMentor</div>
            <div class="tag">AI Security Platform</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Cybersecurity learning & analysis")

    # =========================================================================
    # Card 1 — CORE PICKS: layout, teaching mode, model, display, advanced
    # =========================================================================
    st.markdown(
        '<div class="sm-card">'
        '<div class="sm-card-title"><span class="dot"></span>Core picks</div>'
        '<div class="sm-card-sub">The three controls that change how the '
        "chat feels and what it's allowed to teach.</div>",
        unsafe_allow_html=True,
    )

    # Layout mode toggle. Drives a class on <body> (`.layout-compact`,
    # `.layout-standard`, `.layout-wide`, `.layout-full`) so the CSS
    # tokens in ``web/styles.css`` pick the right container width, bubble
    # padding, sidebar density, and hero size. Streamlit does not let
    # us set attributes on <body> declaratively, so we ship a tiny
    # client-side script that runs on every page load and applies the
    # class from `localStorage`. The script is idempotent: calling it
    # twice with the same value is a no-op.
    _LAYOUT_OPTIONS: list[str] = ["compact", "standard", "wide", "full"]
    _LAYOUT_LABELS: dict[str, str] = {
        "compact":  "Compact",
        "standard": "Standard",
        "wide":     "Wide",
        "full":     "Full width",
    }
    _LAYOUT_HELP: dict[str, str] = {
        "compact":  "720px column · tighter spacing · denser chat",
        "standard": "920px column · balanced (default)",
        "wide":     "1180px column · room for side panels",
        "full":     "Edge-to-edge · maximum density",
    }
    _previous_layout = st.session_state.get("layout_mode", "standard")
    st.markdown(
        '<div style="font-size:0.78rem;color:#94a3b8;margin:0.35rem 0 0.25rem 0;">'
        "Layout density</div>",
        unsafe_allow_html=True,
    )
    _chosen_layout = st.radio(
        "Layout density",
        options=_LAYOUT_OPTIONS,
        format_func=lambda key: _LAYOUT_LABELS.get(key, key),
        index=_LAYOUT_OPTIONS.index(_previous_layout)
        if _previous_layout in _LAYOUT_OPTIONS
        else 1,
        key="layout_mode",
        label_visibility="collapsed",
        help="Adjusts container width, sidebar spacing, and chat density. "
             "Functionality is unchanged.",
    )
    st.caption(_LAYOUT_HELP.get(_chosen_layout, ""))
    # Persist + apply. The script applies the class on every load and
    # also removes the other three so CSS specificity is clean.
    st.markdown(
        f"""
        <script>
          (function() {{
            var mode = "{_chosen_layout}";
            try {{ localStorage.setItem("secmentor.layout", mode); }} catch (e) {{}}
            var cls = "layout-" + mode;
            ["layout-compact","layout-standard","layout-wide","layout-full"].forEach(function(c) {{
              document.body.classList.remove(c);
            }});
            document.body.classList.add(cls);
          }})();
        </script>
        """,
        unsafe_allow_html=True,
    )

    # Teaching mode radio. When the choice changes, swap the system
    # message in place so the next model call uses the new scope —
    # no need to start a new chat. Streamlit's default already writes
    # the new value to ``st.session_state["teaching_mode"]`` before
    # this body runs on the rerun-after-click, so we read the
    # "previous" mirror key (updated *after* the swap block below)
    # to detect a real change. For a fresh session we fall back to
    # the live key — which is initialised to "mentor" by _init_state,
    # matching the fallback string here so a brand-new session with
    # no previous key is still self-consistent.
    _TEACHING_OPTIONS: list[str] = ["mentor", "defensive"]
    _TEACHING_LABELS: dict[str, str] = {
        "defensive": "🛡️  Defensive (4 pillars)",
        "mentor": "🎯  CTF / Lab mentor",
    }
    _TEACHING_HELP: dict[str, str] = {
        "defensive": (
            "Concept-level teaching only. Refuses working exploit code, "
            "malware, and payloads against real systems. Safest default."
        ),
        "mentor": (
            "Unlocks CTF/lab scope: HackTheBox, TryHackMe, PortSwigger, "
            "DVWA, WebGoat, your own VMs. May produce runnable exploit "
            "snippets framed for the lab, always with the defensive "
            "countermeasure. Still refuses real-target payloads, WAF/EDR/"
            "MFA bypasses, and brand-new malware. Recorded as "
            "Decision 6 in docs/technical_write_up.md."
        ),
    }
    _previous_mode = st.session_state.get(
        "teaching_mode_previous",
        st.session_state.get("teaching_mode", "mentor"),
    )
    st.markdown(
        '<div style="font-size:0.78rem;color:#94a3b8;margin:0.55rem 0 0.25rem 0;">'
        "Teaching mode</div>",
        unsafe_allow_html=True,
    )
    _chosen_mode = st.radio(
        "Teaching mode",
        options=_TEACHING_OPTIONS,
        format_func=lambda key: _TEACHING_LABELS.get(key, key),
        index=_TEACHING_OPTIONS.index(_previous_mode)
        if _previous_mode in _TEACHING_OPTIONS
        else 0,
        key="teaching_mode",
        label_visibility="collapsed",
        help="Pick the scope for the next model call. "
             "Switching mid-chat swaps the system prompt in place.",
    )
    st.caption(_TEACHING_HELP.get(_chosen_mode, ""))
    if _chosen_mode != _previous_mode:
        # We do NOT clear the chat — only the system role at index 0
        # changes. We do drop the response cache because the *system
        # prompt* is an implicit input to every reply and stale cache
        # entries from the old scope would be confusing.
        st.session_state["messages"][0] = {
            "role": "system",
            "content": _active_system_prompt(st.session_state),
        }
        st.session_state["response_cache"] = {}
        st.session_state["teaching_mode_previous"] = _chosen_mode
        st.toast(
            f"Switched to {_TEACHING_LABELS.get(_chosen_mode, _chosen_mode)}. "
            "System prompt updated; next message uses the new scope.",
            icon="🔁",
        )

    # Model selector. Curated free-tier list is the easy default;
    # the advanced expander is for users who want a different id.
    st.markdown(
        '<div style="font-size:0.78rem;color:#94a3b8;margin:0.55rem 0 0.25rem 0;">'
        "Model</div>",
        unsafe_allow_html=True,
    )
    if FREE_MODEL_CHOICES:
        _labels = [m["label"] for m in FREE_MODEL_CHOICES]
        # When a custom model override is active the curated dropdown is
        # *not* the active selection. We still render it (disabled, with a
        # lock caption) so the user can see what's underneath and re-enable
        # it from the Advanced section. The previous version of this block
        # always overwrote ``session_state["model"]`` with the dropdown's
        # resolved id, which silently clobbered any custom id the user had
        # typed in the Advanced expander on the next rerun.
        _override_active = bool(st.session_state.get("custom_model_override"))
        _current = st.session_state["model"]
        _current_label = next(
            (m["label"] for m in FREE_MODEL_CHOICES if m["id"] == _current),
            _labels[DEFAULT_SELECTED_MODEL_INDEX],
        )
        _chosen_label = st.selectbox(
            "Model",
            _labels,
            index=_labels.index(_current_label),
            label_visibility="collapsed",
            disabled=_override_active,
            help=(
                "Free OpenRouter models. The engine response cache keys "
                "on the model id, so switching gives you a clean cache miss."
                if not _override_active
                else "Locked — a custom model is set in Advanced. Clear it to re-enable the dropdown."
            ),
        )
        if not _override_active:
            st.session_state["model"] = next(
                m["id"] for m in FREE_MODEL_CHOICES if m["label"] == _chosen_label
            )
        # Show the chosen id + a one-line blurb so the user always knows
        # exactly what they're talking to.
        _chosen = next(
            m for m in FREE_MODEL_CHOICES if m["label"] == _chosen_label
        )
        st.caption(f"`{_chosen['id']}` — {_chosen['blurb']}")
        if _override_active:
            st.caption(
                f"🔒 Custom model locked: `{st.session_state['custom_model_override']}` "
                "(use Advanced → Clear to re-enable the curated list)."
            )
        # Surface the router pool size so the user can tell at a glance
        # how many (key, model) pairs the engine will try before giving up.
        # We build the router lazily and only for the *display* call here;
        # the real `_ask` path uses the same cached factory (see
        # `_get_router`) so the count here matches what the engine uses.
        try:
            _pool_router = _get_router()
            _pool_size = _pool_router.healthy_slot_count()
            _key_count = len(list(iter_api_keys()))
            st.caption(
                f"Router pool: **{_pool_size} slot(s)** "
                f"({_key_count} key(s) × {len(_pool_router.slot_labels()) // max(_key_count, 1)} model(s))"
            )
        except (NoFreeModelConfiguredError, ValueError) as _router_cfg_err:
            # iter_models() or iter_api_keys() produced nothing — the
            # user will see this when the first chat fails too. Surface
            # a one-liner in the sidebar so the misconfig is visible
            # before they type a message.
            st.caption(f"⚠️ Router misconfigured: {_router_cfg_err}")
    else:
        # Curated list is empty (shouldn't happen, but be defensive).
        st.session_state["model"] = st.text_input(
            "OpenRouter model",
            value=st.session_state["model"],
            help="Any OpenRouter model ID. Free models end with :free.",
        )

    # Display toggles.
    st.session_state["show_role_labels"] = st.checkbox(
        "Show 'You' / 'SecMentor' labels",
        value=bool(st.session_state["show_role_labels"]),
        help="Adds a small label above each bubble so the question and "
             "answer are clearly separated.",
    )
    st.session_state["concise"] = st.checkbox(
        "Concise mode",
        value=bool(st.session_state["concise"]),
        help="Ask the model for short answers. Significantly faster on free-tier models.",
    )

    # Advanced expander — custom model id + the three sliders.
    with st.expander("Advanced model & limits", expanded=False):
        if FREE_MODEL_CHOICES:
            # Initialize the override key once per session. We track the
            # override separately from ``session_state["model"]`` because
            # the curated dropdown block above also writes to ``model``
            # and would otherwise clobber a custom id on every rerun.
            if "custom_model_override" not in st.session_state:
                st.session_state["custom_model_override"] = ""
            _custom = st.text_input(
                "Custom OpenRouter model",
                value=st.session_state["custom_model_override"],
                placeholder=OPENROUTER_MODEL,
                help=(
                    "Any free OpenRouter model id (must end with ':free'). "
                    "Leave blank to use the selected curated model. "
                    "Rotates across your configured api keys via the "
                    "router's ephemeral-slot path."
                ),
                key="_custom_model_id_input",
            ).strip()
            # The previous version of this block wrote the raw text into
            # ``session_state["model"]`` on every keystroke. That was racy:
            # the curated dropdown ran *before* this expander on each
            # rerun and silently reset ``model`` to the curated default
            # whenever the custom id didn't match a curated label. We now
            # write into ``custom_model_override`` instead and let the
            # chat driver pick it up.
            if _custom and _custom != st.session_state["custom_model_override"]:
                try:
                    cleaned = validate_free_model_id(_custom)
                except InvalidFreeModelIdError as _e:
                    st.error(f"❌ {_e}")
                else:
                    st.session_state["custom_model_override"] = cleaned
                    st.toast(
                        f"Custom model set to `{cleaned}` — dropdown locked "
                        "until you clear it.",
                        icon="🔒",
                    )
                    st.rerun()
            # "Use curated model" clear button. Visible only when an
            # override is active so the curated list isn't cluttered.
            if st.session_state["custom_model_override"]:
                if st.button(
                    "Use curated model (clear override)",
                    key="_clear_custom_override",
                    help="Drop the custom model id and re-enable the curated dropdown.",
                ):
                    st.session_state["custom_model_override"] = ""
                    st.rerun()
        st.session_state["temperature"] = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state["temperature"]),
            step=0.05,
            help="0.0 = focused, 1.0 = creative. ~0.3 for security Q&A.",
        )
        st.session_state["max_tokens"] = st.slider(
            "Max tokens",
            min_value=128,
            max_value=2048,
            value=int(st.session_state["max_tokens"]),
            step=64,
            help="Cap on the assistant reply length. Lower = faster.",
        )
        st.session_state["max_history"] = st.slider(
            "Max history turns",
            min_value=4,
            max_value=40,
            value=int(st.session_state["max_history"]),
            step=2,
            help="Older turns are dropped. System prompt is always kept. "
                 "Lower = faster first-token time on free-tier models.",
        )

    # Apply the cap now (cheap; runs on every rerun).
    st.session_state["messages"] = _truncate_history(
        st.session_state["messages"],
        max_messages=st.session_state["max_history"],
    )

    st.markdown("</div>", unsafe_allow_html=True)  # close CORE PICKS card

    # =========================================================================
    # Card 2 — TRY A QUESTION
    # =========================================================================
    st.markdown(
        '<div class="sm-card">'
        '<div class="sm-card-title"><span class="dot"></span>Try a question</div>'
        '<div class="sm-card-sub">One-click prompts to see how the persona '
        "responds.</div>",
        unsafe_allow_html=True,
    )
    for prompt in EXAMPLE_PROMPTS:
        if st.button(prompt, key=f"ex_{prompt[:24]}", use_container_width=True):
            st.session_state["pending_prompt"] = prompt
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)  # close Try a question card
    st.markdown("</div>", unsafe_allow_html=True)  # close Try a question card

    # =========================================================================
    # Card 3 — RECON QUICK START (always visible, slim)
    # =========================================================================
    # `/recon <target> [scope=<token>]` is one of the platform's flagship
    # features but it is only useful if the user knows it exists. This card
    # sits right above the conversation controls so the slash command is
    # discoverable without scrolling.
    st.markdown(
        '<div class="sm-card">'
        '<div class="sm-card-title"><span class="dot"></span>Recon quick start</div>'
        '<div class="sm-card-sub">Type a target with <span class="sm-inline-code">/recon</span>'
        " to get a multi-tool passive report. Always stay inside the chosen scope.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="sm-pill-list">
          <span class="sm-pill">/recon example.com</span>
          <span class="sm-pill">scope=engagement</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Valid scope tokens: engagement · ctf · labs · redteam · "
        "personal-lab · bugbounty."
    )
    _recon_row = st.columns([3, 2])
    with _recon_row[0]:
        if st.button(
            "▶ Try /recon now",
            key="_recon_quickstart_btn",
            use_container_width=True,
            help="Inserts `/recon example.com scope=engagement` into "
                 "the chat box and reruns the page.",
        ):
            st.session_state["pending_prompt"] = (
                "/recon example.com scope=engagement"
            )
            st.rerun()
    with _recon_row[1]:
        _scope_options = [
            "(use default)",
            "engagement", "ctf", "labs", "redteam",
            "personal-lab", "bugbounty",
        ]
        _current_scope = st.session_state.get("recon_scope_token")
        try:
            _scope_index = (
                _scope_options.index(_current_scope)
                if _current_scope in _scope_options
                else 0
            )
        except ValueError:
            _scope_index = 0
        _picked = st.selectbox(
            "Scope",
            options=_scope_options,
            index=_scope_index,
            key="_recon_scope_picker",
            label_visibility="collapsed",
            help="Sets the scope used by every `/recon <target>` turn.",
        )
        st.session_state["recon_scope_token"] = (
            None if _picked == "(use default)" else _picked
        )
    st.markdown("</div>", unsafe_allow_html=True)  # close Recon quick start card

    # =========================================================================
    # Card 4 — CONVERSATION (New chat + Download transcript)
    # Moved above CHAT HISTORY so the duplicate "+ New chat" button can
    # be dropped from the chat card. The chat card now only lists past
    # sessions; starting a fresh one is a single click here.
    # =========================================================================
    st.markdown(
        '<div class="sm-card">'
        '<div class="sm-card-title"><span class="dot"></span>Conversation</div>'
        '<div class="sm-card-sub">Reset the chat or export the current '
        "transcript.</div>",
        unsafe_allow_html=True,
    )
    if st.button("➕  New chat", key="_conv_new", use_container_width=True):
        # Re-seed the system prompt from the current teaching mode so
        # a "New chat" started in mentor mode keeps mentor scope (and
        # vice versa). The helper fails closed to the defensive
        # prompt on any unexpected state.
        st.session_state["messages"] = [
            {
                "role": "system",
                "content": _active_system_prompt(st.session_state),
            }
        ]
        st.session_state["response_cache"] = {}
        st.session_state["last_elapsed"] = None
        st.rerun()
    _transcript = _serialize_for_download(
        st.session_state["messages"], model=st.session_state["model"]
    )
    st.download_button(
        "⬇️  Download transcript",
        data=_transcript,
        file_name=f"secmentor-transcript-{int(time.time())}.txt",
        mime="text/plain",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)  # close Conversation card

    # =========================================================================
    # Card 5 — CHAT HISTORY (compact: 3 rows visible, no inner New chat)
    # =========================================================================
    # Storage path note: on Streamlit Cloud (free tier) the SQLite file
    # lives in a path that is wiped on every redeploy. We surface that
    # honestly in the subtitle so the user is not surprised when older
    # chats vanish after a code change ships.
    _is_cloud = bool(os.getenv("STREAMLIT_SHARING")) or (
        Path.home().as_posix().startswith("/home/adminuser")
    )
    _history_sub = (
        "Stored locally in this session \u2014 Cloud redeploys wipe history. "
        "Soft-delete keeps them recoverable."
        if _is_cloud
        else "Stored locally \u00b7 soft-delete keeps them recoverable."
    )
    st.markdown(
        '<div class="sm-card">'
        '<div class="sm-card-title"><span class="dot"></span>Chat history</div>'
        f'<div class="sm-card-sub">{_history_sub}</div>',
        unsafe_allow_html=True,
    )
    # Surface the one-time DB-init warning if storage init failed earlier
    # in ``_init_state``. We do not re-attempt init here: a flaky DB
    # stays flaky, and retrying would mask the cause.
    _init_warn = st.session_state.get("db_init_warning")
    if _init_warn:
        st.warning(_init_warn, icon="⚠️")
    elif not st.session_state.get("db_initialised", False):
        st.warning(
            "Chat history is not initialised this session.",
            icon="⚠️",
        )

    # Lazy-load the list on first render so users without DB still see
    # a usable sidebar (just an empty list).
    if not st.session_state["chats"] and st.session_state.get(
        "db_initialised", False
    ):
        try:
            st.session_state["chats"] = _list_chats(limit=20)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not list chats: {exc}")
            st.session_state["chats"] = []

    _chat_list = st.session_state.get("chats") or []
    if not _chat_list:
        st.caption("No saved chats yet — start a conversation to see it here.")
    else:
        # Compact: one slim pill per row. The whole row is clickable
        # (Open) and a tiny trash icon on the right soft-deletes.
        # Cap to 8 always-visible rows so the card stays small; older
        # rows live behind a single "Show all" expander.
        _MAX_VISIBLE = 3
        _visible = _chat_list[:_MAX_VISIBLE]
        _overflow = _chat_list[_MAX_VISIBLE:]

        def _render_chat_row(_chat: dict) -> None:
            """Render one chat row as a single slim pill (Open + 🗑)."""
            _cid = _chat.get("id")
            _title = _chat.get("title") or "(untitled)"
            _updated = _chat.get("updated_at") or ""
            _title_display = (
                _title if len(_title) <= 30 else _title[:27] + "…"
            )
            _is_active = _cid == st.session_state.get("active_chat_id")
            _row_cls = "sm-chat-row is-active" if _is_active else "sm-chat-row"
            _meta = (
                _format_chat_timestamp(_updated) if _updated else "—"
            )
            # One row: dot + title (Open button, takes the full width
            # via use_container_width=False so it inherits the pill
            # surface) + tiny trash button on the right.
            _row_cols = st.columns([11, 1], gap="small")
            with _row_cols[0]:
                st.markdown(
                    f'<div class="{_row_cls}">'
                    + ("●" if _is_active else "·")
                    + f'<span class="sm-chat-title">{_title_display}</span>'
                    f'<span class="sm-chat-meta">{_meta}</span>'
                    "</div>",
                    unsafe_allow_html=True,
                )
                st.button(
                    "Open",
                    key=f"_open_chat_{_cid}",
                    help=(_meta if _updated else "Open this chat"),
                    on_click=_open_chat,
                    args=(_cid,),
                )
            with _row_cols[1]:
                st.markdown(
                    '<div class="sm-chat-trash">',
                    unsafe_allow_html=True,
                )
                st.button(
                    "🗑",
                    key=f"_del_chat_{_cid}",
                    help="Soft-delete (moves to trash; recoverable later)",
                    on_click=_soft_delete_chat,
                    args=(_cid,),
                )
                st.markdown("</div>", unsafe_allow_html=True)

        for _chat in _visible:
            _render_chat_row(_chat)

        if _overflow:
            with st.expander(
                f"Show all ({len(_chat_list)} total)", expanded=False
            ):
                for _chat in _overflow:
                    _render_chat_row(_chat)
    st.markdown("</div>", unsafe_allow_html=True)  # close Chat history card

    # =========================================================================
    # Card 6 — OVERVIEW (four pillars + where-to-practice, moved to bottom)
    # =========================================================================
    st.markdown(
        '<div class="sm-card">'
        '<div class="sm-card-title"><span class="dot"></span>Overview</div>'
        '<div class="sm-card-sub">Four pillars and the legal practice '
        "playgrounds.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
**Four pillars** this assistant is engineered to teach:
- 🛡️ **Defensive security** — threat modeling, IR, hardening, IAM, network.
- 🔁 **DevSecOps** — secure SDLC, SAST/DAST, supply chain, secrets, K8s.
- 🧠 **AI / ML security** — prompt injection, OWASP LLM Top 10, agent safety.
- 🎯 **Offensive-security education** — *concept-level*: structure of an attack,
  why it works, what defeats it. Not turn-key exploits.

**Teaching mode** (Core picks) lets you swap the system prompt mid-session:
- *Defensive (4 pillars)* — default. Concept-level only.
- *CTF / Lab mentor* — unlocks lab scope (HTB, THM, PortSwigger, DVWA, WebGoat).
  May produce runnable exploit snippets framed for the lab, always paired
  with the defensive countermeasure. See `docs/technical_write_up.md` Decision 6.

**Out of scope (both modes):** working exploit code against a specific real
system, malware, droppers, C2, payloads against specific real WAFs/EDRs/MFAs,
brand-new malware strains, critical-infrastructure targets.

**Where to practice legally:** HackTheBox, TryHackMe, PortSwigger Academy,
DVWA, WebGoat, PicoCTF.
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)  # close Overview card


# Single call-site — opens the sidebar exactly once, then delegates the
# whole card-based layout to ``_render_sidebar``. Streamlit lets you
# open the sidebar more than once; we keep this single call so the
# order of cards is fixed and easy to reason about.
with st.sidebar:
    _render_sidebar()


# --- Header ------------------------------------------------------------------

# The hero subtitle changes slightly when the mentor mode is active so the
# user always knows which scope they are in at a glance.
_hero_subtitle = "AI-Powered Cybersecurity Learning & Analysis Platform"
if st.session_state.get("teaching_mode") == "mentor":
    _hero_subtitle = (
        "AI-Powered Cybersecurity Learning & Analysis Platform — "
        "CTF / Lab mentor mode active"
    )

_CAPABILITY_BADGES = [
    ("Cybersecurity Mentor", "🛡"),
    ("Security Research", "🔍"),
    ("File Analysis", "📄"),
    ("CTF & Lab Guidance", "🎯"),
    ("Multi-Model AI", "🤖"),
]

st.markdown(
    f"""
    <div class="hero">
      <div class="eyebrow">SECURITY OPERATIONS</div>
      <h1><span class="logo">🛡</span> SecMentor</h1>
      <p class="subtitle">{_hero_subtitle}</p>
      <p class="tagline">Learn • Analyze • Defend • Research</p>
      <div class="badges">
        {''.join(f'<span class="badge"><span class="dot"></span>{label}</span>' for _, label in _CAPABILITY_BADGES)}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --- Chat rendering ----------------------------------------------------------

def _render_bubble(message: ChatMessage) -> None:
    """Render one message as a chat bubble.

    Markdown is rendered for assistant messages (so tables, lists, code
    blocks all show up correctly) and escaped for user messages (so the
    user's literal text is never interpreted as markdown).

    A small "You" / "SecMentor" label is shown above the bubble (unless
    the user turned it off in the sidebar) so the question and answer
    are always visually separated.
    """
    role = message.get("role", "assistant")
    content = message.get("content", "")
    # ``content`` can be either a ``str`` (text-only turn) or a
    # ``list[dict]`` of multimodal parts (image-bearing turn, persisted
    # as JSON and re-decoded by ``app.storage.list_messages``). The
    # rendering branches below call ``str``-only methods
    # (``content.replace(...)``, ``st.markdown(content)``, the copy
    # button's markdown-to-plain pipeline) that blow up with
    # ``AttributeError: 'list' object has no attribute 'replace'`` on
    # the list shape. Coerce once at the top so every branch sees a
    # ``str``; ``_coerce_message_text`` preserves the user's text and
    # replaces image parts with a short `[image: ...]` placeholder.
    content = _coerce_message_text(content)
    align = _bubble_alignment(role)
    show_label = bool(st.session_state.get("show_role_labels", True))
    label_text = "You" if role == "user" else "SecMentor"

    def _label_html() -> str:
        if not show_label or role == "system":
            return ""
        return f'<div class="role-label">{label_text}</div>'

    if role == "system":
        st.markdown(
            f'<div class="row left"><div class="bubble-system">{content}</div></div>',
            unsafe_allow_html=True,
        )
        return

    if role == "user":
        # Escape backticks minimally so user text never renders as code.
        safe = content.replace("`", "\u200b`")
        st.markdown(
            f'<div class="row right">{_label_html()}'
            f'<div class="bubble-user">{safe}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        # Assistant — render markdown via st.markdown inside a styled wrapper.
        # The copy-to-clipboard button (Tier 1 #4) lives *inside* the
        # assistant bubble, right after the rendered markdown, ChatGPT
        # style. The button is its own ``st.components.v1.html`` call so
        # the click handler runs inside an iframe (same-origin to the
        # Streamlit server), and ``navigator.clipboard.writeText`` works
        # as a secure context. The plain-text payload is computed by
        # ``_render_copy_button_for_bubble`` so the user pastes a
        # rendered reply, not the markdown source. The HTML escaping
        # lives in ``web/chat_helpers._copy_button_iframe_html`` and is
        # unit-tested independently of Streamlit.
        st.markdown(
            f'<div class="row left">{_label_html()}'
            f'<div class="bubble-assistant">',
            unsafe_allow_html=True,
        )
        st.markdown(content)
        _render_copy_button_for_bubble(content)
        st.markdown("</div></div>", unsafe_allow_html=True)


def _render_friendly_error(
    exc: BaseException,
    model: str,
    *,
    vision_model: str | None = None,
    text_fallback_attempted: bool = False,
) -> None:
    """Show a user-readable error banner and the raw exception in a toggle.

    The raw ``OpenRouterError`` message contains the full upstream JSON
    payload, which is great for debugging and awful for end users. We
    pass the exception through ``_friendly_error_message`` to get a
    short, actionable headline + body, and we keep the raw text
    available behind ``st.exception`` so a developer can still inspect
    it with one click.

    Args:
        exc: The exception that bubbled out of the engine call.
        model: The model id that produced the error (or, when the
            vision path failed and a text fallback was attempted, the
            *fallback* model id so the user's chosen model is named in
            the banner).
        vision_model: When set, the headline additionally names the
            vision model that was tried first. Used by the pinned
            image/PDF branch so a "vision failed → retried as text"
            failure is clear.
        text_fallback_attempted: ``True`` when the helper already
            tried a text fallback (and that fallback also failed).
            Adds a small note so the user knows the work-around was
            attempted, not merely available.
    """
    headline, body = _friendly_error_message(exc, model)
    st.error(headline)
    if vision_model and vision_model != model:
        st.caption(
            f"Vision call to `{vision_model}` failed first; "
            f"the text fallback to `{model}` also failed."
        )
    elif text_fallback_attempted and vision_model:
        st.caption(
            f"Vision call to `{vision_model}` failed; "
            f"the reply above was produced by `{model}`."
        )
    st.caption(body)
    with st.expander("Raw error (for debugging)", expanded=False):
        st.exception(exc)


def _request_stop() -> bool:
    """Render a small "Stop" button next to the in-flight bubble.

    Returns ``True`` when the user has clicked it (and therefore wants
    the current streaming turn to abort at the next chunk boundary),
    ``False`` otherwise.

    The button is wired via a session-state flag rather than a callback
    so a Streamlit rerun can be triggered from the click *and* the
    chunk-loop in :func:`_ask` can still observe the flag on its next
    iteration. Without a session flag the loop would never know the
    button was clicked because the click and the rerun it triggers both
    happen between chunk iterations.

    The key is namespaced (``chat_stop_button``) so a future second
    stop button (e.g. for a sidebar action) cannot collide.
    """
    # Reset the flag at the *start* of a fresh turn so a stale click
    # from a previous turn cannot cancel the new one. ``_ask`` writes
    # ``_stop_requested`` back to ``False`` when a new turn begins.
    if "stop_requested" not in st.session_state:
        st.session_state["stop_requested"] = False
    clicked = st.button(
        "Stop",
        key="chat_stop_button",
        help="Cancel the current request at the next token boundary. "
             "Anything the model has already streamed stays in the transcript.",
        type="secondary",
    )
    if clicked:
        st.session_state["stop_requested"] = True
    return bool(st.session_state["stop_requested"])


def _consume_stop_flag() -> bool:
    """Read and clear the cooperative stop flag.

    Used at the start of :func:`_ask` so a brand-new turn starts with a
    clean slate (and so a user clicking Stop between turns does not
    poison the next request). The pure read/clear logic lives in
    :func:`web.chat_helpers.consume_stop_flag` so it is unit-testable
    without booting Streamlit.
    """
    return consume_stop_flag(st.session_state)


def _render_chatbox_model_picker() -> None:
    """Render a compact model chip above the chat input.

    The chip is a single button-shaped pill on the left side of the
    chat composer (mirroring the Claude UI's "Sonnet 5 ▾" placement).
    Tapping it opens a popover that lists every entry in
    :data:`FREE_MODEL_CHOICES`; picking one updates
    ``session_state["model"]`` so the sidebar dropdown stays in sync.

    The chip label shows *only* the chosen model name — no role badge,
    no blurb, no complexity knob. Keeping the surface minimal matches
    the user's request for a plain "tap → list → pick" switch.

    The pure label→id resolution and the "did the value actually
    change?" check live in
    :func:`web.chat_helpers.resolve_chatbox_model_id` so the contract
    is pinned by a unit test without booting Streamlit. The helper
    is a no-op when :data:`FREE_MODEL_CHOICES` is empty.
    """
    if not FREE_MODEL_CHOICES:
        return
    labels = [m["label"] for m in FREE_MODEL_CHOICES]
    current = st.session_state.get("model") or (
        FREE_MODEL_CHOICES[DEFAULT_SELECTED_MODEL_INDEX]["id"]
    )
    current_row = next(
        (m for m in FREE_MODEL_CHOICES if m["id"] == current),
        FREE_MODEL_CHOICES[DEFAULT_SELECTED_MODEL_INDEX],
    )
    current_label = current_row["label"]

    # --- Chip column ------------------------------------------------------
    # A single popover button. The face label shows *only* the model
    # name (no "· Balanced", no role, no complexity tier). A chevron
    # is appended so the affordance reads as "tap to expand".
    with st.popover(
        f"{current_label}  ▾",
        use_container_width=False,
    ):
        # Plain radio-style picker inside the popover: a vertical list
        # of every curated model name. No descriptors, no captions,
        # no role/balanced/complexity options — exactly what the user
        # asked for.
        chosen_label = st.radio(
            "Available models",
            options=labels,
            index=labels.index(current_label),
            key="chatbox_model_picker",
            label_visibility="collapsed",
        )

    # --- Pure label→id resolution ----------------------------------------
    # The picker is a no-op when the chosen label matches the current
    # one, so a popover-open/close cycle does not needlessly invalidate
    # the ``_ask`` cache key. The fallback to ``current`` keeps the
    # helper safe against a curated list that has drifted.
    chosen_id, changed = resolve_chatbox_model_id(
        FREE_MODEL_CHOICES,
        chosen_label=st.session_state.get(
            "chatbox_model_picker", current_label
        ),
        current_id=current,
    )
    if changed:
        st.session_state["model"] = chosen_id
        # The cache key in ``_ask()`` includes the model id, so a swap
        # means the next turn cannot accidentally hit a stale cached
        # reply from the previous model. No explicit cache invalidation
        # is needed — the key change is sufficient.


# The helpers above are intentionally kept in the view module so the
# Streamlit decorators (``@st.cache_resource``) and the constant
# ``FREE_MODEL_CHOICES`` are in scope. The pure-logic parts of the
# cooperative-stop contract are re-exported to ``web.chat_helpers`` for
# unit-testing without booting the whole UI module — see the matching
# ``_stop_flag_logic`` shim there.


_visible_messages = st.session_state["messages"][1:]  # skip the system prompt
if not _visible_messages:
    # First-run / post-"New chat" experience. A single card with the
    # product framing + a few example directions so the user does not
    # stare at an empty white column. All copy is static; clicking
    # suggestions is handled by the existing "Try a question" buttons
    # in the sidebar (they set pending_prompt which _ask() consumes).
    _empty_chips = [
        "Explain a recent CVE",
        "Walk me through a web attack chain",
        "Harden a Linux server",
        "Review a log excerpt for IOCs",
    ]
    st.markdown(
        f"""
        <div class="empty-state">
          <div class="icon">💬</div>
          <h3>Start a security conversation</h3>
          <p>
            Ask about a vulnerability, walk through an attack chain, paste a log
            for triage, or switch to <strong>CTF / Lab mentor</strong> in the
            sidebar for hands-on guidance.
          </p>
          <div class="suggestions">
            {''.join(f'<span class="chip">{c}</span>' for c in _empty_chips)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
for message in _visible_messages:
    _render_bubble(message)


# --- Input handling ----------------------------------------------------------

def _ask(prompt: str | dict[str, object] | None) -> None:
    """Send the pending user turn to the model and stream the reply.

    The argument shape is the new dict emitted by the input driver:

        {"text": str, "content": str | list[dict], "model": str,
         "had_files": bool, "signature": int}

    The text-only callers (test harness, legacy code paths) can still
    pass a bare ``str`` — the function normalises it into the dict
    shape using the currently-selected model.

    Streamlit's render loop is single-pass per script run. If we do
    everything in one run, the user only sees the final state — they
    miss their own question bubble while the model is thinking. The
    fix is a two-pass pattern:

        pass 1: append the user turn, persist the request under
                session_state["pending_request"], st.rerun(). The rerun
                causes the user bubble to render (the loop at the top of
                the script reads from session_state).
        pass 2: on the next run, _ask() sees "pending_request", runs the
                model, appends the reply, clears the pending flag, and
                reruns again so the assistant bubble renders.

    Pass 2 is driven by a top-level call site further down (see the
    "Two-pass continuation" block). When that block fires, it calls
    ``_ask(None)`` — the ``None`` is a signal to ignore the argument
    and consume ``pending_request`` instead. The empty-prompt guard
    below only runs in pass 1 (when ``pending_request`` is NOT in
    session_state), so ``None`` never reaches it.
    """
    # Pass 1: persist the request, rerun, and return. The rerun will
    # re-execute the script top-to-bottom; the chat-render loop will
    # paint the user bubble from the textual
    # summary, not from the raw content parts. The summary comes
    # from the text field, with a small marker for image turns
    # so the user can see something was attached.
    if "pending_request" not in st.session_state:
        # Normalise a bare string (legacy / test) into the dict shape.
        if isinstance(prompt, str):
            if not prompt or not prompt.strip():
                return  # empty prompt -> nothing to do
            request = {
                "text": prompt,
                "content": prompt,
                "model": st.session_state["model"],
                "had_files": False,
                "signature": hash(prompt) & 0xFFFFFFFF,
            }
        elif prompt is None:
            return
        else:
            request = prompt
        # Guard: the content may be a text string or a list of
        # content parts. Either way, an "empty" turn is one with
        # no text *and* no image parts.
        content_obj = request["content"]
        if isinstance(content_obj, str):
            if not content_obj.strip():
                return
        elif not content_obj:
            return
        # The history loop paints user bubbles from the textual
        # summary, not from the raw content parts. The summary comes
        # from the text field, with a small marker for image turns
        # so the user can see something was attached.
        display_text = request.get("text", "") or ""
        if not display_text and isinstance(content_obj, list):
            display_text = "[attached image(s)]"
        st.session_state["messages"].append(
            {"role": "user", "content": display_text}
        )

        # --- Chat history persistence (pass 1) ---------------------------
        # If this is the first message of a new session, ensure we have
        # a row in the ``chats`` table and bind ``active_chat_id`` to it.
        # The chat's *title* is the first 60 chars of the user's first
        # message — easier to spot in the sidebar list than ``(untitled)``.
        if (
            st.session_state.get("db_initialised", False)
            and st.session_state.get("active_chat_id") is None
        ):
            try:
                _title_seed = (display_text or "").strip()[:60] or "(new chat)"
                _new_chat_id = _storage.create_chat(
                    title=_title_seed
                )
                st.session_state["active_chat_id"] = _new_chat_id
                # Invalidate the sidebar cache so the new chat shows up.
                st.session_state["chats"] = _storage.list_chats(limit=20)
            except Exception as exc:  # noqa: BLE001
                # Storage is optional — never block the user from chatting
                # just because SQLite had a bad day. The sidebar widget
                # surfaces the same warning once.
                st.session_state.setdefault(
                    "_db_persist_warning_shown", False
                )
                if not st.session_state["_db_persist_warning_shown"]:
                    st.warning(
                        f"Chat history will not persist this session: {exc}",
                        icon="⚠️",
                    )
                    st.session_state["_db_persist_warning_shown"] = True

        # Persist the user turn. We store the *raw* multimodal content
        # (the list of parts) when the request carried image attachments,
        # so retrieval / future RAG can still match against the exact
        # payload the model saw. For text-only turns, we store the
        # display string. ``append_message`` JSON-encodes list payloads
        # for us.
        if st.session_state.get("db_initialised", False) and st.session_state.get(
            "active_chat_id"
        ):
            try:
                _user_payload = (
                    content_obj
                    if isinstance(content_obj, list)
                    else display_text
                )
                _storage.append_message(
                    chat_id=st.session_state["active_chat_id"],
                    role="user",
                    content=_user_payload,
                )
            except Exception as exc:  # noqa: BLE001
                st.session_state.setdefault(
                    "_db_persist_warning_shown", False
                )
                if not st.session_state["_db_persist_warning_shown"]:
                    st.warning(
                        f"Could not persist the user turn: {exc}",
                        icon="⚠️",
                    )
                    st.session_state["_db_persist_warning_shown"] = True

        st.session_state["pending_request"] = request
        st.session_state["pending_started_at"] = time.perf_counter()
        # Reset the cooperative stop flag so a click from a previous
        # turn does not abort the one we are about to dispatch. The
        # widget itself is keyed and only visible while a request is
        # in flight; flipping the flag to ``False`` here is purely a
        # belt-and-braces guarantee against a stale click that landed
        # in the same rerun cycle as the rerun above.
        st.session_state["stop_requested"] = False
        st.rerun()
        return

    # Pass 2: we are on the rerun triggered above. Pull the request
    # out, run the model, append the reply, clear the flag, rerun
    # again so the assistant bubble actually appears.
    request = st.session_state.pop("pending_request", None)
    started = st.session_state.pop("pending_started_at", time.perf_counter())
    # Re-seed the stop flag to ``False`` at the *start* of every
    # pass-2 turn. This is the canonical "fresh slate" — even if the
    # flag was somehow left ``True`` by a buggy earlier session
    # version (or by the user clicking Stop between pass 1 and pass
    # 2, which can happen because the placeholders are visible
    # during the rerun), this guarantees the new turn starts
    # uninterruptible until the new placeholders render.
    _consume_stop_flag()
    if not request:
        return  # defensive: nothing to do
    # Legacy fallback: a bare string might have been left in
    # pending_request by a prior version of the view.
    if isinstance(request, str):
        request = {
            "text": request,
            "content": request,
            "model": st.session_state["model"],
            "had_files": False,
            "signature": hash(request) & 0xFFFFFFFF,
        }
    content = request["content"]
    model = request.get("model") or st.session_state["model"]
    # ``text_model`` is the user's original sidebar selection, preserved
    # so the streaming helper can degrade to it if the vision path
    # fails. When the user already picked a vision model there is no
    # separate text preference; in that case we fall back to ``model``
    # itself (the degrade will still hit the same upstream, but the
    # helper re-emits it as a text-only payload so the user still
    # gets a reply).
    text_model = request.get("text_model") or model

    # Build the messages list the engine will see. If "Concise mode" is
    # on, we prepend a short instruction to the system prompt. This is
    # cheaper than trimming the system prompt itself and it is easy to
    # toggle from the sidebar.
    history_for_api = list(st.session_state["messages"])
    # Make sure the user turn in history uses the original content
    # (string or list of parts), not the display summary we appended
    # in pass 1 — otherwise the model would never see the images.
    if history_for_api:
        history_for_api[-1] = {"role": "user", "content": content}
    if st.session_state.get("concise") and history_for_api:
        first = history_for_api[0]
        if first.get("role") == "system":
            history_for_api[0] = {
                "role": "system",
                "content": (
                    first["content"]
                    + "\n\nBe concise. Default to under 6 sentences unless "
                    "the user explicitly asks for depth or a list."
                ),
            }

    try:
        messages_for_api = _build_messages(history_for_api, content)
    except ValueError as exc:
        st.error(f"Could not build the request: {exc}")
        return

    # 3. Cache check: identical (model, temperature, max_tokens,
    #    content signature) is served from the previous reply
    #    without a network call. This makes repeat questions feel
    #    instant while the user is exploring. The signature field
    #    folds in image parts, so a re-send with a different image
    #    is correctly treated as a fresh request.
    temperature = float(st.session_state["temperature"])
    max_tokens = int(st.session_state["max_tokens"])
    cache_key = (model, temperature, max_tokens, request.get("signature", 0))
    cached = st.session_state["response_cache"].get(cache_key)
    if cached is not None:
        st.session_state["last_elapsed"] = 0.0
        st.session_state["messages"].append(
            {"role": "assistant", "content": cached}
        )
        # Rerun so the history loop at the top of the script re-paints
        # the bubbles with the new assistant turn. Without this, the
        # reply is appended to session_state but never rendered until
        # the user takes some other action (e.g. sends another message).
        st.rerun()
        return

    # 4. Show a thinking bubble and time the call. We render a placeholder
    #    and replace it with the streamed reply. Tier 1 #1: the unpinned
    #    (router-managed) path now streams token-by-token via
    #    ``st.write_stream``; the pinned (had_files) path still uses the
    #    blocking ``chat()`` call because vision / PDF payloads are not
    #    safe to stream — we cannot rotate to a different model mid-turn
    #    once a partial multimodal reply has started.
    show_label = bool(st.session_state.get("show_role_labels", True))
    label_html = (
        '<div class="role-label">SecMentor</div>' if show_label else ""
    )
    # Co-locate the thinking bubble and the Stop button in the same
    # Streamlit container so they render as one row. ``st.columns``
    # is the supported way to do this: the bubble takes the wider
    # fraction, the button takes a small narrow fraction on the right.
    # The placeholder is inside the left column so the markdown
    # rewrite loop (``placeholder.markdown(...)`` further down) keeps
    # working without re-acquiring a widget handle.
    _thinking_cols = st.columns([0.88, 0.12])
    with _thinking_cols[0]:
        placeholder = st.empty()
        placeholder.markdown(
            f'<div class="row left">{label_html}'
            f'<div class="bubble-thinking">'
            f'<span class="pulse"></span>Thinking…</div></div>',
            unsafe_allow_html=True,
        )
    with _thinking_cols[1]:
        # Render the Stop control. The click itself does not interrupt
        # a blocking httpx call from outside the Python process — we
        # can only observe the flag at the next chunk boundary — but
        # the rerun it triggers frees the Streamlit script and lets
        # the user send a different message or refresh state. The
        # chunk loops below also check the flag directly so the visual
        # placeholder updates on the next delta.
        _user_stopped = _request_stop()
    # Browser-level toast in case the page itself is unresponsive (the
    # chat bubble above won't redraw until chat() returns, so the user
    # might think the app is frozen). Toast is non-blocking and appears
    # in the bottom-right corner.
    st.toast(f"Calling {model}…", icon="⏳")

    started = time.perf_counter()
    # Mark the status line "thinking" so the user can see the request is
    # alive even though Streamlit is blocked on the network call. The
    # status line is rendered on every rerun, so the dot pulses; we just
    # change the label so it reads "thinking" instead of a stale time.
    st.session_state["last_elapsed"] = "thinking"
    # Use the multi-key, multi-model router instead of a single direct
    # call. The router is built once per Streamlit process via
    # `@st.cache_resource`, so the slot health state (disabled, last
    # error) persists across reruns and the next user request picks up
    # where the last one left off. We let the router pick the slot —
    # do NOT pass `model=model` here, because the sidebar selection is
    # the *preferred* model but the router may legitimately rotate to
    # another model if the preferred one is rate-limited.
    #
    # Exception: when the request attached files, the model's
    # capability matters (vision vs text-only, PDF-text extraction
    # quality). In that case we pin the model by calling
    # ``app.openrouter.chat`` directly. The router does not expose a
    # per-call model filter (it rotates across its slot pool by
    # design), so the only way to guarantee "this model id only" is
    # to skip the router for that turn. The trade-off is no automatic
    # key rotation, which is acceptable because vision requests are
    # user-initiated and a single 429 surfaces the same friendly
    # error path as the router would have produced.
    try:
        router = _get_router()
    except (NoFreeModelConfiguredError, ValueError) as cfg_exc:
        placeholder.empty()
        st.session_state["last_elapsed"] = None
        st.error("Router is not configured. Set `OPENROUTER_API_KEY` and "
                 "optionally `OPENROUTER_MODELS` in your environment.")
        st.caption(f"Underlying error: {cfg_exc}")
        return
    # ``reply`` holds the assembled assistant text. The streaming path
    # uses ``st.write_stream`` (which returns the concatenated string),
    # and the pinned path uses the blocking ``chat()`` call directly.
    # We initialise to an empty string so the cache-write and the
    # ``append`` below have a value to work with even if every streaming
    # branch raises before yielding any text (in which case we
    # ``return`` from the except blocks without reaching the append).
    reply: str = ""
    had_streaming_failure: BaseException | None = None
    # ``has_image_attachments`` is the *actual* gate for the vision
    # helper path. ``had_files`` is true for PDFs too (PDFs are
    # uploaded via the same file picker), but the vision helper exists
    # to carry image parts in the wire payload — a PDF turn was
    # already reduced to plain text by ``build_user_turn_content``
    # and just needs the regular text-streaming path. Without this
    # gate, a PDF would be sent through the vision helper with the
    # 90s vision timeout, the user's text model id as the
    # ``vision_model_id``, and a "Vision call failed" toast for a
    # turn that never had a vision payload to begin with.
    has_image_attachments = isinstance(content, list)
    try:
        if request.get("had_files") and has_image_attachments:
            # Pinned-model path: bypass the router so the model id is
            # guaranteed. The streaming helper drives the call via
            # ``router.stream_chat(model=vision_model_id, timeout=...)``
            # so the user sees tokens as they arrive (the old blocking
            # ``chat()`` call left them staring at "Thinking…" for up to
            # 60 seconds before any text appeared). When the vision
            # stream fails before producing any delta — the Nemotron VL
            # cold-start + 504 pattern the user has been hitting — the
            # helper degrades to a text-only payload on the user's
            # preferred model so the reply still arrives.
            #
            # We pass ``router`` (not the raw client) so the helper
            # reuses the same slot health state the text path uses; the
            # ``model=`` pin keeps it on the vision slot for the first
            # attempt and on the text slot for the degrade. ``timeout``
            # comes from ``vision_timeout_seconds()`` (currently 45s;
            # the degrade path absorbs anything longer). The shrink
            # matters because a stuck vision slot used to burn the full
            # 90s before the user saw any text — now the text fallback
            # kicks in within ~45s of the click, which is the threshold
            # at which users start to perceive a stall.
            _vision_degraded = False
            _vision_buffer = ""
            _chunk_iter = stream_vision_turn_with_fallback(
                router=router,
                messages=messages_for_api,
                vision_model_id=model,
                fallback_model_id=text_model,
                content=content,
                files=None,  # file objects are not persisted across the rerun;
                             # ``degrade_vision_to_text`` accepts ``None`` and
                             # falls through to text-only with no per-file stubs.
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=vision_timeout_seconds(),
            )
            try:
                for _chunk, _source in _chunk_iter:
                    # Cooperative stop: check the flag on every chunk
                    # so a user click between deltas aborts the turn
                    # without waiting for the upstream to finish.
                    # ``stop_requested`` is reset at the start of
                    # every new turn by ``_consume_stop_flag``, so a
                    # stale click from a previous turn cannot poison
                    # this one.
                    if st.session_state.get("stop_requested"):
                        st.toast(
                            "Stopped. Partial reply kept.",
                            icon="⏹",
                        )
                        break
                    if _source == "degraded":
                        # The vision stream failed before yielding any
                        # delta; the helper is about to emit text chunks
                        # from the fallback model. Switch the placeholder
                        # to a one-line notice so the user knows what is
                        # happening, then keep accumulating the streamed
                        # text.
                        _vision_degraded = True
                        placeholder.markdown(
                            f'<div class="row left">{label_html}'
                            f'<div class="bubble-thinking">'
                            f'<span class="pulse"></span>Vision call '
                            f'failed — retrying with {text_model}…'
                            f'</div></div>',
                            unsafe_allow_html=True,
                        )
                        continue
                    _vision_buffer += _chunk
                    placeholder.markdown(
                        f'<div class="row left">{label_html}'
                        f'<div class="bubble">'
                        f'{_vision_buffer}▌'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )
            except OpenRouterError:
                # The helper re-raises after either (a) the vision
                # stream failed *and* the degrade path was not eligible
                # (auth error, partial reply that we cannot mix, etc.)
                # or (b) the *text* fallback also failed. In both cases
                # the outer ``except OpenRouterError`` block below
                # renders the friendly banner with the correct
                # attribution; we just keep whatever partial reply the
                # helper already emitted in ``_vision_buffer``.
                pass
            reply = _vision_buffer
            placeholder.empty()
        else:
            # Streaming path: ``st.write_stream`` consumes the router's
            # generator and renders each delta into its own auto-managed
            # container below the page. It returns the concatenated
            # final text, which becomes ``reply``. If the stream yields
            # zero deltas (an upstream that swallows content) the
            # router raises ``OpenRouterError`` *before* ``st.write_stream``
            # ever produces text, so the placeholder is still on screen
            # when the except block runs and ``reply`` stays "".
            # ``st.write_stream`` consumes a generator and renders each
            # yielded delta into a single auto-managed Streamlit
            # container; it returns the concatenated string once the
            # generator is exhausted. To let the user cancel a long
            # run, we wrap the router generator in a thin
            # stop-checking shim. The shim is intentionally tiny: a
            # dict lookup on ``stop_requested`` per chunk is cheaper
            # than the network round-trip it gates.
            def _stream_with_stop() -> "Iterator[str]":
                for delta in router.stream_chat(
                    messages_for_api,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ):
                    if st.session_state.get("stop_requested"):
                        st.toast(
                            "Stopped. Partial reply kept.",
                            icon="⏹",
                        )
                        return
                    yield delta

            reply = st.write_stream(_stream_with_stop())
            # Defensive: a streaming endpoint that *silently* returns an
            # empty string is a different failure mode from one that
            # raises (e.g. an upstream that signals "done" without ever
            # sending ``data: [DONE]``). The router's empty-stream guard
            # catches the most common case; this branch covers the
            # corner where the stream completes cleanly with whitespace
            # only. We treat both as transient failures and surface
            # the same friendly banner the view already shows for
            # empty-reply 4xx.
            #
            # Exception: the user explicitly clicked Stop. In that case
            # the empty reply is *intentional* (the model may have
            # produced zero deltas before the user cancelled), and the
            # ``stop_requested`` flag is still ``True`` here. We must
            # not raise — the user got what they asked for. The
            # downstream commit logic (which uses ``had_streaming_failure``
            # to distinguish success from partial) already handles an
            # empty ``reply`` cleanly: the turn is silently dropped.
            _user_stopped = bool(st.session_state.get("stop_requested"))
            if not _user_stopped and (not reply or not reply.strip()):
                raise OpenRouterError(
                    "Model returned an empty reply.",
                    status=None,
                    model=model,
                )
            placeholder.empty()
    except AllSlotsExhaustedError as exc:
        # Every (key, model) slot failed in turn. This is a harder
        # error than a single-model 429: render a dedicated banner
        # that tells the user the entire pool is exhausted and lists
        # the slot labels (with redacted keys) so they can see exactly
        # what was tried.
        placeholder.empty()
        st.session_state["last_elapsed"] = None
        # If the streaming path produced a partial reply before the
        # final slot failed, surface a thin divider so the user can
        # tell where the partial text ends and the error banner
        # begins. The divider is purely cosmetic; it is a `<hr>` and
        # uses default Streamlit theming.
        if reply:
            st.markdown("---")
        st.error(
            "All router slots are exhausted. Every configured "
            "(key, model) pair failed — most likely the daily per-account "
            "free-tier cap has been hit on every account."
        )
        st.caption(
            "Tried: " + ", ".join(router.slot_labels())
        )
        # Vision-aware diagnostic note. When the failing turn was
        # pinned to the only free vision model
        # (``nvidia/nemotron-nano-12b-vl:free``), the pool collapses
        # from ``len(keys) × len(models)`` to ``len(keys) × 1`` — every
        # slot shares the same upstream provider. A single Nemotron
        # throttle therefore burns the whole pool. Surface a one-line
        # hint so the user can tell "Nemotron is throttling" from
        # "all my keys are wrong". Detection: the request attached
        # files (so we were on the pinned vision path) AND the pinned
        # model is the only known vision model id. The text path
        # keeps the original generic banner.
        _vision_pool_collapsed = bool(
            request.get("had_files") and model_supports_vision(model)
        )
        if _vision_pool_collapsed:
            st.caption(
                "Vision pool is single-model "
                f"(`{model}`) — every (key, model) slot shares the "
                "same upstream provider, so one provider throttle "
                "exhausts the pool. Wait a minute and retry, or "
                "switch to a text-only turn for now."
            )
        with st.expander("Last error per slot (for debugging)", expanded=False):
            for slot_label in router.slot_labels():
                # Re-derive the slot to surface its last_error. The
                # router exposes slot_labels() but keeps internal
                # state private; this is the smallest leak that gives
                # a useful debug view.
                _slot = next(s for s in router._slots if s.short_label() == slot_label)
                if _slot.last_error is not None:
                    st.caption(f"`{slot_label}`: {_slot.last_error}")
                else:
                    st.caption(f"`{slot_label}`: (no error recorded)")
        return
    except OpenRouterError as exc:
        # Single-slot OpenRouter error that bubbled out of router.chat
        # / router.stream_chat (the router normally handles these
        # internally; reaching here means something unexpected — e.g.
        # a bad 4xx we don't catch, or the helper raised before the
        # router could rotate). On the streaming path, ``reply`` may
        # hold a partial reply (the router deliberately re-raises
        # after the first delta so the user keeps what they got);
        # we render it above the error banner so the user does not
        # lose the work the model already produced.
        placeholder.empty()
        st.session_state["last_elapsed"] = None
        if reply:
            st.markdown("---")
        # When the pinned-vision branch was active, name the vision
        # model explicitly so the user can tell whether the failure
        # happened on the *vision* call or the *text* degrade. The
        # helper leaves ``had_streaming_failure`` un-set on its own
        # re-raise (we suppress it in the inner ``except`` above), so
        # any ``OpenRouterError`` reaching here either came from the
        # empty-reply guard or from a model path that does not
        # degrade (e.g. a corrupted multipart payload).
        _vision_label = model if request.get("had_files") else None
        _fallback_attempted = bool(
            request.get("had_files") and reply
        )
        _render_friendly_error(
            exc,
            model,
            vision_model=_vision_label,
            text_fallback_attempted=_fallback_attempted,
        )
        had_streaming_failure = exc
        # We intentionally fall through to the cache + history-append
        # block so a partial reply stays in the conversation. The
        # history-append is guarded by ``had_streaming_failure is None``
        # below, so on a *non-partial* error (zero deltas) we do NOT
        # pollute the transcript with an empty assistant turn.
    elapsed = time.perf_counter() - started
    st.session_state["last_elapsed"] = elapsed

    # Cache only the unpinned path. Multimodal turns are rarely repeated
    # verbatim, and the cache key is currently model-only, so a vision
    # request and a text-only request with the same temperature / token
    # cap would collide.
    if not request.get("had_files") and reply:
        st.session_state["response_cache"][cache_key] = reply

    # Commit the reply to history when we actually have one. The
    # history-render loop at the top of the script (the
    # ``for message in st.session_state["messages"][1:]`` block) is
    # what actually paints the assistant bubble to the screen. We
    # rerun to re-execute the script top-to-bottom with the new turn
    # in session_state. The placeholder above only ever showed the
    # Thinking… state; the revealed answer lives in the history list.
    if reply and had_streaming_failure is None:
        st.session_state["messages"].append(
            {"role": "assistant", "content": reply}
        )
        # --- Chat history persistence (assistant, success) --------------
        # Best-effort — if SQLite is wedged we still want the user to
        # see the reply. The same warning pattern as pass 1.
        if (
            st.session_state.get("db_initialised", False)
            and st.session_state.get("active_chat_id")
        ):
            try:
                _storage.append_message(
                    chat_id=st.session_state["active_chat_id"],
                    role="assistant",
                    content=reply,
                )
            except Exception as exc:  # noqa: BLE001
                st.session_state.setdefault(
                    "_db_persist_warning_shown", False
                )
                if not st.session_state["_db_persist_warning_shown"]:
                    st.warning(
                        f"Could not persist the assistant reply: {exc}",
                        icon="⚠️",
                    )
                    st.session_state["_db_persist_warning_shown"] = True
        # Rerun now: repaint the history loop with the new assistant
        # turn. See comment above the if/elif for why the partial
        # branch below omits this call.
        st.rerun()
    elif reply and had_streaming_failure is not None:
        # Partial streamed reply: keep the half-finished text in the
        # transcript so the user can see what the model produced
        # before the connection died. Mark it with a trailing note so
        # the next turn can still be understood in context.
        _partial_content = (
            f"{reply}\n\n"
            f"_⚠️ Reply interrupted: {had_streaming_failure}_"
        )
        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": _partial_content,
            }
        )
        # --- Chat history persistence (assistant, partial) --------------
        # We persist the same combined string the user sees, so reload
        # shows the interrupted-turn marker in the same place.
        if (
            st.session_state.get("db_initialised", False)
            and st.session_state.get("active_chat_id")
        ):
            try:
                _storage.append_message(
                    chat_id=st.session_state["active_chat_id"],
                    role="assistant",
                    content=_partial_content,
                )
            except Exception as exc:  # noqa: BLE001
                # The success path already may have surfaced a warning;
                # swallow the second one silently rather than spamming
                # the user. The next pass-1 warning still fires if the
                # user submits another turn and we lose it again.
                _ = exc


# --- Recon (Phase 15) slash-command handler ----------------------------------
# ``/recon <target> [scope=<token>]`` is a tool turn, not a conversation
# turn — we short-circuit the LLM path and dispatch straight to the
# orchestrator. The orchestrator fans out across DNS / URL / IP / WHOIS /
# crt.sh in parallel, normalises everything, and returns a ReconReport.
#
# Three behaviours to preserve:
#   1. Live progress — the user sees each tool name appear as it
#      completes (no blank stare at a spinner for 30 s). We use
#      ``st.status`` as the container and ``st.write`` inside it.
#   2. Audit log — every request, blocked or successful, lands in the
#      ``recon_requests`` table via ``storage.log_recon_request``. The
#      sidebar's "Recent recon" panel reads from the same table.
#   3. Refusal safety — ``TargetBlockedError`` is raised by the safety
#      layer for public-internet targets; we catch it, show a clear
#      refusal, log ``status="blocked"``, and skip the LLM entirely.
#
# Recon turns never enter ``st.session_state["messages"]`` — the chat
# transcript stays clean (chat = conversation, recon = tool).
def _handle_recon_command(cmd: "parse_recon_command.__class__ | object") -> bool:
    """Run a ``/recon`` turn and return ``True`` if it was handled.

    The argument is the :class:`ReconCommand` dataclass returned by
    ``parse_recon_command``. Returning ``True`` tells the input driver
    to skip the LLM path; ``False`` means the caller should fall through
    to normal chat dispatch (currently we never return False here, but
    keeping the signature makes the call site self-documenting).
    """
    import time as _time

    target = cmd.target
    scope_token = cmd.scope_token or "engagement"
    started_at = _time.perf_counter()
    # Bind recon turns to the currently-open chat so the audit log and
    # the persisted transcript share the same chat row. ``create_chat``
    # returns a uuid4 hex string, and ``append_message`` / ``log_recon_request``
    # expect the same shape. We coerce defensively so a buggy upstream
    # that stored an int cannot crash the audit log.
    _recon_chat_id = st.session_state.get("active_chat_id") or None
    if not isinstance(_recon_chat_id, str):
        _recon_chat_id = None

    # Header chip — surfaces the parsed scope so the user can confirm
    # the engine saw what they typed before any network call goes out.
    st.markdown(
        f"""<div class="recon-header">
            <span class="recon-icon">🔍</span>
            <span class="recon-target">{target}</span>
            <span class="recon-scope">scope: {scope_token}</span>
        </div>""",
        unsafe_allow_html=True,
    )

    # Status container — collapses to a one-line "✅ Recon complete" once
    # the orchestrator returns. We update the label as each stage fires.
    #
    # ``stream_recon`` yields one tuple per tool *as it finishes*, then a
    # final ``("report", ReconReport)`` sentinel:
    #
    #     ("dns", ToolResult)
    #     ("whois", ToolResult)
    #     ...
    #     ("report", ReconReport)
    #
    # The first element is the tool name (or the literal string
    # ``"report"`` for the sentinel); the second element is the result
    # object — a :class:`ToolResult` dataclass for intermediate stages
    # and a :class:`ReconReport` for the final sentinel. We log each
    # tool name into the status box and capture the report on the final
    # iteration. The previous implementation assumed a dict-shaped
    # payload (``{"kind": "report", "report": ...}``) and therefore
    # missed every report — the UI rendered only an empty status box.
    with st.status("🔍 Recon in progress…", expanded=True) as status_box:
        final_report = None
        blocked_message: str | None = None
        try:
            for stage_label, payload in stream_recon(
                target, scope_token=scope_token
            ):
                # Final sentinel: ``stage_label == "report"`` and
                # ``payload`` is the :class:`ReconReport` itself.
                if stage_label == "report":
                    final_report = payload
                    status_box.write("📋 Report ready")
                else:
                    # Per-tool stage: pretty-print duration + ok/err.
                    _ok = getattr(payload, "ok", None)
                    _ms = getattr(payload, "duration_ms", 0)
                    _icon = "✅" if _ok else "⚠️"
                    status_box.write(f"{_icon} {stage_label} · {_ms} ms")
        except TargetBlockedError as exc:
            blocked_message = str(exc)
            status_box.update(
                label=f"⛔ Blocked: {blocked_message}", state="error"
            )
        except Exception as exc:  # noqa: BLE001 — surface anything
            status_box.update(label=f"❌ Recon failed: {exc}", state="error")
            _storage.log_recon_request(
                target=target,
                tool="orchestrator",
                scope_token=scope_token,
                chat_id=_recon_chat_id,
                status="error",
                duration_ms=int((_time.perf_counter() - started_at) * 1000),
                result_excerpt=str(exc)[:500],
            )
            st.error(f"Recon failed: {exc}")
            return True

        if blocked_message is not None:
            _storage.log_recon_request(
                target=target,
                tool="orchestrator",
                scope_token=scope_token,
                chat_id=_recon_chat_id,
                status="blocked",
                duration_ms=int((_time.perf_counter() - started_at) * 1000),
                result_excerpt=blocked_message[:500],
            )
            st.error(
                f"Recon refused for **{target}** "
                f"(scope `{scope_token}`): {blocked_message}"
            )
            st.caption(
                "Allowed scope tokens are: engagement, ctf, lab, labs, "
                "redteam, personal-lab, bugbounty. "
                "Production targets require explicit written authorisation."
            )
            return True

    if final_report is None:
        # Orchestrator finished without a report — defensive.
        st.warning("Recon finished but no report was produced.")
        _storage.log_recon_request(
            target=target,
            tool="orchestrator",
            scope_token=scope_token,
            chat_id=_recon_chat_id,
            status="empty",
            duration_ms=int((_time.perf_counter() - started_at) * 1000),
            result_excerpt="orchestrator returned no report",
        )
        return True

    duration_ms = int((_time.perf_counter() - started_at) * 1000)
    _storage.log_recon_request(
        target=target,
        tool="orchestrator",
        scope_token=scope_token,
        chat_id=_recon_chat_id,
        status="ok",
        duration_ms=duration_ms,
        result_excerpt=render_report_json(final_report)[:500],
    )

    # Render: Markdown inline for reading, JSON download for export.
    _report_md = render_report_markdown(final_report)
    st.markdown(_report_md)
    st.download_button(
        label="⬇️ Download report (JSON)",
        data=render_report_json(final_report),
        file_name=f"recon_{_safe_filename(target)}.json",
        mime="application/json",
        key=f"recon_dl_{hash(target + scope_token) & 0xFFFFFFFF}",
    )

    # --- Persist the recon turn in the visible chat transcript -------------
    # A recon turn is two messages: the operator's slash command
    # (rendered as a user bubble so the audit trail reads as a
    # conversation) and the report body (rendered as an assistant
    # bubble so reload shows the same Markdown the user just saw).
    # We deliberately use ``st.session_state["messages"]`` here so the
    # bubble above the input re-renders on the next rerun; storage is
    # the durability layer for when the user closes the tab.
    _user_recon_turn = (
        f"/recon {target} scope:{scope_token}"
    )
    st.session_state["messages"].append(
        {"role": "user", "content": _user_recon_turn}
    )
    st.session_state["messages"].append(
        {"role": "assistant", "content": _report_md}
    )
    # Best-effort persistence — DB failures never block the recon flow,
    # they just mean the report won't survive a page reload.
    if (
        st.session_state.get("db_initialised", False)
        and st.session_state.get("active_chat_id")
    ):
        try:
            _storage.append_message(
                chat_id=st.session_state["active_chat_id"],
                role="user",
                content=_user_recon_turn,
            )
            _storage.append_message(
                chat_id=st.session_state["active_chat_id"],
                role="assistant",
                content=_report_md,
            )
        except Exception as exc:  # noqa: BLE001
            # Match the pattern used by chat-turn persistence: a one-shot
            # warning so the user knows recon ran but the transcript is
            # not being saved this session.
            st.session_state.setdefault(
                "_db_persist_warning_shown", False
            )
            if not st.session_state["_db_persist_warning_shown"]:
                st.warning(
                    f"Could not persist the recon transcript: {exc}",
                    icon="⚠️",
                )
                st.session_state["_db_persist_warning_shown"] = True
    return True


def _safe_filename(s: str) -> str:
    """Make a string safe to use as a filename: alnum + dot/underscore/dash."""
    out = "".join(
        c if c.isalnum() or c in "._-" else "_" for c in s.strip()
    )
    return out or "target"


# --- Two-pass continuation ---------------------------------------------------
# This is the missing piece. Pass 1 (above) sets ``pending_request`` and
# calls ``st.rerun()``. On the rerun, the chat_input widget has already
# been consumed and returns "", and ``pending_prompt`` is None — so
# nothing in the input drivers below would call ``_ask()`` again, and
# the Thinking… placeholder, the st.toast, and the status-line marker
# would all sit in session_state and never be reached.
#
# We fix it by explicitly driving pass 2 *here*, before any input
# widget has a chance to run. The ``None`` argument is a signal: in
# pass 2, the function ignores its argument and pops ``pending_request``
# instead (see _ask() docstring). This block is a no-op on the very
# first run because ``pending_request`` is not yet set.
if st.session_state.get("pending_request"):
    _ask(None)


# Handle an "example prompt" click from the sidebar.
pending = st.session_state.pop("pending_prompt", None)
if pending:
    _ask(pending)
    st.rerun()


# The chat input at the bottom of the page. `accept_file="multiple"`
# turns the widget into a chat-completions-style uploader; the user can
# attach one or more files (text or image) alongside their question.
# We restrict the file picker to a small set of formats the helper can
# meaningfully inline: plain text, common source/log formats, and the
# common image types. Binary blobs (executables, archives) are still
# allowed because the browser needs *some* whitelist to offer, but the
# helper will summarise them rather than inline them.
_ACCEPTED_FILE_TYPES: list[str] = [
    # Plain text + common source/log formats. Inlined verbatim.
    "txt", "md", "log", "csv", "json", "yaml", "yml", "xml",
    "py", "js", "ts", "html", "css", "sql", "sh", "ps1", "env", "ini",
    # Images — base64-encoded into a data URL and sent as an
    # ``image_url`` content part to a vision-capable model.
    # (Whitelisted formats match ``app.file_processor._SUPPORTED_IMAGE_MIMES``.)
    "png", "jpg", "jpeg",
    # PDFs — text is extracted page-by-page and inlined as text.
    "pdf",
]
# --- Claude-style chatbox row ---------------------------------------------
# Render the compact model chip directly above the chat input, hugging
# the left edge the same way Claude's "Sonnet 5 ▾" pill hugs the input
# row. The chip is a popover that opens the full model picker so the
# user can swap models without scrolling back up to the sidebar. Both
# the popover and the sidebar dropdown write to
# ``st.session_state["model"]`` so they stay in lock-step.
#
# Layout (no second column — a stranded right-side hint just left a
# gap in the screenshot):
#
#   ┌─────────────────────────┐
#   │ Gemma 4 31B (default) ▾ │   <- chip, popover opens picker
#   └─────────────────────────┘
#   ┌─────────────────────────┐
#   │ Ask a cybersecurity …   │   <- chat input (full width)
#   └─────────────────────────┘
#
# The chip's button uses ``use_container_width=False`` so it sizes to
# its content; placing it at the top of a single-column container
# leaves its right edge anchored to the chat input's left edge below.
_render_chatbox_model_picker()

# --- Chat input with media-file safety net ---------------------------------
# ``st.chat_input(accept_file="multiple")`` is the standard way to let
# users attach files alongside their question, but Streamlit stores the
# file bytes in a per-session in-memory store keyed by content hash.
# When the server restarts (or the Streamlit worker is recycled) and
# the user keeps the tab open, the browser still echoes the previous
# file's media ID back to the new server, which then raises
# ``MediaFileStorageError`` from inside the widget's render call —
# blanking the whole page with a server-side traceback.
#
# The guard below catches that exact failure mode, shows a one-time
# recovery banner, clears the pending request so a partial answer is
# not finalised against lost data, and swaps in a plain text
# ``st.chat_input`` for the rest of the session. The downgrade is
# sticky (per-session) so the next rerun is silent; a manual "Reset
# attachments" button re-enables the file picker for users who
# restarted the server intentionally and want to attach again.
if st.session_state.get(_SESSION_STATE_MEDIA_ERROR):
    st.warning(
        "An attachment from a previous server session is no longer "
        "available — the file input has been temporarily disabled. "
        "Re-attach your file (or re-type the question) and the "
        "next reply will work normally.",
        icon="📎",
    )
    if st.button(
        "Reset attachments (re-enable file picker)",
        key="_reset_attachments_btn",
    ):
        st.session_state.pop(_SESSION_STATE_MEDIA_ERROR, None)
        st.rerun()
    # Plain text fallback — no accept_file, so the chat input never
    # touches the media store and the original error cannot fire
    # again this session.
    user_chat = st.chat_input(
        "Ask a cybersecurity question…",
    )
else:
    try:
        user_chat = st.chat_input(
            "Ask a cybersecurity question… (attach files with the 📎 button)",
            accept_file="multiple",
            file_type=_ACCEPTED_FILE_TYPES,
        )
    except _MediaFileStorageError as _media_exc:
        # The widget's own re-serialisation step raised. Mark the
        # session so the *next* render uses the text-only fallback,
        # surface a one-time friendly banner, drop the in-flight
        # request so a partial reply is not finalised against lost
        # file data, and rerun into the safe path.
        st.session_state[_SESSION_STATE_MEDIA_ERROR] = True
        st.session_state.pop("pending_request", None)
        st.session_state.pop("pending_started_at", None)
        # Log the raw exception for debugging without exposing the
        # full storage error to the user (the underlying message
        # embeds the lost file's media ID, which is meaningless to
        # the user but a useful breadcrumb in the server log).
        logging.getLogger(__name__).warning(
            "Media file storage raised during chat input render: %s",
            _media_exc,
        )
        st.rerun()
if user_chat:
    # `st.chat_input` returns a `ChatInputValue` when `accept_file` is
    # set, with `.text` (str) and `.files` (list[UploadedFile]).
    #
    # Two paths through the helper:
    #   * Text-only turn  -> ``str``  (unchanged behaviour).
    #   * Image-bearing  -> ``list[dict[str, object]]`` of content parts,
    #     base64-encoded via ``app.file_processor.process_image`` and
    #     passed straight to the model.
    # Text-only files (PDF, plain source) are summarised by the helper
    # and folded into the textual prompt; non-textual files that fail to
    # process become a stub note so the user knows the attachment was
    # dropped.
    uploaded_text = getattr(user_chat, "text", None) or str(user_chat)
    uploaded_files = getattr(user_chat, "files", None) or []

    # --- /recon slash-command interception ----------------------------
    # ``/recon <target> [scope=<token>]`` is a tool turn, not a
    # conversation turn. We parse first; if it's a recon turn, dispatch
    # to the orchestrator and skip the LLM path entirely. Attached
    # files are ignored on a recon turn (the target is what matters).
    _recon_cmd = parse_recon_command(uploaded_text)
    if _recon_cmd is not None:
        _handle_recon_command(_recon_cmd)
        st.rerun()
    content = build_user_turn_content(
        uploaded_text,
        uploaded_files,
        image_processor=_file_processor.process_image,
        pdf_processor=_file_processor.process_pdf,
    )
    # The display bubble (and the textual portion of the model prompt for
    # image-bearing turns) is built by the long-standing text-only helper.
    # This keeps the chat-renderer contract stable and lets the existing
    # `build_user_turn_text`-based tests pin the user-visible wording.
    prompt_text = build_user_turn_text(
        uploaded_text,
        uploaded_files,
        pdf_processor=_file_processor.process_pdf,
    )
    has_images = isinstance(content, list)
    # Auto-swap the model when the user attached images but the
    # currently-selected model can't see them. The curated list is
    # computed once per turn; the helper falls back to a hard-coded
    # default when no curated vision model is available.
    vision_ids = [
        m["id"]
        for m in FREE_MODEL_CHOICES
        if model_supports_vision(m["id"])
    ]
    # ``requested_model`` is the model id the user picked for this turn.
    # When a custom-id override is active (Advanced expander) that
    # override wins over the curated dropdown — the dropdown is
    # rendered disabled while the override is set, so by the time we
    # reach this line the override is the only signal we have.
    requested_model = (
        st.session_state.get("custom_model_override")
        or st.session_state["model"]
    )
    effective_model, swapped = select_model_for_request(
        requested_model,
        has_images=has_images,
        vision_model_ids=vision_ids,
    )
    # IMPORTANT: do NOT overwrite ``st.session_state['model']`` with the
    # auto-swapped vision id. The vision model is the right choice *for
    # the current turn*, but the user's sidebar preference is the right
    # choice for every subsequent text-only turn. Without this guard an
    # earlier image attachment would pin every later PDF/plain-text turn
    # to the vision slot, where it would fail the same 504/idle-timeout
    # cycle the image auto-swap was meant to absorb. The toast below is
    # the only on-screen signal the swap happened; the sidebar dropdown
    # stays where the user left it.
    if swapped:
        st.toast(
            f"Image attached — using {effective_model} for this turn "
            f"(sidebar stays on {requested_model}).",
            icon="🖼️",
        )
    # Persist everything _ask() needs in pass 2. Using a dict (not a
    # bare string) means the cache key, the model override, and the
    # multimodal content list all survive the rerun. The signature is
    # a coarse content hash used only for dedup / cache stability.
    if has_images:
        signature_src = json.dumps(content, sort_keys=True, default=str)
    else:
        signature_src = content
    pending = {
        "text": prompt_text,
        "content": content,
        "model": effective_model,
        # ``requested_model`` is what the user *picked* in the sidebar. On
        # an image turn it is usually a text model (e.g. the default
        # Mistral), which becomes the natural text-only fallback if the
        # vision call fails. We persist both so the degrade path can
        # switch back without overwriting the user's choice for the next
        # turn.
        "text_model": requested_model,
        "had_files": bool(uploaded_files),
        "signature": hash(signature_src) & 0xFFFFFFFF,
    }
    # The test-pinned guard handles the common case (user typed something
    # and it isn't whitespace). The `or has_images` clause covers the
    # edge case of an image-only turn where `build_user_turn_text` was
    # not given any text input and might return a short header-only
    # string; the model still needs to see the turn.
    if (prompt_text and prompt_text.strip()) or has_images:
        _ask(pending)
        st.rerun()

# --- Status line ------------------------------------------------------------
total_msgs = len(st.session_state["messages"])
total_chars = _count_chars(st.session_state["messages"])
elapsed = st.session_state.get("last_elapsed")
if elapsed is None:
    timing = "—"
elif elapsed == "thinking":
    # Compute how long we've been waiting so the user sees the seconds tick.
    thinking_secs = time.perf_counter() - float(
        st.session_state.get("pending_started_at", time.perf_counter())
    )
    timing = f"thinking · {thinking_secs:.0f}s"
elif elapsed < 0.5:
    timing = "cache hit"
else:
    timing = f"{elapsed:.1f}s"
cache_size = len(st.session_state.get("response_cache", {}))
_mode_label = (
    "🎯 CTF / Lab mentor" if st.session_state.get("teaching_mode") == "mentor" else "🛡 Defensive"
)
_thinking = elapsed == "thinking"
_verb = "thinking" if _thinking else "ready"
_status_class = "status thinking" if _thinking else "status"
st.markdown(
    f'<div class="{_status_class}">'
    f'<span class="pulse"></span>'
    f'<span>system {_verb}</span>'
    f'<span class="sep">·</span>'
    f'<span>{total_msgs} messages</span>'
    f'<span class="sep">·</span>'
    f'<span>~{total_chars:,} chars</span>'
    f'<span class="sep">·</span>'
    f'<span>mode {_mode_label}</span>'
    f'<span class="sep">·</span>'
    f'<span>last reply: {timing}</span>'
    f'<span class="sep">·</span>'
    f'<span>cache: {cache_size}</span>'
    f'</div>',
    unsafe_allow_html=True,
)

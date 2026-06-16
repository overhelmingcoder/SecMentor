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
)
from app.openrouter import OpenRouterError, chat
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
    _count_chars,
    _friendly_error_message,
    _serialize_for_download,
    _truncate_history,
    build_user_turn_content,
    build_user_turn_text,
    select_model_for_request,
)
from app import file_processor as _file_processor
from app.config import model_supports_vision


# --- Page configuration -------------------------------------------------------

st.set_page_config(
    page_title="SecMentor — AI-Powered Cybersecurity Platform",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Custom CSS for a production-grade cybersecurity platform ----------------
# The look-and-feel is inspired by Microsoft Security Copilot, CrowdStrike
# Falcon, Palo Alto Cortex, Datadog, and GitHub Enterprise: deep navy/slate
# surfaces, restrained blue accents, no neon, no glow, no ChatGPT-isms.
# Layout density is driven by a CSS class on the root container
# (`.layout-compact` / `.layout-standard` / `.layout-wide` / `.layout-full`)
# toggled from the sidebar. All rules below only restyle — no backend.

_CUSTOM_CSS = """
<style>
    /* ---------- Design tokens -----------------------------------------
       Force a light color scheme so the browser's UA dark-mode stylesheet
       can never recolor our bubbles, inputs, or code blocks. This is the
       single most important rule in the whole stylesheet. */
    :root {
        color-scheme: light !important;
        --bg-page:        #eef2f7;
        --bg-surface:     #ffffff;
        --bg-surface-2:   #f8fafc;
        --bg-sidebar:     #0b1220;
        --bg-sidebar-2:   #111a2e;
        --border-subtle:  #e2e8f0;
        --border-strong:  #cbd5e1;
        --text-primary:   #0f172a;
        --text-secondary: #334155;
        --text-muted:     #64748b;
        --text-inverse:   #e2e8f0;
        --accent:         #1d4ed8;
        --accent-soft:    #dbe5ff;
        --accent-strong:  #1e3a8a;
        --success:        #047857;
        --warn:           #b45309;
        --user-bubble:    #1d4ed8;
        --user-bubble-2:  #1e40af;
        --shadow-sm:      0 1px 2px rgba(15, 23, 42, 0.06);
        --shadow-md:      0 4px 14px rgba(15, 23, 42, 0.08);
        --radius-sm:      8px;
        --radius-md:      12px;
        --radius-lg:      16px;

        --container-max:  920px;
        --content-pad:    1.25rem;
        --bubble-max:     82%;
        --bubble-pad-y:   0.85rem;
        --bubble-pad-x:   1.1rem;
        --row-gap:        0.65rem;
        --hero-pad:       1.75rem 2rem;
    }

    /* Layout density variants. The radio in the sidebar adds one of these
       classes to a wrapper <div> right under <body> via a small
       st.markdown below. Everything else reads the CSS variables above. */
    body.layout-compact  { --container-max: 720px;  --content-pad: 0.75rem;
                           --bubble-max: 76%; --bubble-pad-y: 0.55rem;
                           --bubble-pad-x: 0.8rem; --row-gap: 0.35rem;
                           --hero-pad: 1rem 1.1rem; }
    body.layout-standard { --container-max: 920px;  --content-pad: 1.25rem;
                           --bubble-max: 82%; --bubble-pad-y: 0.75rem;
                           --bubble-pad-x: 1rem; --row-gap: 0.55rem;
                           --hero-pad: 1.5rem 1.75rem; }
    body.layout-wide     { --container-max: 1180px; --content-pad: 1.75rem;
                           --bubble-max: 88%; --bubble-pad-y: 0.85rem;
                           --bubble-pad-x: 1.1rem; --row-gap: 0.7rem;
                           --hero-pad: 1.75rem 2rem; }
    body.layout-full     { --container-max: 100%;   --content-pad: 2.25rem;
                           --bubble-max: 92%; --bubble-pad-y: 0.95rem;
                           --bubble-pad-x: 1.2rem; --row-gap: 0.8rem;
                           --hero-pad: 2rem 2.25rem; }

    /* ---------- Global page chrome ------------------------------------ */
    /* `!important` everywhere below to beat Streamlit's own high-
       specificity theme rules and the browser's prefers-color-scheme
       dark UA stylesheet. */
    html, body, .stApp {
        background: var(--bg-page) !important;
        color: var(--text-primary) !important;
    }
    body, .stApp, .stApp * { color-scheme: light !important; }
    .stApp header[data-testid="stHeader"] {
        background: linear-gradient(90deg, #0b1220 0%, #15233f 50%, #0b1220 100%) !important;
        height: 3.25rem; box-shadow: var(--shadow-sm);
    }
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }

    /* ---------- Main column ------------------------------------------- */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 6rem;          /* leave room for the floating chat input */
        padding-left: var(--content-pad);
        padding-right: var(--content-pad);
        max-width: var(--container-max);
        color: var(--text-primary);
    }
    /* Force dark text on every direct Streamlit container in the main
       area. Without this, the chat_input's inner span and any caption
       inherits a white color from somewhere in the cascade. */
    .main .block-container,
    .main .block-container p,
    .main .block-container span,
    .main .block-container div,
    .main .block-container li,
    .main .block-container label,
    .main .block-container small { color: var(--text-primary) !important; }
    .main .block-container h1,
    .main .block-container h2,
    .main .block-container h3 { color: var(--text-primary) !important; }

    /* ---------- Sidebar ----------------------------------------------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--bg-sidebar) 0%, var(--bg-sidebar-2) 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] * { color: var(--text-inverse) !important; }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: #c7d2fe !important; letter-spacing: 0.01em; }
    section[data-testid="stSidebar"] .stMarkdown p { color: #cbd5e1 !important; }
    section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.08); }
    /* Sidebar brand block */
    .sm-brand {
        display: flex; align-items: center; gap: 0.6rem;
        padding: 0.25rem 0 0.5rem 0;
    }
    .sm-brand .logo {
        width: 32px; height: 32px;
        display: inline-flex; align-items: center; justify-content: center;
        background: linear-gradient(135deg, #1d4ed8 0%, #4338ca 100%);
        color: #fff; border-radius: 8px;
        box-shadow: 0 2px 6px rgba(29, 78, 216, 0.35);
        font-size: 0.95rem;
    }
    .sm-brand .name { font-size: 1.1rem; font-weight: 700; color: #f8fafc; letter-spacing: 0.01em; }
    .sm-brand .tag  { font-size: 0.7rem; color: #94a3b8; letter-spacing: 0.04em; text-transform: uppercase; }
    /* Sidebar section headings */
    section[data-testid="stSidebar"] h3 {
        font-size: 0.72rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #93c5fd !important;
        margin-top: 0.75rem !important;
        margin-bottom: 0.35rem !important;
        font-weight: 600 !important;
    }
    /* Sidebar inputs — dark surfaces, light text */
    section[data-testid="stSidebar"] .stTextInput input,
    section[data-testid="stSidebar"] .stTextArea textarea,
    section[data-testid="stSidebar"] .stNumberInput input,
    section[data-testid="stSidebar"] .stChatInput input {
        background: rgba(255,255,255,0.05) !important;
        color: #f1f5f9 !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: var(--radius-sm) !important;
    }
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
        background: rgba(255,255,255,0.05) !important;
        color: #f1f5f9 !important;
        border-color: rgba(255,255,255,0.1) !important;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown small,
    section[data-testid="stSidebar"] .stCaption { color: #cbd5e1 !important; }
    section[data-testid="stSidebar"] .stButton > button {
        background: rgba(255,255,255,0.06);
        color: #f1f5f9;
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: var(--radius-sm);
        font-weight: 500;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255,255,255,0.1);
        border-color: rgba(255,255,255,0.2);
    }
    section[data-testid="stSidebar"] .stDownloadButton > button {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
        color: #fff;
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: var(--radius-sm);
    }
    section[data-testid="stSidebar"] .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    }
    /* Sidebar example-prompt buttons — left aligned, lighter */
    section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
        background: transparent;
        border: 1px solid rgba(255,255,255,0.08);
        text-align: left;
        justify-content: flex-start;
        font-size: 0.82rem;
        color: #cbd5e1;
    }

    /* ---------- Hero (replaces the old h1+p block) ------------------- */
    .hero {
        background: linear-gradient(135deg, #0b1220 0%, #15233f 60%, #1e3a8a 100%);
        color: #f8fafc;
        padding: var(--hero-pad);
        border-radius: var(--radius-lg);
        margin-bottom: 1.25rem;
        box-shadow: var(--shadow-md);
        border: 1px solid rgba(255,255,255,0.05);
        position: relative;
        overflow: hidden;
    }
    .hero::after {
        /* Subtle radial accent — no glow, no neon. */
        content: "";
        position: absolute; right: -80px; top: -80px;
        width: 280px; height: 280px;
        background: radial-gradient(circle, rgba(59,130,246,0.18) 0%, rgba(59,130,246,0) 70%);
        pointer-events: none;
    }
    .hero .eyebrow {
        font-size: 0.7rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #93c5fd;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 1.85rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.01em;
        display: flex; align-items: center; gap: 0.55rem;
    }
    .hero h1 .logo {
        width: 36px; height: 36px;
        display: inline-flex; align-items: center; justify-content: center;
        background: linear-gradient(135deg, #2563eb 0%, #4338ca 100%);
        color: #fff; border-radius: 9px;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
        font-size: 1.05rem;
    }
    .hero p.subtitle {
        margin: 0.45rem 0 0 0;
        color: #cbd5e1;
        font-size: 0.98rem;
        line-height: 1.45;
        max-width: 56ch;
    }
    .hero p.tagline {
        margin: 0.6rem 0 0 0;
        color: #93c5fd;
        font-size: 0.85rem;
        letter-spacing: 0.04em;
    }
    .hero .badges {
        display: flex; flex-wrap: wrap; gap: 0.45rem;
        margin-top: 1rem;
    }
    .hero .badge {
        display: inline-flex; align-items: center; gap: 0.35rem;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.14);
        color: #e0e7ff;
        border-radius: 999px;
        padding: 4px 12px;
        font-size: 0.78rem;
        font-weight: 500;
    }
    .hero .badge .dot {
        width: 6px; height: 6px; border-radius: 50%;
        background: #60a5fa;
    }

    /* ---------- Hero (kept dark on purpose — it's the brand surface) */
    .hero, .hero *, .hero p, .hero h1 { color: #f8fafc !important; }
    .hero .eyebrow, .hero .tagline  { color: #93c5fd !important; }
    .hero .badge  { color: #e0e7ff !important; }
    .hero .subtitle { color: #cbd5e1 !important; }

    /* ---------- Status pill (light surface) ------------------------- */
    .status, .status * { color: var(--text-secondary) !important; }

    /* ---------- Chat bubbles -----------------------------------------
       High-specificity color rules so the assistant bubble's text is
       always dark on white, no matter what Streamlit's theme does. */
    .row { display: flex; margin: var(--row-gap) 0; }
    .row.right { justify-content: flex-end; }
    .row.left  { justify-content: flex-start; }

    .bubble-user {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
        color: #ffffff !important;
        padding: var(--bubble-pad-y) var(--bubble-pad-x);
        border-radius: 14px 14px 4px 14px;
        display: inline-block;
        max-width: var(--bubble-max);
        box-shadow: var(--shadow-sm);
        line-height: 1.5;
        font-size: 0.95rem;
        word-wrap: break-word;
    }
    .bubble-user * { color: #ffffff !important; }

    .bubble-assistant {
        background: var(--bg-surface) !important;
        color: var(--text-primary) !important;
        padding: var(--bubble-pad-y) var(--bubble-pad-x);
        border-radius: 14px 14px 14px 4px;
        display: inline-block;
        max-width: var(--bubble-max);
        border: 1px solid var(--border-subtle);
        box-shadow: var(--shadow-sm);
        line-height: 1.6;
        font-size: 0.95rem;
        word-wrap: break-word;
    }
    /* Force every element inside the assistant bubble to inherit dark
       text — Streamlit wraps rendered markdown in a <p> which would
       otherwise pick up the page-level white text. */
    .bubble-assistant,
    .bubble-assistant *,
    .bubble-assistant p,
    .bubble-assistant li,
    .bubble-assistant span,
    .bubble-assistant strong,
    .bubble-assistant em { color: var(--text-primary) !important; }
    .bubble-assistant p:first-child { margin-top: 0; }
    .bubble-assistant p:last-child  { margin-bottom: 0; }
    .bubble-assistant a { color: var(--accent) !important; text-decoration: underline; }
    .bubble-assistant pre {
        background: #0f172a !important;
        color: #e2e8f0 !important;
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
        font-size: 0.82rem;
        overflow-x: auto;
        margin: 0.6rem 0;
        border: 1px solid #1e293b;
    }
    .bubble-assistant pre * { color: #e2e8f0 !important; }
    .bubble-assistant code {
        background: #eef2f7 !important;
        color: #1e293b !important;
        padding: 1px 6px;
        border-radius: 4px;
        font-size: 0.85em;
    }
    .bubble-assistant pre code {
        background: transparent !important; color: #e2e8f0 !important; padding: 0;
    }
    .bubble-assistant table {
        border-collapse: collapse;
        font-size: 0.85rem;
        margin: 0.5rem 0;
    }
    .bubble-assistant th, .bubble-assistant td {
        border: 1px solid var(--border-subtle);
        padding: 4px 8px;
    }
    .bubble-assistant th {
        background: var(--bg-surface-2) !important;
        text-align: left;
        font-weight: 600;
    }
    .bubble-system {
        background: #fef3c7 !important;
        color: #78350f !important;
        padding: 0.5rem 0.85rem;
        border-radius: 10px;
        font-size: 0.85rem;
        border: 1px solid #fde68a;
    }
    .bubble-thinking {
        background: var(--bg-surface) !important;
        color: var(--text-secondary) !important;
        padding: var(--bubble-pad-y) var(--bubble-pad-x);
        border-radius: 14px 14px 14px 4px;
        display: inline-flex; align-items: center; gap: 0.5rem;
        max-width: var(--bubble-max);
        border: 1px solid var(--border-subtle);
        box-shadow: var(--shadow-sm);
        font-size: 0.9rem;
        font-style: italic;
    }
    .bubble-thinking * { color: var(--text-secondary) !important; }

    /* ---------- Role labels (You / SecMentor) ------------------------ */
    .role-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--text-muted) !important;
        margin: 0 0 3px 0;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .row.right .role-label { text-align: right; }
    .row.left  .role-label { text-align: left;  }

    /* ---------- Status pill (bottom) --------------------------------- */
    .status {
        display: inline-flex; align-items: center; flex-wrap: wrap; gap: 0.55rem;
        font-size: 0.78rem;
        color: var(--text-muted);
        background: var(--bg-surface) !important;
        border: 1px solid var(--border-subtle);
        border-radius: 999px;
        padding: 6px 14px;
        margin-top: 0.75rem;
        box-shadow: var(--shadow-sm);
    }
    .status .sep { color: var(--border-strong) !important; }
    .status .pulse {
        display: inline-block; width: 8px; height: 8px;
        background: #10b981; border-radius: 50%;
        margin-right: 2px;
        animation: pulse 1.8s ease-in-out infinite;
    }
    .status.thinking .pulse { background: #f59e0b; }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50%      { opacity: 0.45; }
    }

    /* ---------- Empty state ------------------------------------------ */
    .empty-state {
        background: var(--bg-surface) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        padding: 2rem 1.75rem;
        margin: 1rem 0 1.25rem 0;
        text-align: center;
        box-shadow: var(--shadow-sm);
    }
    .empty-state * { color: var(--text-primary) !important; }
    .empty-state p  { color: var(--text-secondary) !important; }
    .empty-state .icon {
        width: 48px; height: 48px; margin: 0 auto 0.6rem auto;
        display: inline-flex; align-items: center; justify-content: center;
        background: var(--accent-soft) !important; color: var(--accent-strong) !important;
        border-radius: 12px; font-size: 1.3rem;
    }
    .empty-state h3 {
        margin: 0 0 0.35rem 0;
        font-size: 1.05rem;
        font-weight: 600;
    }
    .empty-state .suggestions {
        display: flex; flex-wrap: wrap; gap: 0.4rem; justify-content: center;
        margin-top: 1rem;
    }
    .empty-state .suggestions .chip {
        background: var(--bg-surface-2) !important;
        border: 1px solid var(--border-subtle);
        color: var(--text-secondary) !important;
        font-size: 0.78rem;
        padding: 4px 10px;
        border-radius: 999px;
    }

    /* ---------- Chat input (floating card) --------------------------- */
    /* The chat input is bottom-positioned by Streamlit. Wrap it in a
       card-like surface so it visually anchors the layout. */
    [data-testid="stChatInput"] {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border-strong) !important;
        border-radius: 14px !important;
        box-shadow: 0 6px 24px rgba(15, 23, 42, 0.10) !important;
        padding: 4px 6px !important;
    }
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] input {
        color: var(--text-primary) !important;
        background: transparent !important;
        font-size: 0.96rem !important;
        caret-color: var(--accent) !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--text-muted) !important;
        opacity: 1 !important;
    }
    [data-testid="stChatInput"] button,
    [data-testid="stChatInput"] [data-testid="baseButton-secondary"] {
        background: var(--accent) !important;
        color: #ffffff !important;
        border-radius: 10px !important;
    }

    /* ---------- Tighter captions ------------------------------------- */
    .stCaption, [data-testid="stCaptionContainer"],
    .stCaption p, [data-testid="stCaptionContainer"] p {
        color: var(--text-muted) !important;
    }
    /* Captions inside the main column: a touch darker for legibility. */
    .main .stCaption, .main [data-testid="stCaptionContainer"],
    .main .stCaption p {
        color: var(--text-secondary) !important;
    }

    /* ---------- Streamlit default widget fixes ----------------------- */
    .stAlert p { color: inherit !important; }
    .stAlert[data-baseweb="notification"] * { color: inherit !important; }
</style>
"""
st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


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
    # Last call's elapsed seconds, shown in the status line.
    if "last_elapsed" not in st.session_state:
        st.session_state["last_elapsed"] = None
    # Show a small "You" / "SecMentor" label above each bubble so the user
    # can tell at a glance which side said what. ON by default — that's
    # what every modern chat UI does.
    if "show_role_labels" not in st.session_state:
        st.session_state["show_role_labels"] = True
    # Layout density mode — drives a CSS class on <body> via a small
    # st.markdown in the sidebar block. See _CUSTOM_CSS for token
    # overrides. The default ("standard") matches the original 920px
    # max-width so existing layouts look identical on first load.
    if "layout_mode" not in st.session_state:
        st.session_state["layout_mode"] = "standard"


_init_state()


# --- Sidebar -----------------------------------------------------------------

with st.sidebar:
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
    st.divider()

    # Layout mode toggle. Drives a class on <body> (`.layout-compact`,
    # `.layout-standard`, `.layout-wide`, `.layout-full`) so the CSS
    # tokens in _CUSTOM_CSS pick the right container width, bubble
    # padding, sidebar density, and hero size. Streamlit does not let
    # us set attributes on <body> declaratively, so we ship a tiny
    # client-side script that runs on every page load and applies the
    # class from `localStorage`. The script is idempotent: calling it
    # twice with the same value is a no-op.
    st.markdown("### Layout mode")
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
    _chosen_layout = st.radio(
        "Density",
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

    st.divider()

    st.markdown("### Conversation")
    if st.button("➕  New chat", use_container_width=True):
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

    transcript = _serialize_for_download(
        st.session_state["messages"], model=st.session_state["model"]
    )
    st.download_button(
        "⬇️  Download transcript",
        data=transcript,
        file_name=f"secmentor-transcript-{int(time.time())}.txt",
        mime="text/plain",
        use_container_width=True,
    )

    st.divider()
    st.markdown("### Teaching mode")
    # The radio lets the user switch between the safer four-pillar
    # defensive prompt (default) and the CTF/lab "SecMentor" prompt.
    # When the choice changes, swap the system message in place so the
    # next model call uses the new scope — no need to start a new
    # chat. We compare against the current value explicitly so the
    # rerun that Streamlit triggers after a widget change is the only
    # one that actually mutates session_state (Streamlit's default
    # already writes the new value to the key, so we just react to
    # the delta and re-derive the system message).
    # Order matters: the first entry is the radio's default when no
    # previous-mode index can be derived, so "mentor" is listed first to
    # make the wider, lab-scoped persona the visible default.
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
    # We track the *previous* teaching mode in a separate session_state key
    # ("teaching_mode_previous") instead of reading the live "teaching_mode"
    # key, because the radio widget with `key="teaching_mode"` writes the
    # user's new selection into st.session_state["teaching_mode"] *before*
    # this script body runs on the rerun-after-click. Reading the live key
    # here would always see the new value, so `_chosen_mode != _previous_mode`
    # would never be true and the swap block below would never fire.
    # We only update "teaching_mode_previous" *after* the swap runs, so the
    # comparison is stable across reruns. For a fresh session, fall back to
    # the live key (which is initialized to "mentor" by _init_state — that
    # is the web UI's default teaching mode). The fallback string here must
    # match `_init_state`'s seed so a brand-new session with no previous
    # key is still self-consistent and does not spuriously swap.
    _previous_mode = st.session_state.get(
        "teaching_mode_previous",
        st.session_state.get("teaching_mode", "mentor"),
    )
    _chosen_mode = st.radio(
        "Scope",
        options=_TEACHING_OPTIONS,
        format_func=lambda key: _TEACHING_LABELS.get(key, key),
        index=_TEACHING_OPTIONS.index(_previous_mode)
        if _previous_mode in _TEACHING_OPTIONS
        else 0,
        key="teaching_mode",
        help="Pick the scope for the next model call. "
             "Switching mid-chat swaps the system prompt in place.",
    )
    st.caption(_TEACHING_HELP.get(_chosen_mode, ""))
    # If the radio produced a new value, swap the live system message
    # so subsequent calls in this session use the new prompt. We do
    # NOT clear the chat — only the system role at index 0 changes.
    # We also drop the response cache because the *system prompt* is
    # an implicit input to every reply and stale cache entries from
    # the old scope would be confusing.
    if _chosen_mode != _previous_mode:
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

    st.divider()
    st.markdown("### Model settings")
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
    # Model selector. Curated free-tier list is the easy default;
    # the expander below is for users who want a different id.
    if FREE_MODEL_CHOICES:
        _labels = [m["label"] for m in FREE_MODEL_CHOICES]
        _current = st.session_state["model"]
        _current_label = next(
            (m["label"] for m in FREE_MODEL_CHOICES if m["id"] == _current),
            _labels[DEFAULT_SELECTED_MODEL_INDEX],
        )
        _chosen_label = st.selectbox(
            "Model",
            _labels,
            index=_labels.index(_current_label),
            help="Free OpenRouter models. The engine response cache keys "
                 "on the model id, so switching gives you a clean cache miss.",
        )
        st.session_state["model"] = next(
            m["id"] for m in FREE_MODEL_CHOICES if m["label"] == _chosen_label
        )
        # Show the chosen id + a one-line blurb so the user always knows
        # exactly what they're talking to.
        _chosen = next(
            m for m in FREE_MODEL_CHOICES if m["label"] == _chosen_label
        )
        st.caption(f"`{_chosen['id']}` — {_chosen['blurb']}")
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
        with st.expander("Advanced: custom model ID", expanded=False):
            _custom = st.text_input(
                "Custom OpenRouter model",
                value="",
                placeholder=OPENROUTER_MODEL,
                help="Any OpenRouter model id. Leave blank to use the "
                     "selected free model. Overrides the dropdown until "
                     "you clear it.",
            ).strip()
            if _custom:
                st.session_state["model"] = _custom
    else:
        # Curated list is empty (shouldn't happen, but be defensive).
        st.session_state["model"] = st.text_input(
            "OpenRouter model",
            value=st.session_state["model"],
            help="Any OpenRouter model ID. Free models end with :free.",
        )
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

    st.divider()
    st.markdown("### Try a question")
    for prompt in EXAMPLE_PROMPTS:
        if st.button(prompt, key=f"ex_{prompt[:24]}", use_container_width=True):
            st.session_state["pending_prompt"] = prompt
            st.rerun()

    st.divider()
    with st.expander("ℹ️  About / scope", expanded=False):
        st.markdown(
            """
**Four pillars** this assistant is engineered to teach:
- 🛡️ **Defensive security** — threat modeling, IR, hardening, IAM, network.
- 🔁 **DevSecOps** — secure SDLC, SAST/DAST, supply chain, secrets, K8s.
- 🧠 **AI / ML security** — prompt injection, OWASP LLM Top 10, agent safety.
- 🎯 **Offensive-security education** — *concept-level*: structure of an attack,
  why it works, what defeats it. Not turn-key exploits.

**Teaching mode** (sidebar above) lets you swap the system prompt mid-session:
- *Defensive (4 pillars)* — default. Concept-level only.
- *CTF / Lab mentor* — unlocks lab scope (HTB, THM, PortSwigger, DVWA, WebGoat).
  May produce runnable exploit snippets framed for the lab, always paired
  with the defensive countermeasure. See `docs/technical_write_up.md` Decision 6.

**Out of scope (both modes):** working exploit code against a specific real
system, malware, droppers, C2, payloads against specific real WAFs/EDRs/MFAs,
brand-new malware strains, critical-infrastructure targets.

|**Where to practice legally:** HackTheBox, TryHackMe, PortSwigger Academy,
DVWA, WebGoat, PicoCTF.
            """,
            unsafe_allow_html=True,
        )

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
        st.markdown(
            f'<div class="row left">{_label_html()}'
            f'<div class="bubble-assistant">',
            unsafe_allow_html=True,
        )
        st.markdown(content)
        st.markdown("</div></div>", unsafe_allow_html=True)


def _render_friendly_error(exc: BaseException, model: str) -> None:
    """Show a user-readable error banner and the raw exception in a toggle.

    The raw ``OpenRouterError`` message contains the full upstream JSON
    payload, which is great for debugging and awful for end users. We
    pass the exception through ``_friendly_error_message`` to get a
    short, actionable headline + body, and we keep the raw text
    available behind ``st.exception`` so a developer can still inspect
    it with one click.
    """
    headline, body = _friendly_error_message(exc, model)
    st.error(headline)
    st.caption(body)
    with st.expander("Raw error (for debugging)", expanded=False):
        st.exception(exc)


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
        st.session_state["pending_request"] = request
        st.session_state["pending_started_at"] = time.perf_counter()
        st.rerun()
        return

    # Pass 2: we are on the rerun triggered above. Pull the request
    # out, run the model, append the reply, clear the flag, rerun
    # again so the assistant bubble actually appears.
    request = st.session_state.pop("pending_request", None)
    started = st.session_state.pop("pending_started_at", time.perf_counter())
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
    #    and update it as the full reply comes back. (OpenRouter's free tier
    #    doesn't expose token-level SSE, so we wait for the full text and
    #    then reveal it in one smooth block.)
    show_label = bool(st.session_state.get("show_role_labels", True))
    label_html = (
        '<div class="role-label">SecMentor</div>' if show_label else ""
    )
    placeholder = st.empty()
    placeholder.markdown(
        f'<div class="row left">{label_html}'
        f'<div class="bubble-thinking">'
        f'<span class="pulse"></span>Thinking…</div></div>',
        unsafe_allow_html=True,
    )
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
    try:
        if request.get("had_files"):
            # Pinned-model path: bypass the router so the model id is
            # guaranteed. The OpenRouter client still picks the API
            # key from the env (or the first slot's key) so we are
            # not free-loading the user.
            reply = chat(
                messages_for_api,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            # Normal path: let the router pick the slot.
            reply = router.chat(
                messages_for_api,
                temperature=temperature,
                max_tokens=max_tokens,
            )
    except AllSlotsExhaustedError as exc:
        # Every (key, model) slot failed in turn. This is a harder
        # error than a single-model 429: render a dedicated banner
        # that tells the user the entire pool is exhausted and lists
        # the slot labels (with redacted keys) so they can see exactly
        # what was tried.
        placeholder.empty()
        st.session_state["last_elapsed"] = None
        st.error(
            "All router slots are exhausted. Every configured "
            "(key, model) pair failed — most likely the daily per-account "
            "free-tier cap has been hit on every account."
        )
        st.caption(
            "Tried: " + ", ".join(router.slot_labels())
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
        # (the router normally handles these internally; reaching here
        # means something unexpected — e.g. a bad 4xx we don't catch,
        # or the helper raised before the router could rotate). Show
        # the user the model id the user *asked* for, not whatever the
        # router was actually trying, since that's what they recognise.
        placeholder.empty()
        st.session_state["last_elapsed"] = None
        _render_friendly_error(exc, model)
        return
    elapsed = time.perf_counter() - started
    st.session_state["last_elapsed"] = elapsed
    st.session_state["response_cache"][cache_key] = reply

    # Commit the reply to history. The history-render loop at the top of
    # the script (the `for message in st.session_state["messages"][1:]`
    # block) is what actually paints the assistant bubble to the screen.
    # We rerun to re-execute the script top-to-bottom with the new turn
    # in session_state. The placeholder above only ever showed the
    # Thinking… state; the revealed answer lives in the history list.
    st.session_state["messages"].append({"role": "assistant", "content": reply})
    st.rerun()


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
user_chat = st.chat_input(
    "Ask a cybersecurity question… (attach files with the 📎 button)",
    accept_file="multiple",
    file_type=_ACCEPTED_FILE_TYPES,
)
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
    requested_model = st.session_state["model"]
    effective_model, swapped = select_model_for_request(
        requested_model,
        has_images=has_images,
        vision_model_ids=vision_ids,
    )
    if swapped:
        st.session_state["model"] = effective_model
        st.toast(
            f"Image attached — switched to {effective_model} for vision.",
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

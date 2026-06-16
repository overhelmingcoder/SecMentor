"""Application configuration.

Loads secrets and settings from the .env file exactly once, at import time,
and exposes them as module-level constants. Every other module in the project
imports from here — no other file should call dotenv, os.getenv, or hardcode
the model name, API key, or endpoint URL.

If ``OPENROUTER_API_KEY`` is missing the module raises a clear error at import
time. That is intentional: we want the app to fail loudly at startup, not
silently send unauthenticated requests.

Multi-key + multi-model support
-------------------------------

The OpenRouter *free* tier caps usage per *account*, not per API key. So to
keep the demo running we support up to **five** keys (``OPENROUTER_API_KEY``,
``OPENROUTER_API_KEY_2``, ``OPENROUTER_API_KEY_3``, ``OPENROUTER_API_KEY_4``,
``OPENROUTER_API_KEY_5``) and any number of model ids (``OPENROUTER_MODELS``
as a comma-separated list; falls back to the single ``OPENROUTER_MODEL`` if
not set). Use ``iter_api_keys()`` and ``iter_models()`` to read these lists
— do not hard-code the maximum key count anywhere else.

Pool size: with 5 keys × 1 model you get a 5-slot pool; with 5 keys × 3
models you get 15 slots. The router walks the slots round-robin, so each
key gets equal exposure and the per-account daily cap takes 5× longer to
exhaust.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv

# --- Load .env exactly once ---------------------------------------------------
# find_dotenv() walks up from this file's location until it finds a .env.
# load_dotenv() reads it and copies its values into os.environ WITHOUT
# overwriting values that are already set in the real environment.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=False)


# --- Required values ----------------------------------------------------------
def _require_env(name: str) -> str:
    """Return the env var or raise a clear error at startup."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Add it to your .env file (see .env.example)."
        )
    return value


OPENROUTER_API_KEY: str = _require_env("OPENROUTER_API_KEY")
# Default single model id. Kept required (the existing single-model
# setup is still the supported minimum) but iter_models() can override
# the pool with OPENROUTER_MODELS.
OPENROUTER_MODEL: str = _require_env("OPENROUTER_MODEL")
OPENROUTER_BASE_URL: str = os.getenv(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1/chat/completions",
)

# --- Optional values ----------------------------------------------------------
OPENROUTER_APP_NAME: str = os.getenv(
    "OPENROUTER_APP_NAME", "AI Security Chatbot (Stage 1)"
)


# --- Multi-key + multi-model helpers -----------------------------------------
# We support up to five keys (1 required + 4 optional). A higher cap is
# easy to add later; 5 is enough to keep a single-user demo running
# through the daily free-tier cap for the whole day, and adding slots 6+
# would mostly add boilerplate with no UX benefit. The list is read
# lazily via iter_api_keys() so tests can monkeypatch the env at runtime.
_MAX_KEY_SLOTS: int = 5


def iter_api_keys() -> Iterator[str]:
    """Yield every non-empty ``OPENROUTER_API_KEY[_N]`` in slot order.

    Slot 1 (``OPENROUTER_API_KEY``) is required and is the only one
    that triggers a startup error if missing. Slots 2..5 are optional
    and are silently skipped if not set or set to the empty string.

    Order is stable: slot 1 first, then 2, 3, 4, 5. The router uses
    this order to pick the *primary* key (slot 1) for the first call.

    To raise the cap further, bump ``_MAX_KEY_SLOTS`` here; the rest of
    the codebase reads the constant through this iterator.
    """
    yield OPENROUTER_API_KEY
    for n in range(2, _MAX_KEY_SLOTS + 1):
        value = os.getenv(f"OPENROUTER_API_KEY_{n}")
        if value:
            yield value


def iter_models() -> Iterator[str]:
    """Yield every model id the router should use.

    Priority:

    1. ``OPENROUTER_MODELS`` env var, a comma-separated list. Whitespace
       around commas is stripped; empty entries are skipped. This is
       the override path — set it in .env when you want to pin the
       demo to a specific list of free models.
    2. The single ``OPENROUTER_MODEL`` env var (the original behaviour).
       Always yielded exactly once so a single-model deployment keeps
       working with no .env change.

    Order is preserved within (1) so a deliberate primary/secondary
    ordering in the .env survives into the router pool.
    """
    raw = os.getenv("OPENROUTER_MODELS")
    if raw:
        for entry in raw.split(","):
            cleaned = entry.strip()
            if cleaned:
                yield cleaned
        return
    yield OPENROUTER_MODEL


# --- Vision-capable model allow-list ------------------------------------------
# This is the single source of truth for "which free-tier models can
# see images". The list is intentionally conservative: only models
# we have confirmed accept the OpenRouter ``image_url`` part shape on
# the free tier. Adding a model here that does NOT support vision
# will produce 400 Bad Request errors at request time — fail-safe in
# production, but annoying. The list is keyed by the exact model id
# the provider returns, including the ``:free`` suffix.
#
# The view layer (``web/streamlit_app.py``) reads this via
# ``model_supports_vision()`` to decide whether to swap the user's
# selected model for a vision-capable one when an image is attached.
# Keeping the set in ``app/`` (not in the view) means tests can patch
# it without dragging in Streamlit.
#
# When the OpenRouter free-tier pool changes, this set is the one
# place to update.
# Free-tier vision allow-list (probed live on 2026-06-15).
#
# As of mid-2026, the only free-tier model on OpenRouter that actually
# accepts image inputs and returns 200 is:
#
#     nvidia/nemotron-nano-12b-v2-vl:free
#
# The other vision models historically listed here (gemini-2.0-flash-exp,
# gemma-3-27b, qwen-2-vl-7b, qwen2.5-vl-32b, mistral-small-3.1-24b,
# llama-3.2-11b-vision, llama-3.2-90b-vision) all return 404 from
# OpenRouter on the free tier. Three of them still work as PAID slugs
# (drop the ``:free`` suffix) but that is out of scope for the
# free-tier demo. Keep the list tight: every id here has been
# live-probed and returned 200 with a real 1x1 PNG payload.
_VISION_MODEL_IDS: frozenset[str] = frozenset({
    "nvidia/nemotron-nano-12b-v2-vl:free",
})


def model_supports_vision(model_id: str) -> bool:
    """Return True if the given model id is known to accept image inputs.

    The comparison is exact and case-insensitive. Whitespace is stripped
    so a model id with a trailing newline (common when read from a CSV
    or an env var) does not silently miss the allow-list.

    Models not in :data:`_VISION_MODEL_IDS` are assumed to be text-only.
    Adding a new vision-capable model is a one-line edit to that set
    — there is no per-provider "capability probe" because OpenRouter
    does not expose a stable capabilities endpoint for free models.
    """
    if not model_id:
        return False
    return model_id.strip().lower() in _VISION_MODEL_IDS


# --- Tunable defaults ---------------------------------------------------------
# Low temperature -> focused, factual answers. Good for security Q&A.
DEFAULT_TEMPERATURE: float = 0.3
# Cap on assistant reply length. Free models are usually generous; 1024 is plenty.
DEFAULT_MAX_TOKENS: int = 5000
# HTTP timeout in seconds. Avoids hanging forever on a stalled connection.
# 30s is plenty for free-tier models: typical reply is 10-15s, and anything
# longer usually means a queued/silently-rejected call that the user would
# rather see as a clean error than wait through.
HTTP_TIMEOUT_SECONDS: int = 60

"""Application configuration.

Loads secrets and settings from the .env file exactly once, at import time,
and exposes them as module-level constants. Every other module in the project
imports from here — no other file should call dotenv, os.getenv, or hardcode
the model name, API key, or endpoint URL.

Provider selection
-----------------

Two providers are supported:

**OpenRouter** (default) — routes to many LLM providers via a single API key.
  Set ``OPENROUTER_API_KEY`` and ``OPENROUTER_MODEL`` / ``OPENROUTER_MODELS``.
  Uses the ``:free`` suffix on model ids. Supports up to 5 API keys for
  rotation.

**Google Gemini** — direct access to Gemini models via Google AI Studio.
  Set ``GEMINI_API_KEY`` (from ai.google.dev). No ``:free`` suffix needed;
  the free tier is based on your project's quota (typically 1500 req/day for
  Gemini 2.0 Flash). All Gemini models natively support vision.

The provider is auto-detected: if ``GEMINI_API_KEY`` is set, Google is used;
otherwise OpenRouter is used. To force OpenRouter when both keys are present,
set ``ACTIVE_PROVIDER=openrouter`` in your .env.

Multi-key + multi-model support (OpenRouter)
-------------------------------------------

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

import logging
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

logger = logging.getLogger(__name__)


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


# --- Provider selection -------------------------------------------------------
# ``ACTIVE_PROVIDER`` can be ``openrouter`` or ``gemini``. When unset (None)
# the module auto-detects: if ``GEMINI_API_KEY`` is set, Gemini is used;
# otherwise OpenRouter is used. Setting this to ``openrouter`` forces OpenRouter
# even when a Gemini key is present.
_ACTIVE_PROVIDER_RAW: str | None = os.getenv("ACTIVE_PROVIDER")
if _ACTIVE_PROVIDER_RAW is not None:
    _ACTIVE_PROVIDER_RAW = _ACTIVE_PROVIDER_RAW.strip().lower()

# --- OpenRouter config -------------------------------------------------------
# Required when using OpenRouter (when Gemini is active, these are ignored).
OPENROUTER_API_KEY: str = _require_env("OPENROUTER_API_KEY")
# Optional default model id. ``iter_models()`` is the authoritative
# source for the model pool: when ``OPENROUTER_MODELS`` is set
# (comma-separated list) it is preferred and ``OPENROUTER_MODEL``
# may be left unset. When only ``OPENROUTER_MODEL`` is set it is
# used as the single-model fallback. When *neither* is set —
# empty string after stripping — the module raises a single,
# clear "no models configured" error so a half-edited .env still
# fails loudly at startup instead of silently sending every turn
# to ``OPENROUTER_MODEL=""``.
OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "").strip()
OPENROUTER_BASE_URL: str = os.getenv(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1/chat/completions",
)

# --- Google Gemini config -----------------------------------------------------
# Optional API key for Google Gemini (from ai.google.dev). When this is set
# AND ``ACTIVE_PROVIDER`` is not ``openrouter``, the app uses Google Gemini
# instead of OpenRouter. Gemini offers generous free-tier quotas (typically
# 1500 req/day for Gemini 2.0 Flash) and all Gemini models support vision
# natively — no separate vision model selection needed.
#
# If ``GEMINI_API_KEY`` is missing, ``iter_api_keys()`` and ``iter_models()``
# yield OpenRouter values. If it IS set, those functions yield Gemini values
# (only one key needed; Google does not have the per-provider free-tier
# rotation problem that OpenRouter has).
_GEMINI_API_KEY_RAW: str | None = os.getenv("GEMINI_API_KEY", "").strip()

# Determine active provider:
# 1. Explicit override via ACTIVE_PROVIDER env var
# 2. Auto-detect: Gemini if GEMINI_API_KEY is set, else OpenRouter
_PROVIDER_EXPLICIT = _ACTIVE_PROVIDER_RAW in {"openrouter", "gemini"}
_PROVIDER_FORCED = _ACTIVE_PROVIDER_RAW if _PROVIDER_EXPLICIT else None

if _PROVIDER_FORCED:
    ACTIVE_PROVIDER: str = _PROVIDER_FORCED  # type: ignore[assignment]
elif _GEMINI_API_KEY_RAW:
    ACTIVE_PROVIDER = "gemini"
else:
    ACTIVE_PROVIDER = "openrouter"


def _gemini_key() -> str:
    """Return the Gemini API key, raising if not configured."""
    if not _GEMINI_API_KEY_RAW:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. "
            "Set it in your .env file (get it from https://aistudio.google.com/app/apikey) "
            "or switch back to OpenRouter by setting ACTIVE_PROVIDER=openrouter."
        )
    return _GEMINI_API_KEY_RAW


# Gemini is always single-key (no multi-key rotation needed), but we keep
# the same interface so the router can call iter_api_keys() generically.
# For Gemini, this always yields exactly one key.
def _iter_gemini_keys() -> Iterator[str]:
    if _GEMINI_API_KEY_RAW:
        yield _GEMINI_API_KEY_RAW


GEMINI_API_KEY: str | None = _GEMINI_API_KEY_RAW or None
GEMINI_BASE_URL: str = os.getenv(
    "GEMINI_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai",
)
# Gemini model to use when no override is set. Gemini 2.0 Flash is the
# recommended default: fast, free-tier friendly (1500 req/day), and
# natively supports vision.
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()

# Comma-separated list of Gemini models, same pattern as OPENROUTER_MODELS.
# Leave blank to use just GEMINI_MODEL.
GEMINI_MODELS_RAW: str = os.getenv("GEMINI_MODELS", "").strip()

# Built-in list of common Gemini models for the UI dropdown.
# These are all verified free-tier models from ai.google.dev.
_GEMINI_DEFAULT_MODELS: tuple[str, ...] = (
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.5-pro",
    "gemini-exp-1206",
)


def _iter_gemini_models() -> Iterator[str]:
    """Yield configured Gemini model ids (comma-separated or single fallback)."""
    if GEMINI_MODELS_RAW:
        for entry in GEMINI_MODELS_RAW.split(","):
            cleaned = entry.strip()
            if cleaned:
                yield cleaned
    elif GEMINI_MODEL:
        yield GEMINI_MODEL


#: Tracks whether the module has already validated that at least
#: one model id is reachable. Populated by :func:`_validate_model_pool`
#: at module-import time so downstream ``import`` statements can rely
#: on either ``OPENROUTER_MODELS`` or ``OPENROUTER_MODEL`` being set
#: without re-checking. The singleton-pattern guards against double
#: validation if the function is ever called more than once (it can
#: be, when tests reload the module).
_MODEL_POOL_VALIDATED: bool = False

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

#: Minimum acceptable length for an OpenRouter API key. As of mid-2026
#: real keys are 67-73 characters long (the ``sk-or-v1-`` prefix plus a
#: ~61-69 char random body). Anything shorter is almost certainly
#: truncated -- which is exactly what happens when an editor hard-wraps
#: a long line at 80 columns and the user saves without noticing. A
#: truncated key looks valid to ``python-dotenv`` (it has the right
#: shape, just the wrong length) but every upstream call returns 401,
#: which disables the slot and leaves the router with zero working
#: keys. Validating here turns that silent failure into a loud skip.
_OPENROUTER_KEY_MIN_LEN: int = 60


def _is_usable_openrouter_key(value: str | None) -> bool:
    """Return True only for keys that *look* complete enough to try.

    The check is intentionally cheap: prefix + length. We do not
    regex-validate the body because OpenRouter occasionally rotates the
    internal encoding and a regex would need to be kept in sync.
    Anything that fails this gate is almost certainly truncation, a
    placeholder, or a typo, and the right behaviour is to skip the
    slot rather than burn an upstream round-trip on a 401.
    """
    if not value:
        return False
    value = value.strip()
    return (
        value.startswith("sk-or-v1-")
        and len(value) >= _OPENROUTER_KEY_MIN_LEN
    )


def iter_api_keys() -> Iterator[str]:
    """Yield API keys for the currently active provider.

    When ``ACTIVE_PROVIDER == "gemini"``:
        Yields exactly one key from ``GEMINI_API_KEY`` (if set).
        Gemini does not require multi-key rotation; the free tier
        is per-project, not per-key.

    When ``ACTIVE_PROVIDER == "openrouter"``:
        Yields up to five ``OPENROUTER_API_KEY[_N]`` values in slot order.
        Slot 1 (``OPENROUTER_API_KEY``) is required. Slots 2..5 are
        optional and silently skipped if malformed / truncated / missing.

    To raise the OpenRouter cap further, bump ``_MAX_KEY_SLOTS`` here.
    """
    if ACTIVE_PROVIDER == "gemini":
        yield from _iter_gemini_keys()
        return

    # OpenRouter path
    yield OPENROUTER_API_KEY
    for n in range(2, _MAX_KEY_SLOTS + 1):
        value = os.getenv(f"OPENROUTER_API_KEY_{n}")
        if not _is_usable_openrouter_key(value):
            if value:
                logger.warning(
                    "OPENROUTER_API_KEY_%d looks truncated or malformed "
                    "(prefix=%r, length=%d); skipping. Re-paste the full "
                    "key from your OpenRouter dashboard to fix.",
                    n,
                    (value.strip()[:10] + "...") if len(value.strip()) > 10 else value.strip(),
                    len(value.strip()),
                )
            continue
        yield value


def _validate_model_pool() -> None:
    """Raise once at module-import time if no model id is reachable.

    Handles both providers:
    - Gemini: validates ``GEMINI_API_KEY`` is set (models have defaults)
    - OpenRouter: validates at least one ``:free`` model id is configured

    Fails loudly at startup so the operator sees a clear error before
    the first chat turn. Idempotent via :data:`_MODEL_POOL_VALIDATED`.
    """
    global _MODEL_POOL_VALIDATED
    if _MODEL_POOL_VALIDATED:
        return

    if ACTIVE_PROVIDER == "gemini":
        if not _GEMINI_API_KEY_RAW:
            raise RuntimeError(
                "Gemini is the active provider but GEMINI_API_KEY is not set. "
                "Get your key from https://aistudio.google.com/app/apikey "
                "and add it to your .env file as GEMINI_API_KEY=..., "
                "or set ACTIVE_PROVIDER=openrouter to use OpenRouter instead."
            )
        # Gemini models have defaults; no further validation needed at startup.
        _MODEL_POOL_VALIDATED = True
        return

    # OpenRouter path
    raw_list = os.getenv("OPENROUTER_MODELS") or ""
    parsed_from_list = [
        entry.strip()
        for entry in raw_list.split(",")
        if entry.strip() and _is_usable_model_id(entry.strip())
    ]
    single = (os.getenv("OPENROUTER_MODEL") or "").strip()
    if not parsed_from_list and not _is_usable_model_id(single):
        raise RuntimeError(
            "No OPENROUTER_MODELS or OPENROUTER_MODEL found in the "
            "environment. Add at least one free-tier id (e.g. "
            "'mistralai/mistral-small-3.2-24b-instruct:free') to your "
            ".env file or the Streamlit Cloud Secrets panel. See "
            ".env.example for the full syntax."
        )
    _MODEL_POOL_VALIDATED = True


def iter_models() -> Iterator[str]:
    """Yield model ids for the currently active provider.

    When ``ACTIVE_PROVIDER == "gemini"``:
        Yields from ``GEMINI_MODELS`` (comma-separated) or falls back
        to the single ``GEMINI_MODEL``. No ``:free`` suffix needed.
        Gemini model names look like ``gemini-2.0-flash``.

    When ``ACTIVE_PROVIDER == "openrouter"``:
        Yields from ``OPENROUTER_MODELS`` (comma-separated) or falls
        back to ``OPENROUTER_MODEL``. Each id must be ``vendor/model:free``.
    """
    if ACTIVE_PROVIDER == "gemini":
        yield from _iter_gemini_models()
        return

    # OpenRouter path
    raw = os.getenv("OPENROUTER_MODELS")
    if raw:
        for entry in raw.split(","):
            cleaned = entry.strip()
            if _is_usable_model_id(cleaned):
                yield cleaned
            elif cleaned:
                logger.warning(
                    "OPENROUTER_MODELS entry %r looks malformed "
                    "(must contain '/' and end with ':free'); skipping.",
                    cleaned,
                )
        return
    single = os.getenv("OPENROUTER_MODEL", "").strip()
    if _is_usable_model_id(single):
        yield single


def _is_usable_model_id(value: str | None) -> bool:
    """Return True for ids that look like ``vendor/model:free``.

    Cheap shape check used by :func:`iter_models`. We do not validate
    against the OpenRouter catalogue here — a typo'd vendor will be
    caught by the upstream 404 on the first call, which the router
    already rotates past.
    """
    if not value:
        return False
    value = value.strip()
    return "/" in value and value.endswith(":free")


class InvalidFreeModelIdError(ValueError):
    """Raised by :func:`validate_free_model_id` when the user-supplied
    OpenRouter id is not a valid ``vendor/model:free`` shape.

    Kept as a distinct class so the view layer can catch it and show
    a friendly sidebar banner (rather than a raw traceback) when a
    user types a paid id, a typo, or whitespace into the Advanced
    model's "Custom OpenRouter model" field.
    """


class InvalidGeminiModelIdError(ValueError):
    """Raised by :func:`validate_gemini_model_id` when the user-supplied
    Gemini model id is not recognisable.

    Kept as a distinct class so the view layer can catch it and show
    a clean banner.
    """


def validate_free_model_id(raw: str) -> str:
    """Validate a user-supplied OpenRouter free-tier model id.

    Only used when ``ACTIVE_PROVIDER == "openrouter"``. The id must
    contain a ``/`` (vendor/model) and end with ``:free``. Rejects
    anything else with :class:`InvalidFreeModelIdError`.

    Parameters
    ----------
    raw
        The raw text the user typed.

    Returns
    -------
    str
        The cleaned id (whitespace stripped).

    Raises
    ------
    InvalidFreeModelIdError
        If the cleaned id does not contain a ``/`` or does not end
        with ``:free``.
    """
    if raw is None:
        raise InvalidFreeModelIdError("Model id is empty.")
    cleaned = raw.strip()
    if not cleaned:
        raise InvalidFreeModelIdError("Model id is empty.")
    if "/" not in cleaned:
        raise InvalidFreeModelIdError(
            f"Model id {cleaned!r} is missing the 'vendor/' prefix. "
            "OpenRouter ids always look like 'vendor/model:free'."
        )
    if not cleaned.endswith(":free"):
        raise InvalidFreeModelIdError(
            f"Model id {cleaned!r} is not a free-tier id (must end with "
            "':free'). Paid models are intentionally not supported — "
            "rotation across free keys is the whole point."
        )
    return cleaned


def validate_gemini_model_id(raw: str) -> str:
    """Validate a user-supplied Gemini model id.

    Only used when ``ACTIVE_PROVIDER == "gemini"``. Accepts any model
    name that starts with ``gemini-`` (the standard prefix for all
    Gemini models on Google AI Studio). Rejects anything else with
    :class:`InvalidGeminiModelIdError`.

    Parameters
    ----------
    raw
        The raw text the user typed.

    Returns
    -------
    str
        The cleaned id (whitespace stripped).

    Raises
    ------
    InvalidGeminiModelIdError
        If the cleaned id does not look like a Gemini model.
    """
    if raw is None:
        raise InvalidGeminiModelIdError("Gemini model id is empty.")
    cleaned = raw.strip()
    if not cleaned:
        raise InvalidGeminiModelIdError("Gemini model id is empty.")
    if not cleaned.startswith("gemini-"):
        raise InvalidGeminiModelIdError(
            f"Gemini model id {cleaned!r} does not start with 'gemini-'. "
            "All Gemini models follow the 'gemini-X.Y-name' pattern. "
            "Example: 'gemini-2.0-flash'."
        )
    return cleaned


# --- Vision-capable model allow-list ------------------------------------------
# Two cases:
#
# 1. Google Gemini (ACTIVE_PROVIDER == "gemini"): ALL Gemini models natively
#    support vision — no swap needed, no allow-list. The helper returns True
#    for any gemini-* model.
#
# 2. OpenRouter: the list is intentionally conservative — only models we have
#    confirmed accept the OpenRouter ``image_url`` part shape on the free tier.
#    Adding a model here that does NOT support vision produces 400 errors.
#
# The view layer reads this via ``model_supports_vision()`` to decide whether
# to auto-swap the user's selected model when an image is attached.
# Keeping the set in ``app/`` (not in the view) means tests can patch it.
#
# OpenRouter free-tier vision allow-list (probed live on 2026-06-15).
# As of mid-2026, only nemotron-nano-12b-v2-vl works on OpenRouter's :free tier.
_OPENROUTER_VISION_MODEL_IDS: frozenset[str] = frozenset({
    "nvidia/nemotron-nano-12b-v2-vl:free",
})


def model_supports_vision(model_id: str) -> bool:
    """Return True if the given model id is known to accept image inputs.

    For Google Gemini: all models support vision natively — return True
    for any gemini-* model id.

    For OpenRouter: exact match against the free-tier vision allow-list.
    Whitespace is stripped so a trailing newline doesn't silently miss.
    """
    if not model_id:
        return False
    cleaned = model_id.strip().lower()
    if ACTIVE_PROVIDER == "gemini":
        return cleaned.startswith("gemini-")
    return cleaned in _OPENROUTER_VISION_MODEL_IDS


# --- Module-load validation --------------------------------------------------
# Run the model-pool check *after* every helper it depends on is
# defined. Idempotent, so re-imports (tests that reload the module,
# Streamlit's auto-reloader on .env change) skip the work. If this
# raises the message is shown verbatim in the Streamlit traceback
# panel — operators see "No OPENROUTER_MODELS or OPENROUTER_MODEL
# found" and know exactly which panel to edit.
_validate_model_pool()


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


# --- Recon / OSINT tunables (Phase 15) ---------------------------------------
# These are the *only* knobs the recon subsystem reads from the environment.
# Every other recon constant is derived (e.g. the hard-coded RFC1918 ranges
# in :mod:`app.recon.safety`). Keep new knobs here so the operator has one
# place to look in ``.env`` and so the test suite can patch them via
# ``monkeypatch.setenv`` without touching the recon modules directly.

#: Optional ipinfo.io token. Without a token the API returns a limited
#: "lite" payload (no org/company, no abuse contact, rate-limited to
#: ~1k/day per IP). With a token you get the full record. Stored as a
#: plain string — never logged. The default of ``""`` is intentional:
#: it means "no token, run in lite mode" rather than raising at import.
IPINFO_TOKEN: str = os.getenv("IPINFO_TOKEN", "")

#: HTTP timeout, in seconds, for every outbound recon call (ipinfo.io,
#: urlinfo.io, crt.sh, the WHOIS TCP probe). 15s is generous for the
#: typical 1-3s response and gives crt.sh — which can be slow on a cold
#: query — enough headroom. The orchestrator passes this to every
#: transport so a stalled crt.sh query cannot hang the whole report.
RECON_HTTP_TIMEOUT_SECONDS: float = float(
    os.getenv("RECON_HTTP_TIMEOUT_SECONDS", "15")
)

#: Per-tool override for the crt.sh HTTP timeout. crt.sh is a hobby
#: project with cold-start latencies well above the shared 15s default
#: (sometimes 30s+ on first hit) and we don't want a slow crt.sh query
#: to spill into the rest of the report. Defaults to ``20`` (a tight
#: cap; crt.sh typically answers in 2-5s once warm, and the orchestrator
#: already runs crt.sh in parallel with the other four tools so a
#: timeout here only delays the crt.sh row, not the rest of the
#: report). Operators with frequent crt.sh timeouts can set
#: ``RECON_CRT_SH_TIMEOUT_SECONDS=45`` (or similar) in their
#: ``.env`` to give crt.sh more headroom without slowing the other
#: tools down.
RECON_CRT_SH_TIMEOUT_SECONDS: float = float(
    os.getenv("RECON_CRT_SH_TIMEOUT_SECONDS", "20")
)


def _parse_bool_env(name: str, default: bool) -> bool:
    """Parse a boolean env var with a forgiving truthy/falsy vocabulary.

    Accepts ``"1" / "0"``, ``"true" / "false"``,
    ``"yes" / "no"``, ``"on" / "off"`` (case-insensitive,
    whitespace-stripped). Anything not on the truthy list — including
    the empty string — returns :data:`default`. This mirrors the
    convention used by ``configparser`` / ``distutils.util.strtobool``
    without pulling in a deprecated stdlib symbol.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    token = raw.strip().lower()
    if token in {"1", "true", "yes", "on"}:
        return True
    if token in {"0", "false", "no", "off"}:
        return False
    return default


#: Kill-switch for the crt.sh lookup. crt.sh is a hobby project that
#: occasionally returns 502 / 503 for long stretches (whole days
#: during the 2025 winter outage, for example). When the upstream is
#: unhealthy the orchestrator short-circuits the crt.sh tool and
#: returns a synthetic, ``ok=True`` :class:`CrtShResult` instead —
#: the recon report still renders cleanly with no scary **Error**
#: row, and the rest of the pipeline (DNS / IP info / URL info /
#: WHOIS) keeps running. Default ``True`` so deployments that do
#: not set the knob keep the historical behaviour. Operators flip it
#: to ``"false"`` in ``.env`` when crt.sh is the bottleneck and
#: want the recon turn to stop paying its latency tax.
#:
#: Accepts the forgiving truthy / falsy vocabulary in
#: :func:`_parse_bool_env` (``1 / 0``, ``true / false``,
#: ``yes / no``, ``on / off``).
RECON_CRT_SH_ENABLED: bool = _parse_bool_env("RECON_CRT_SH_ENABLED", True)


def _parse_fallback_hosts() -> tuple[str, ...]:
    """Parse :data:`RECON_FALLBACK_SUBDOMAINS` into a deduplicated tuple.

    The env var is a comma-separated list of hostnames (whitespace
    tolerated around the commas). Empty entries, entries without a
    dot, and a leading ``*.`` wildcard prefix are stripped so a
    typo or a copy-paste from a crt.sh-style row can't poison the
    report with garbage like ``"foo, , *.bar, baz"`` →
    ``("foo", "bar", "baz")``. The result is sorted alphabetically
    so it matches the ordering used by :func:`crt_sh.lookup` —
    operators reading the report see a stable order regardless of
    how they wrote the env var.
    """
    raw = os.getenv("RECON_FALLBACK_SUBDOMAINS", "") or ""
    seen: set[str] = set()
    for entry in raw.split(","):
        h = entry.strip().lower().lstrip("*.").strip()
        if not h or "." not in h:
            continue
        seen.add(h)
    return tuple(sorted(seen))


#: Optional manual subdomain list used as a substitute when crt.sh
#: is disabled (see :data:`RECON_CRT_SH_ENABLED`). The string is
#: parsed at import time via :func:`_parse_fallback_hosts`. The
#: orchestrator merges these hosts into the crt.sh report so the
#: "Hosts seen" section still has content even when the live
#: crt.sh query is skipped. The tuple is frozen so a downstream
#: consumer cannot mutate it by accident.
#:
#: Typical sources for the fallback list:
#: - ``securitytrails.com`` free preview,
#: - a ``subfinder`` / ``amass`` JSON dump the operator already ran,
#: - a manually-curated list from an internal asset inventory.
RECON_FALLBACK_SUBDOMAINS: tuple[str, ...] = _parse_fallback_hosts()

#: How long recon audit rows live before the cleanup job (Phase 2)
#: expires them. 90 days is enough for a real engagement audit trail
#: while bounding the table size for a single-user demo. The value is
#: read at write time so a config change takes effect on the next row.
RECON_AUDIT_RETENTION_DAYS: int = int(
    os.getenv("RECON_AUDIT_RETENTION_DAYS", "90")
)

#: Whitelisted ``scope_token`` values. The orchestrator requires a scope
#: token on every call (the chat slash command supplies it), and the
#: token must be in this set. The set is small and opinionated on
#: purpose: only the engagements the operator has actually authorised
#: can run recon. Adding a new value is a one-line edit and is the
#: *operator's* responsibility — not the user's.
#:
#: ``engagement``  : paid / written-authorisation work
#: ``ctf``         : capture-the-flag competition infrastructure
#: ``lab``         : intentionally vulnerable practice boxes (the
#:                   short form used by the slash-command parser and
#:                   the test suite; ``labs`` is kept as an alias)
#: ``labs``        : alias of ``lab``
#: ``redteam``     : internal red-team operations
#: ``personal-lab``: home network and devices the operator owns
#: ``bugbounty``   : a public bug-bounty programme listed in scope
_RECON_SCOPE_TOKENS: frozenset[str] = frozenset({
    "engagement",
    "ctf",
    "lab",
    "labs",
    "redteam",
    "personal-lab",
    "bugbounty",
})

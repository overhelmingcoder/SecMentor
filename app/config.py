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
    """Yield every non-empty ``OPENROUTER_API_KEY[_N]`` in slot order.

    Slot 1 (``OPENROUTER_API_KEY``) is required and is the only one
    that triggers a startup error if missing. Slots 2..5 are optional
    and are silently skipped if not set, set to the empty string, or
    fail :func:`_is_usable_openrouter_key` (truncated, missing the
    ``sk-or-v1-`` prefix, or suspiciously short).

    Order is stable: slot 1 first, then 2, 3, 4, 5. The router uses
    this order to pick the *primary* key (slot 1) for the first call.

    To raise the cap further, bump ``_MAX_KEY_SLOTS`` here; the rest of
    the codebase reads the constant through this iterator.
    """
    yield OPENROUTER_API_KEY
    for n in range(2, _MAX_KEY_SLOTS + 1):
        value = os.getenv(f"OPENROUTER_API_KEY_{n}")
        if not _is_usable_openrouter_key(value):
            # A non-empty but malformed value is the "editor hard-wrapped
            # my .env" case. We log a single warning at module-import
            # time so the operator notices in the console without
            # flooding every chat turn. The test suite patches
            # ``logging.getLogger(__name__)`` so this stays silent in
            # CI.
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

    Each yielded id is validated to look like ``vendor/model:free``:
    the router will refuse it again at construction time, but checking
    here turns a "value silently truncated to ``gemma-4-31b-it:fre``"
    into a single clear warning at startup.
    """
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
    if _is_usable_model_id(OPENROUTER_MODEL):
        yield OPENROUTER_MODEL
    else:
        # The single-model path. The router would catch this at
        # construction time, but the message here is friendlier and
        # points operators at the actual fix.
        raise RuntimeError(
            f"OPENROUTER_MODEL={OPENROUTER_MODEL!r} is not a valid "
            "free-tier model id. It must contain a '/' (vendor/model) "
            "and end with ':free'. Update your .env."
        )


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
    id is not a valid ``vendor/model:free`` shape.

    Kept as a distinct class so the view layer can catch it and show
    a friendly sidebar banner (rather than a raw traceback) when a
    user types a paid id, a typo, or whitespace into the Advanced
    model's "Custom OpenRouter model" field.
    """


def validate_free_model_id(raw: str) -> str:
    """Validate a user-supplied free-tier model id.

    Used by the "Custom OpenRouter model" field in the Advanced
    expander of ``web/streamlit_app.py``. The rule is the same one
    :func:`iter_models` applies at startup: the id must contain a
    ``/`` (vendor/model) and end with ``:free``. Anything else is
    rejected with :class:`InvalidFreeModelIdError` so the caller can
    show a clean error instead of silently burning a paid slot.

    Parameters
    ----------
    raw
        The raw text the user typed. Leading and trailing whitespace
        is stripped; empty strings are rejected (the caller can use
        the empty string as the "no override" signal without going
        through this function).

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

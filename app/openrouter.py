"""Thin HTTP client for the OpenRouter chat-completions API.

The module exposes one public function (``chat``) and a small exception
hierarchy. It deliberately contains *no* retry, rotation, or rate-limit
state — that lives in ``app/router.py`` so a future swap to a different
provider (or to a local LLM) is a one-line wiring change. Keep this
file boring: parse, send, raise on non-2xx, return the assistant text.

Exception hierarchy
-------------------

The router (``app/router.py``) needs to *classify* failures, not just
catch them. So we expose a small hierarchy rooted at ``OpenRouterError``:

- ``OpenRouterAuthError``      — HTTP 401 / 403 (key is bad; do not retry)
- ``OpenRouterRateLimitError`` — HTTP 429 (transient; safe to retry/backoff)
- ``OpenRouterClientError``    — any other 4xx (bad request; do not retry)
- ``OpenRouterServerError``    — HTTP 5xx (upstream blip; safe to retry)

All four inherit from ``OpenRouterError``, so existing
``except OpenRouterError:`` blocks in the view and the CLI keep working
unchanged. New code that wants to rotate / skip / backoff can use
``isinstance(err, OpenRouterRateLimitError)`` etc.

Every exception carries:

- ``status``   — the HTTP status code (or ``None`` for non-HTTP failures
                 like a network error or an empty message list)
- ``provider`` — the OpenRouter "provider" header from the response, when
                 the upstream told us (useful for diagnosing which
                 model host is throttling). ``None`` otherwise.
- ``model``    — the model id we asked for. Stored on the exception so
                 logs and the router can see *which* slot in the pool
                 failed without re-passing the model id separately.
- ``body``     — the raw response body (parsed JSON when possible, raw
                 text otherwise). Truncated to a sane length so we do
                 not blow up the log on a 1 MB error payload.

Network-level failures (``requests.ConnectionError``,
``requests.Timeout``, etc.) are wrapped into ``OpenRouterServerError``
with ``status=None`` — the router treats them as transient.
"""

from __future__ import annotations

import json
from typing import Any

import requests

from app.config import (
    HTTP_TIMEOUT_SECONDS,
    OPENROUTER_APP_NAME,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
)

# Cap the length of the raw upstream body we keep on the exception. The
# router's logs already include the model id and the error class, so
# this is only here for the developer-facing "show raw error" expander
# in the Streamlit view. 2 KB is plenty for any real OpenRouter error
# envelope and prevents a runaway error payload from bloating memory.
_MAX_BODY_CHARS: int = 2000


# --- Exception hierarchy -----------------------------------------------------


class OpenRouterError(RuntimeError):
    """Base class for every failure this module can raise.

    Subclasses exist so callers (``app/router.py``) can branch on the
    failure *kind* (``isinstance(err, OpenRouterRateLimitError)``)
    instead of pattern-matching the message string. The base class is
    the one to catch when you do not care about the kind.

    All keyword arguments are optional and default to ``None`` so the
    non-HTTP failure modes (empty message list, empty assistant text,
    network error) can still raise a fully-formed exception without
    pretending a status code exists.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        provider: str | None = None,
        model: str | None = None,
        body: str | None = None,
    ) -> None:
        super().__init__(message)
        # HTTP status when the failure is an HTTP response. None for
        # empty-input, empty-output, and network-level failures.
        self.status: int | None = status
        # OpenRouter's "provider" header value from the response, when
        # present. Helps the router log *which upstream host* rejected
        # a call (useful when the same model id is served by multiple
        # providers and one of them is the slow one).
        self.provider: str | None = provider
        # The model id we asked for. Stored on the exception so the
        # router and the UI can attribute the failure to the exact
        # slot in the pool without re-passing the model id.
        self.model: str | None = model
        # Raw upstream body, truncated. None when the failure happens
        # before we get a response.
        self.body: str | None = body


class OpenRouterAuthError(OpenRouterError):
    """HTTP 401/403 — the API key is invalid, expired, or unauthorised.

    The router treats this as a *permanent* failure for the key in
    question: there is no point retrying, the key will never start
    working. The router marks the slot disabled for the rest of the
    session so subsequent calls skip it without a network round-trip.
    """


class OpenRouterRateLimitError(OpenRouterError):
    """HTTP 429 — upstream asked us to slow down.

    The router backs off briefly and then either retries the same slot
    or rotates to the next one, depending on policy. Carries the same
    fields as the base class; nothing extra for now.
    """


class OpenRouterServerError(OpenRouterError):
    """HTTP 5xx or a network-level failure (status is None in that case).

    Treated as transient: the router may retry the same slot once and
    then rotate.
    """


class OpenRouterClientError(OpenRouterError):
    """Any 4xx that is not 401/403/429 (e.g. 400 bad request, 404 model
    not found, 413 context too long).

    Treated as *non-retryable*: the request is malformed or the model
    is wrong, so the router skips to the next slot rather than burning
    the same call again.
    """


# --- Helpers -----------------------------------------------------------------


def _build_headers(api_key: str) -> dict[str, str]:
    """Build the HTTP headers for an OpenRouter chat-completions call.

    ``api_key`` is a parameter (not a module-level lookup) so the
    router can pass a key from the pool. The default callers (the
    Streamlit view, the CLI) pass ``OPENROUTER_API_KEY`` from
    ``app.config``.
    """
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/local/stage1",
        "X-Title": OPENROUTER_APP_NAME,
    }


def _build_payload(
    messages: list[dict[str, Any]],
    *,
    model: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    """Build the JSON payload for the chat-completions call.

    ``messages`` is typed as ``list[dict[str, Any]]`` (not
    ``list[dict[str, str]]``) so vision turns can pass a content *list*
    of parts like
    ``[{"type": "text", "text": "..."}, {"type": "image_url", "image_url": ...}]``
    in the user message without forcing the helper layer to stringify
    it. The HTTP layer never inspects the inner shape — it serialises
    the dict verbatim — so widening the annotation has no runtime
    effect, it is documentation + type-checker support only.
    """
    return {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def _truncate(text: str | None) -> str | None:
    """Cap ``text`` at ``_MAX_BODY_CHARS`` so a runaway error does not
    bloat the exception's memory footprint. Returns ``None`` for None."""
    if text is None:
        return None
    if len(text) <= _MAX_BODY_CHARS:
        return text
    return text[:_MAX_BODY_CHARS] + f"... [truncated, {len(text) - _MAX_BODY_CHARS} more chars]"


def _classify(status_code: int) -> type[OpenRouterError]:
    """Map an HTTP status code to the most specific exception class.

    Returns the *class* (not an instance) so the call site can raise
    with a single uniform signature:

        raise _classify(response.status_code)(msg, status=..., ...)

    The class hierarchy is rooted at ``OpenRouterError`` so an
    ``except OpenRouterError`` block at the call site still catches
    every subclass.
    """
    if status_code in (401, 403):
        return OpenRouterAuthError
    if status_code == 429:
        return OpenRouterRateLimitError
    if 400 <= status_code < 500:
        return OpenRouterClientError
    if 500 <= status_code < 600:
        return OpenRouterServerError
    # Defensive default: anything else gets the base class so the
    # caller still gets a structured exception. Should not be
    # reachable in practice (4xx and 5xx cover all HTTP errors).
    return OpenRouterError


def _extract_assistant_text(payload: dict[str, Any]) -> str:
    """Pull the assistant's text out of an OpenRouter response payload.

    OpenRouter mirrors the OpenAI chat-completions schema, so the text
    lives at ``payload["choices"][0]["message"]["content"]``. We guard
    every index access with explicit ``KeyError`` / ``IndexError``
    conversions so the caller sees a clear error instead of a stack
    trace with no context.
    """
    try:
        choice = payload["choices"][0]
        message = choice["message"]
        content = message.get("content") or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise OpenRouterError(
            "OpenRouter response is missing the expected "
            f"choices[0].message.content field: {exc}. "
            f"Raw payload (truncated): {_truncate(str(payload))}",
        ) from exc
    if not content.strip():
        # Empty / whitespace-only content is a valid signal that the
        # model refused or that the upstream is misbehaving. We raise
        # with a clear message instead of returning "" and letting
        # the UI silently render an empty bubble.
        raise OpenRouterError(
            "OpenRouter returned an empty assistant message. "
            f"Raw payload (truncated): {_truncate(str(payload))}",
        )
    return content


# --- Public API --------------------------------------------------------------


def chat(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> str:
    """Send ``messages`` to OpenRouter and return the assistant's text.

    Parameters
    ----------
    messages
        The chat history to send. Must be non-empty and the OpenAI
        schema: a list of ``{"role": ..., "content": ...}`` dicts.
        ``content`` may be a plain string (text-only turns) *or* a
        list of parts (vision turns, where the user message can mix
        ``{"type": "text", "text": ...}`` and
        ``{"type": "image_url", "image_url": {"url": ...}}`` parts).
        The HTTP layer serialises ``content`` verbatim, so the wider
        type is a documentation-level widening only.
    model
        Model id to use. Defaults to ``OPENROUTER_MODEL`` from
        ``app.config``. The router passes an explicit id per slot.
    temperature
        Sampling temperature. Defaults to a low, focused value
        (configured in ``app.config``).
    max_tokens
        Cap on the assistant reply length. Defaults to
        ``DEFAULT_MAX_TOKENS`` from ``app.config``.
    api_key
        Bearer token. Defaults to ``OPENROUTER_API_KEY`` from
        ``app.config``. The router passes a per-slot key from the
        pool. Passing it explicitly here is what lets the same
        function work for both single-key and multi-key deployments.
    base_url
        Override the endpoint. Defaults to ``OPENROUTER_BASE_URL`` from
        ``app.config``. Tests patch this to point at a closed port so
        they can exercise the network-error path without spending an
        API credit.

    Returns
    -------
    str
        The assistant's reply text.

    Raises
    ------
    OpenRouterAuthError
        HTTP 401 / 403.
    OpenRouterRateLimitError
        HTTP 429.
    OpenRouterClientError
        Any other 4xx.
    OpenRouterServerError
        HTTP 5xx, or a network-level failure (``status`` is ``None``
        in that case).
    OpenRouterError
        For non-HTTP failures (empty message list, empty assistant
        text, malformed response).
    """
    if not messages:
        raise OpenRouterError("Cannot call OpenRouter with an empty message list.")

    chosen_model = model or OPENROUTER_MODEL
    chosen_key = api_key if api_key is not None else OPENROUTER_API_KEY
    chosen_url = base_url if base_url is not None else OPENROUTER_BASE_URL
    # Fall back to the config defaults for the two optional numeric
    # parameters. Importing them lazily here (rather than at module
    # top) avoids a circular import with ``app.config`` if either of
    # those constants is ever moved.
    from app.config import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE

    chosen_temperature = DEFAULT_TEMPERATURE if temperature is None else temperature
    chosen_max_tokens = DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens

    payload = _build_payload(
        messages,
        model=chosen_model,
        temperature=chosen_temperature,
        max_tokens=chosen_max_tokens,
    )
    headers = _build_headers(chosen_key)

    try:
        response = requests.post(
            chosen_url,
            headers=headers,
            json=payload,
            timeout=HTTP_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        # Network-level failure: connection refused, DNS error, TLS
        # error, timeout. Wrap as a server error (transient) with
        # ``status=None`` so the router can distinguish it from a
        # definitive HTTP 4xx.
        raise OpenRouterServerError(
            f"OpenRouter request failed before getting a response: {exc}",
            status=None,
            model=chosen_model,
        ) from exc

    if response.status_code >= 400:
        # Pull the upstream body in the best representation we can get.
        # JSON when possible (so the router / log can introspect it),
        # raw text otherwise.
        try:
            detail: Any = response.json()
            body_text: str | None = json.dumps(detail)[:_MAX_BODY_CHARS]
        except ValueError:
            detail = response.text
            body_text = _truncate(response.text)
        provider_header = response.headers.get("x-provider") or None
        # The "X-Provider" header is what OpenRouter sets to identify
        # which model host served the request. Some free providers do
        # not send it; that is fine, we just leave ``provider`` as
        # ``None`` for those calls.
        message = (
            f"OpenRouter returned HTTP {response.status_code}: {detail}"
        )
        exc_class = _classify(response.status_code)
        raise exc_class(
            message,
            status=response.status_code,
            provider=provider_header,
            model=chosen_model,
            body=body_text,
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise OpenRouterError(
            "OpenRouter returned a non-JSON 2xx response. "
            f"Body (truncated): {_truncate(response.text)}",
            status=response.status_code,
            model=chosen_model,
        ) from exc

    return _extract_assistant_text(data)

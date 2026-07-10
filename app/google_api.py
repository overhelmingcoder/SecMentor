"""Thin HTTP client for the Google Gemini API via the OpenAI-compatible endpoint.

This module exposes the same ``chat`` and ``stream_chat`` interface as
``app.openrouter`` so the router and the view can swap providers by
changing ``app/config.py`` without touching any call site.

Why a separate module (not a config flag inside openrouter.py)
-----------------------------------------------------------

The two providers have subtly different:
- Error envelope shapes (Google's ``{"error": {"message": ..., "status": ...}}``
  vs OpenRouter's ``{"error": {"message": ..., "code": ...}}`` — both need to
  be classified into the same exception hierarchy).
- Key injection: Google requires ``?key=API_KEY`` on the URL; OpenRouter
  uses ``Authorization: Bearer`` in headers.
- Model naming conventions: Google uses ``gemini-2.0-flash`` (no ``/vendor``,
  no ``:free`` suffix).

Keeping the implementations separate means each can be read in isolation,
tested in isolation, and updated when its provider changes — without
touching the shared exception hierarchy or the router.

Exception hierarchy
------------------

Mirrors ``app.openrouter`` so callers can use the same
``isinstance(err, OpenRouterRateLimitError)`` branching:

- ``OpenRouterAuthError``    — HTTP 401 / 403 (key invalid / no quota)
- ``OpenRouterRateLimitError`` — HTTP 429 (transient; safe to retry/backoff)
- ``OpenRouterClientError``   — HTTP 4xx other than 429 (bad request; don't retry)
- ``OpenRouterServerError``   — HTTP 5xx (transient; safe to retry)

Network-level failures (``requests.ConnectionError``,
``requests.Timeout``) are wrapped as ``OpenRouterServerError`` with
``status=None``.

The upstream provider is "Google Generative AI" — that label surfaces
in the router's logs and the UI's raw-error expander when present in
the response headers.
"""

from __future__ import annotations

import json
from itertools import chain
from typing import Any, Iterator

import requests

from app.config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    GEMINI_API_KEY,
    GEMINI_BASE_URL,
    GEMINI_MODEL,
    HTTP_TIMEOUT_SECONDS,
    iter_models,
)

# Cap the raw upstream body length on exceptions (same rationale as openrouter.py).
_MAX_BODY_CHARS: int = 2000

# The provider label surfaced in error diagnostics.
_PROVIDER_LABEL = "Google Generative AI"


# --- Exception hierarchy -----------------------------------------------------
# Identical to app.openrouter so the router can catch these without
# knowing which provider was used.


class OpenRouterError(RuntimeError):
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
        self.status: int | None = status
        self.provider: str | None = provider
        self.model: str | None = model
        self.body: str | None = body


class OpenRouterAuthError(OpenRouterError):
    """HTTP 401/403 — the API key is invalid, expired, or unauthorised."""


class OpenRouterRateLimitError(OpenRouterError):
    """HTTP 429 — upstream asked us to slow down."""


class OpenRouterServerError(OpenRouterError):
    """HTTP 5xx or a network-level failure (status is None in that case)."""


class OpenRouterClientError(OpenRouterError):
    """HTTP 4xx other than 401/403/429 (e.g. 400, 404, 413)."""


# --- Helpers -----------------------------------------------------------------


def _build_headers(api_key: str) -> dict[str, str]:
    """Build HTTP headers for the Google OpenAI-compatible endpoint.

    The OpenAI-compatible endpoint requires the API key as a Bearer token.
    """
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }


def _build_url(model: str, api_key: str) -> str:
    """Build the full URL for a generateContent call.

    The OpenAI-compatible endpoint is:
      https://generativelanguage.googleapis.com/v1beta/openai/chat/completions

    The model name and API key are both query parameters.
    """
    base = GEMINI_BASE_URL.rstrip("/")
    return f"{base}/models/{model}:generateContent?key={api_key}"


def _build_payload(
    messages: list[dict[str, Any]],
    *,
    model: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    """Build the JSON payload for a generateContent call.

    Google Gemini's OpenAI-compatible endpoint accepts the same ``model``,
    ``messages``, ``temperature``, and ``max_tokens`` fields. However, for
    vision support we need to convert the content list parts to Google's
    native ``{"text": ...}`` / ``{"inlineData": {"mimeType": ..., "data": ...}}``
    format because the ``{"type": "image_url", "image_url": ...}`` shape that
    works for OpenRouter may not be accepted by this endpoint.

    For pure text messages (content is a string) the format is identical
    to OpenAI and needs no conversion. For multimodal messages the caller
    is responsible for building the content in the format accepted by
    the specific model endpoint being used.

    NOTE: when using the ``/v1beta/openai/chat/completions`` endpoint,
    Google also accepts the raw content arrays from OpenAI directly, so
    this helper just passes the messages through unchanged.
    """
    return {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def _truncate(text: str | None) -> str | None:
    if text is None:
        return None
    if len(text) <= _MAX_BODY_CHARS:
        return text
    return text[:_MAX_BODY_CHARS] + f"... [truncated, {len(text) - _MAX_BODY_CHARS} more chars]"


def _classify(status_code: int) -> type[OpenRouterError]:
    """Map an HTTP status code to the most specific exception class.

    Identical to ``app.openrouter._classify`` — kept here so this module
    is self-contained.
    """
    if status_code in (401, 403):
        return OpenRouterAuthError
    if status_code == 429:
        return OpenRouterRateLimitError
    if 400 <= status_code < 500:
        return OpenRouterClientError
    if 500 <= status_code < 600:
        return OpenRouterServerError
    return OpenRouterError


def _extract_assistant_text(payload: dict[str, Any]) -> str:
    """Pull the assistant text from a Google generateContent response.

    Google Gemini returns structured candidates. The OpenAI-compatible
    endpoint wraps candidates in ``{"candidates": [...]}`` and the text
    lives in ``candidates[0].content.parts[0].text``.

    We also accept the raw OpenAI shape (``choices[0].message.content``)
    for API compatibility — the OpenAI-compatible endpoint returns that
    shape too. We try the Google shape first, then fall back to OpenAI.
    """
    # Try Google's native shape: candidates[0].content.parts[0].text
    try:
        candidates = payload.get("candidates") or []
        if candidates:
            first = candidates[0]
            content = first.get("content") or {}
            parts = content.get("parts") or []
            if parts:
                text = parts[0].get("text") or ""
                if text.strip():
                    return text
    except (KeyError, IndexError, TypeError):
        pass

    # Fall back to OpenAI-compatible shape
    try:
        choice = payload["choices"][0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        if isinstance(content, str) and content.strip():
            return content
    except (KeyError, IndexError, TypeError):
        pass

    raise OpenRouterError(
        "Google API response is missing the expected text field. "
        f"Raw payload (truncated): {_truncate(str(payload))}",
    )


def _extract_stream_delta(line: str) -> str | None:
    """Extract text delta from a Google streaming SSE line.

    Google returns SSE events in two shapes:
    - ``data: {...}`` — dict with candidates[0].content.parts[0].text
    - ``data: [DONE]`` — sentinel

    We handle both, and also the OpenAI-compatible shape
    ``choices[0].delta.content`` for API compatibility.
    """
    stripped = line.strip()
    if not stripped.startswith("data:"):
        return None
    data_str = stripped[len("data:"):].strip()
    if data_str == "[DONE]":
        return None

    try:
        data = json.loads(data_str)
    except ValueError:
        return None

    # Try Google's native streaming shape
    try:
        candidates = data.get("candidates") or []
        if candidates:
            parts = (candidates[0].get("content") or {}).get("parts") or []
            if parts:
                text = parts[0].get("text")
                if text:
                    return text
    except (KeyError, IndexError, TypeError):
        pass

    # Fall back to OpenAI-compatible delta shape
    try:
        delta = (data.get("choices") or [{}])[0].get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str) and content:
            return content
    except (KeyError, IndexError, TypeError):
        pass

    return None


# --- Public API --------------------------------------------------------------


def chat(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> str:
    """Send ``messages`` to Google Gemini and return the assistant's text.

    Signature mirrors ``app.openrouter.chat`` so the router can call either
    module interchangeably.

    Parameters
    ----------
    messages
        Chat history in OpenAI schema: ``[{"role": ..., "content": ...}]``.
        For multimodal turns, content may be a list of parts in the format
        accepted by the OpenAI-compatible endpoint.
    model
        Gemini model name (e.g. ``gemini-2.0-flash``). Defaults to
        ``GEMINI_MODEL`` from ``app.config``.
    temperature, max_tokens, timeout
        Same as ``app.openrouter.chat``.
    api_key
        Defaults to ``GEMINI_API_KEY`` from ``app.config``.
    base_url
        Override endpoint base URL. Defaults to ``GEMINI_BASE_URL``.

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
        HTTP 4xx other than 429.
    OpenRouterServerError
        HTTP 5xx or network failure.
    OpenRouterError
        Non-HTTP failures (empty message list, empty assistant text).
    """
    if not messages:
        raise OpenRouterError("Cannot call Gemini with an empty message list.")

    chosen_model = model or GEMINI_MODEL or next(iter_models(), "")
    chosen_key = api_key if api_key is not None else GEMINI_API_KEY
    chosen_url_base = base_url if base_url is not None else GEMINI_BASE_URL
    from app.config import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE

    chosen_temperature = DEFAULT_TEMPERATURE if temperature is None else temperature
    chosen_max_tokens = DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens
    chosen_timeout = HTTP_TIMEOUT_SECONDS if timeout is None else timeout

    # Build URL for the OpenAI-compatible endpoint.
    # Base is https://generativelanguage.googleapis.com/v1beta/openai
    base = chosen_url_base.rstrip("/")
    url = f"{base}/chat/completions"

    payload = _build_payload(
        messages,
        model=chosen_model,
        temperature=chosen_temperature,
        max_tokens=chosen_max_tokens,
    )
    headers = _build_headers(chosen_key)

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=chosen_timeout,
        )
    except requests.RequestException as exc:
        raise OpenRouterServerError(
            f"Google API request failed before getting a response: {exc}",
            status=None,
            model=chosen_model,
            provider=_PROVIDER_LABEL,
        ) from exc

    # Same UTF-8 pin as openrouter.py — prevents mojibake on multi-byte chars.
    response.encoding = "utf-8"

    if response.status_code >= 400:
        try:
            detail = response.json()
            body_text = json.dumps(detail)[:_MAX_BODY_CHARS]
        except ValueError:
            detail = response.text
            body_text = _truncate(response.text)
        provider_header = response.headers.get("x-goog-api-runtime") or _PROVIDER_LABEL
        message = f"Google API returned HTTP {response.status_code}: {detail}"
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
            "Google API returned a non-JSON 2xx response. "
            f"Body (truncated): {_truncate(response.text)}",
            status=response.status_code,
            model=chosen_model,
            provider=_PROVIDER_LABEL,
        ) from exc

    # Embedded error in 2xx body (Google sometimes returns this).
    embedded_error = data.get("error") if isinstance(data, dict) else None
    if embedded_error:
        try:
            status_code = int(embedded_error.get("code", 502)) if isinstance(embedded_error, dict) else 502
        except (TypeError, ValueError):
            status_code = 502
        exc_class = _classify(status_code)
        raise exc_class(
            f"Google API returned an embedded error (upstream {status_code}): "
            f"{embedded_error}",
            status=status_code,
            provider=_PROVIDER_LABEL,
            model=chosen_model,
            body=_truncate(str(data)),
        )

    return _extract_assistant_text(data)


def stream_chat(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Iterator[str]:
    """Stream the assistant reply from Google Gemini as a token iterator.

    Mirrors ``app.openrouter.stream_chat`` in contract and structure.
    Yields text deltas as they arrive; raises typed exceptions on failure.
    """
    if not messages:
        raise OpenRouterError("Cannot stream from Gemini with an empty message list.")

    chosen_model = model or GEMINI_MODEL or next(iter_models(), "")
    chosen_key = api_key if api_key is not None else GEMINI_API_KEY
    chosen_url_base = base_url if base_url is not None else GEMINI_BASE_URL

    chosen_temperature = DEFAULT_TEMPERATURE if temperature is None else temperature
    chosen_max_tokens = DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens
    chosen_timeout = HTTP_TIMEOUT_SECONDS if timeout is None else timeout

    base = chosen_url_base.rstrip("/")
    # Streaming uses the OpenAI-compatible endpoint with stream=True
    url = f"{base}/chat/completions"

    payload = _build_payload(
        messages,
        model=chosen_model,
        temperature=chosen_temperature,
        max_tokens=chosen_max_tokens,
    )
    payload = {**payload, "stream": True}
    headers = _build_headers(chosen_key)

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            stream=True,
            timeout=chosen_timeout,
        )
    except requests.RequestException as exc:
        raise OpenRouterServerError(
            f"Google API stream failed before getting a response: {exc}",
            status=None,
            model=chosen_model,
            provider=_PROVIDER_LABEL,
        ) from exc

    response.encoding = "utf-8"

    try:
        if response.status_code >= 400:
            try:
                detail = response.json()
                body_text = json.dumps(detail)[:_MAX_BODY_CHARS]
            except ValueError:
                detail = response.text
                body_text = _truncate(response.text)
            provider_header = response.headers.get("x-goog-api-runtime") or _PROVIDER_LABEL
            exc_class = _classify(response.status_code)
            raise exc_class(
                f"Google API returned HTTP {response.status_code}: {detail}",
                status=response.status_code,
                provider=provider_header,
                model=chosen_model,
                body=body_text,
            )

        # Check for embedded error in first SSE frame before iterating
        line_iter = response.iter_lines(decode_unicode=True)
        try:
            preview = next(line_iter)
        except StopIteration:
            preview = ""

        if preview and not preview.lstrip().startswith("data:"):
            try:
                preview_data = json.loads(preview)
            except ValueError:
                preview_data = None
            if isinstance(preview_data, dict) and preview_data.get("error"):
                embedded_error = preview_data["error"]
                try:
                    status_code = int(embedded_error.get("code", 502)) if isinstance(embedded_error, dict) else 502
                except (TypeError, ValueError):
                    status_code = 502
                exc_class = _classify(status_code)
                raise exc_class(
                    f"Google API stream returned an embedded error "
                    f"(upstream {status_code}): {embedded_error}",
                    status=status_code,
                    provider=_PROVIDER_LABEL,
                    model=chosen_model,
                    body=_truncate(str(preview_data)),
                )

        # Process all SSE frames
        for raw_line in chain([preview], line_iter):
            if not raw_line:
                continue
            delta = _extract_stream_delta(raw_line)
            if delta:
                yield delta

    except requests.RequestException as exc:
        raise OpenRouterServerError(
            f"Google API stream interrupted: {exc}",
            status=None,
            model=chosen_model,
            provider=_PROVIDER_LABEL,
        ) from exc
    finally:
        response.close()

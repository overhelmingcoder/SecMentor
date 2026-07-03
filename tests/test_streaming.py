"""Tests for the streaming LLM-response path (Tier 1 #1).

These tests cover two new public surfaces:

- ``app.openrouter.stream_chat`` — an SSE consumer that yields
  ``content`` deltas from an OpenRouter chat-completions response
  opened with ``stream=True``. It is the streaming counterpart of
  :func:`app.openrouter.chat`.
- ``app.router.ModelRouter.stream_chat`` — the round-robin slot
  rotation wrapper around ``openrouter.stream_chat`` that mirrors
  the policy of :meth:`app.router.ModelRouter.chat`.

We never touch the real network. The transport is stubbed with a
duck-typed ``_FakeResponse`` that quacks like
``requests.Response`` (status_code, iter_lines, close) and a
``_FakeStream`` that produces canned SSE lines. The router is
exercised by patching ``app.openrouter.stream_chat`` to a function
that returns a generator of strings, the same pattern
``tests/test_smoke.py`` uses to patch ``app.openrouter.chat``.

House style follows the rest of ``tests/``: ``unittest.TestCase``,
duck-typed fakes, ``os.path.dirname`` + ``sys.path`` bootstrap so
``from app...`` resolves regardless of cwd, ``pytestmark`` for
marker compatibility with ``pytest -m smoke``.
"""

import base64
import importlib
import json
import os
import sys
import unittest
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.smoke

# --- Path bootstrap ----------------------------------------------------------
# Mirror the project-root-cd pattern used in tests/test_smoke.py so this
# file works no matter where the user invokes `python -m unittest` from.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


# --- Transport fakes ---------------------------------------------------------


class _FakeStream:
    """Duck-typed ``requests.Response`` for the streaming call.

    ``stream_chat`` calls three methods on the response: ``iter_lines``,
    ``status_code`` (a plain attribute), and ``close()``. The
    ``headers`` dict is consulted for ``x-provider`` on the error
    path; for the happy path it is irrelevant. We also expose
    ``raise_for_status`` so any caller in the future that wants to
    assert against it has a no-op to call.
    """

    def __init__(
        self,
        *,
        status_code: int = 200,
        lines: list[str] | None = None,
        json_body: object = None,
        text_body: str = "",
        headers: dict[str, str] | None = None,
    ):
        self.status_code = status_code
        self._lines = lines or []
        self._json_body = json_body
        self.text = text_body
        self.headers = headers or {}
        self.closed = False

    def iter_lines(self, decode_unicode: bool = True):  # noqa: ARG002
        # `decode_unicode=True` is the default in requests and the
        # one we always pass; we accept it for signature parity but
        # do not act on it because our canned lines are already
        # Python strings.
        for line in self._lines:
            yield line

    def close(self) -> None:
        self.closed = True

    def json(self):
        if self._json_body is None:
            raise ValueError("No JSON body configured for this fake.")
        return self._json_body


def _sse_chunk(delta_content: str) -> str:
    """Build a single OpenRouter SSE ``data:`` line for one delta."""
    payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "delta": {"content": delta_content},
                "finish_reason": None,
            }
        ],
    }
    return f"data: {json.dumps(payload)}"


def _sse_done() -> str:
    return "data: [DONE]"


def _sse_role_only() -> str:
    """First SSE event a real OpenRouter response sends: delta with no content.

    The streaming consumer must skip it (empty content) and not raise.
    """
    payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None,
            }
        ],
    }
    return f"data: {json.dumps(payload)}"


# --- openrouter.stream_chat tests ------------------------------------------


class OpenRouterStreamChatTests(unittest.TestCase):
    """Direct tests of ``app.openrouter.stream_chat``.

    These pin the wire-protocol expectations: SSE line format, the
    ``[DONE]`` sentinel, JSON delta shape, and the error classification
    that mirrors the non-streaming ``chat()``.
    """

    def test_yields_deltas_in_order(self):
        from app.openrouter import stream_chat

        response = _FakeStream(
            lines=[
                _sse_role_only(),
                _sse_chunk("Hello"),
                _sse_chunk(", "),
                _sse_chunk("world!"),
                _sse_done(),
            ]
        )
        with patch("app.openrouter.requests.post", return_value=response):
            chunks = list(
                stream_chat(
                    [{"role": "user", "content": "hi"}],
                    model="m:free",
                    api_key="k",
                    base_url="http://test/api",
                )
            )
        self.assertEqual(chunks, ["Hello", ", ", "world!"])

    def test_terminates_on_done_sentinel(self):
        from app.openrouter import stream_chat

        response = _FakeStream(
            lines=[_sse_chunk("one"), _sse_done(), _sse_chunk("SHOULD NOT APPEAR")]
        )
        with patch("app.openrouter.requests.post", return_value=response):
            chunks = list(
                stream_chat(
                    [{"role": "user", "content": "hi"}],
                    model="m:free",
                    api_key="k",
                    base_url="http://test/api",
                )
            )
        self.assertEqual(chunks, ["one"])

    def test_skips_malformed_sse_lines(self):
        from app.openrouter import stream_chat

        response = _FakeStream(
            lines=[
                "this is not an sse line",
                "",
                _sse_chunk("kept"),
                "data: {not valid json",
                _sse_chunk("also kept"),
                _sse_done(),
            ]
        )
        with patch("app.openrouter.requests.post", return_value=response):
            chunks = list(
                stream_chat(
                    [{"role": "user", "content": "hi"}],
                    model="m:free",
                    api_key="k",
                    base_url="http://test/api",
                )
            )
        self.assertEqual(chunks, ["kept", "also kept"])

    def test_skips_empty_delta_strings(self):
        from app.openrouter import stream_chat

        response = _FakeStream(
            lines=[
                _sse_chunk(""),
                _sse_chunk("real"),
                _sse_chunk(""),
                _sse_done(),
            ]
        )
        with patch("app.openrouter.requests.post", return_value=response):
            chunks = list(
                stream_chat(
                    [{"role": "user", "content": "hi"}],
                    model="m:free",
                    api_key="k",
                    base_url="http://test/api",
                )
            )
        self.assertEqual(chunks, ["real"])

    def test_closes_response_even_when_consumer_stops(self):
        """The ``finally`` block in stream_chat must close the response.

        If a user navigates away mid-stream, the generator gets
        garbage-collected. Without the finally block, the underlying
        TCP connection leaks. We exercise the path by partially
        consuming the stream and then breaking out.
        """
        from app.openrouter import stream_chat

        response = _FakeStream(
            lines=[_sse_chunk("a"), _sse_chunk("b"), _sse_chunk("c"), _sse_done()]
        )
        with patch("app.openrouter.requests.post", return_value=response):
            gen = stream_chat(
                [{"role": "user", "content": "hi"}],
                model="m:free",
                api_key="k",
                base_url="http://test/api",
            )
            # Pull only the first chunk, then drop the generator.
            first = next(gen)
            gen = None  # noqa: F841 — force GC of the generator
        self.assertEqual(first, "a")
        self.assertTrue(response.closed, "stream_chat must close the response on early exit")

    def test_raises_on_empty_message_list(self):
        from app.openrouter import OpenRouterError, stream_chat

        with self.assertRaises(OpenRouterError):
            list(stream_chat([], model="m:free", api_key="k", base_url="http://test/api"))

    def test_classifies_4xx_as_client_error(self):
        from app.openrouter import (
            OpenRouterClientError,
            stream_chat,
        )

        response = _FakeStream(
            status_code=400,
            text_body='{"error": "bad model id"}',
            json_body={"error": "bad model id"},
        )
        with patch("app.openrouter.requests.post", return_value=response):
            with self.assertRaises(OpenRouterClientError) as ctx:
                list(
                    stream_chat(
                        [{"role": "user", "content": "hi"}],
                        model="m:free",
                        api_key="k",
                        base_url="http://test/api",
                    )
                )
        self.assertEqual(ctx.exception.status, 400)
        self.assertIn("bad model id", ctx.exception.body or "")

    def test_classifies_401_as_auth_error(self):
        from app.openrouter import OpenRouterAuthError, stream_chat

        response = _FakeStream(status_code=401, text_body="unauthorized", json_body={"error": "no"})
        with patch("app.openrouter.requests.post", return_value=response):
            with self.assertRaises(OpenRouterAuthError):
                list(
                    stream_chat(
                        [{"role": "user", "content": "hi"}],
                        model="m:free",
                        api_key="bad",
                        base_url="http://test/api",
                    )
                )

    def test_classifies_429_as_rate_limit_error(self):
        from app.openrouter import OpenRouterRateLimitError, stream_chat

        response = _FakeStream(status_code=429, text_body="slow down", json_body={"error": "no"})
        with patch("app.openrouter.requests.post", return_value=response):
            with self.assertRaises(OpenRouterRateLimitError):
                list(
                    stream_chat(
                        [{"role": "user", "content": "hi"}],
                        model="m:free",
                        api_key="k",
                        base_url="http://test/api",
                    )
                )

    def test_classifies_5xx_as_server_error(self):
        from app.openrouter import OpenRouterServerError, stream_chat

        response = _FakeStream(status_code=502, text_body="bad gateway", json_body={"error": "no"})
        with patch("app.openrouter.requests.post", return_value=response):
            with self.assertRaises(OpenRouterServerError):
                list(
                    stream_chat(
                        [{"role": "user", "content": "hi"}],
                        model="m:free",
                        api_key="k",
                        base_url="http://test/api",
                    )
                )

    def test_wraps_network_failure_during_request_open(self):
        import requests as real_requests
        from app.openrouter import OpenRouterServerError, stream_chat

        with patch(
            "app.openrouter.requests.post",
            side_effect=real_requests.ConnectionError("refused"),
        ):
            with self.assertRaises(OpenRouterServerError) as ctx:
                list(
                    stream_chat(
                        [{"role": "user", "content": "hi"}],
                        model="m:free",
                        api_key="k",
                        base_url="http://test/api",
                    )
                )
        self.assertIsNone(ctx.exception.status)

    def test_wraps_mid_stream_network_failure(self):
        import requests as real_requests
        from app.openrouter import OpenRouterServerError, stream_chat

        def explode_on_iter():
            raise real_requests.ConnectionError("dropped")
            yield  # pragma: no cover — make this a generator

        # iter_lines is a generator method; easiest way to make it
        # raise mid-iteration is to wrap it in a generator that
        # raises on first next(). We achieve that by using a
        # generator function for iter_lines.
        class ExplodingResponse(_FakeStream):
            def iter_lines(self, decode_unicode: bool = True):  # noqa: ARG002
                def _gen():
                    yield _sse_chunk("first")
                    raise real_requests.ConnectionError("dropped")
                return _gen()

        with patch("app.openrouter.requests.post", return_value=ExplodingResponse()):
            with self.assertRaises(OpenRouterServerError):
                list(
                    stream_chat(
                        [{"role": "user", "content": "hi"}],
                        model="m:free",
                        api_key="k",
                        base_url="http://test/api",
                    )
                )

    def test_passes_stream_true_payload_flag(self):
        """The HTTP body must include ``stream: true`` so OpenRouter
        opens an SSE response. Regression: forgetting this flag
        turns the call into a buffered response and we never get
        any chunks.
        """
        from app.openrouter import stream_chat

        captured: dict[str, object] = {}

        def fake_post(url, *, json=None, headers=None, stream=None, timeout=None):  # noqa: A002
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            captured["stream"] = stream
            captured["timeout"] = timeout
            return _FakeStream(lines=[_sse_chunk("ok"), _sse_done()])

        with patch("app.openrouter.requests.post", side_effect=fake_post):
            list(
                stream_chat(
                    [{"role": "user", "content": "hi"}],
                    model="m:free",
                    api_key="k",
                    base_url="http://test/api",
                )
            )
        self.assertTrue(captured["stream"], "stream_chat must pass stream=True to requests")
        payload = captured["json"]
        self.assertIsInstance(payload, dict)
        self.assertTrue(payload.get("stream"), "payload must include stream=true")


# --- ModelRouter.stream_chat tests ------------------------------------------


class ModelRouterStreamChatTests(unittest.TestCase):
    """Tests for ``ModelRouter.stream_chat`` slot-rotation policy.

    We never make real HTTP calls. We patch ``app.openrouter.stream_chat``
    to a function that returns a generator of canned strings, then
    assert on which (key, model) pair the router picked and how many
    calls it made.
    """

    def _build(self, keys, models, **kwargs):
        from app.router import build_from_config

        return build_from_config(keys, models, **kwargs)

    def _flatten_slot_pairs(self, router):
        """Return the list of (key, model) pairs in slot order."""
        return [(s.api_key, s.model_id) for s in router._slots]

    def test_yields_chunks_from_first_slot(self):
        router = self._build(["k1", "k2"], ["m1:free", "m2:free"])

        def fake_stream(_messages, *, model=None, api_key=None, **_kw):
            self.assertEqual(api_key, "k1")
            self.assertEqual(model, "m1:free")
            yield from ["hello", " world"]

        with patch("app.openrouter.stream_chat", side_effect=fake_stream):
            chunks = list(router.stream_chat([{"role": "user", "content": "hi"}]))
        self.assertEqual(chunks, ["hello", " world"])

    def test_rotates_to_next_slot_on_pre_stream_error(self):
        from app.openrouter import OpenRouterClientError
        from app.router import AllSlotsExhaustedError

        router = self._build(["k1", "k2"], ["m:free"])

        # First call: 400 (don't retry, rotate). Second call: 200 -> yields.
        calls: list[str] = []

        def fake_stream(_messages, *, model=None, api_key=None, **_kw):
            calls.append(api_key or "")
            if len(calls) == 1:
                raise OpenRouterClientError("bad", status=400, model=model)
            yield from ["ok"]

        with patch("app.openrouter.stream_chat", side_effect=fake_stream):
            chunks = list(router.stream_chat([{"role": "user", "content": "hi"}]))
        self.assertEqual(chunks, ["ok"])
        self.assertEqual(calls, ["k1", "k2"])

    def test_does_not_rotate_after_partial_stream(self):
        """If the upstream yields a delta then errors, we must raise
        rather than rotate to a second slot. Two interleaved streams
        would corrupt the user's view.
        """
        from app.openrouter import OpenRouterServerError

        router = self._build(["k1", "k2"], ["m:free"])

        def fake_stream(_messages, *, model=None, api_key=None, **_kw):
            assert api_key == "k1"
            yield "partial"
            raise OpenRouterServerError("dropped", status=None, model=model)

        with patch("app.openrouter.stream_chat", side_effect=fake_stream):
            with self.assertRaises(OpenRouterServerError):
                list(router.stream_chat([{"role": "user", "content": "hi"}]))

    def test_disables_slot_on_auth_error_then_exhausts(self):
        from app.openrouter import OpenRouterAuthError
        from app.router import AllSlotsExhaustedError

        router = self._build(["k1", "k2"], ["m:free"])

        def fake_stream(_messages, *, model=None, api_key=None, **_kw):
            raise OpenRouterAuthError("nope", status=401, model=model)

        with patch("app.openrouter.stream_chat", side_effect=fake_stream):
            with self.assertRaises(AllSlotsExhaustedError):
                list(router.stream_chat([{"role": "user", "content": "hi"}]))
        # Both slots disabled.
        self.assertEqual(router.healthy_slot_count(), 0)

    def test_raises_all_slots_exhausted_when_every_slot_fails(self):
        from app.openrouter import OpenRouterServerError
        from app.router import AllSlotsExhaustedError

        router = self._build(["k1", "k2"], ["m:free"], sleep=lambda _s: None)

        def fake_stream(_messages, *, model=None, api_key=None, **_kw):
            raise OpenRouterServerError("down", status=502, model=model)

        with patch("app.openrouter.stream_chat", side_effect=fake_stream):
            with self.assertRaises(AllSlotsExhaustedError) as ctx:
                list(router.stream_chat([{"role": "user", "content": "hi"}]))
        # ``tried_slots`` carries the router's short_label format
        # ``"<model> via ****"`` because we redact keys for the
        # view. Assert against the model name and the redacted-key
        # pattern rather than the raw key.
        self.assertEqual(len(ctx.exception.tried_slots), 2)
        for label in ctx.exception.tried_slots:
            self.assertIn("m:free", label)
            self.assertIn("via", label)
            self.assertIn("****", label)

    def test_advances_cursor_on_success(self):
        router = self._build(["k1", "k2"], ["m:free"])

        def fake_stream(_messages, *, model=None, api_key=None, **_kw):
            yield "ok"

        with patch("app.openrouter.stream_chat", side_effect=fake_stream):
            for _ in range(3):
                list(router.stream_chat([{"role": "user", "content": "hi"}]))

        # After 3 successful calls, the cursor should have advanced
        # by 3. With 2 slots that means we wrap around twice and end
        # up at index 1 (3 mod 2).
        self.assertEqual(router._start_index, 1)

    def test_empty_stream_generator_continues_to_next_slot(self):
        """If the upstream returns zero deltas before [DONE], the
        router should treat that as a transient failure on this slot
        and try the next one. A real-world cause is a safety filter
        on the upstream side that swallows the reply.
        """
        router = self._build(["k1", "k2"], ["m:free"])

        def fake_stream(_messages, *, model=None, api_key=None, **_kw):
            if api_key == "k1":
                # An empty generator — the request opened but the
                # upstream yielded zero deltas before [DONE].
                return
                yield  # pragma: no cover — makes this a generator

            yield "real answer"

        with patch("app.openrouter.stream_chat", side_effect=fake_stream):
            chunks = list(router.stream_chat([{"role": "user", "content": "hi"}]))
        self.assertEqual(chunks, ["real answer"])

    def test_no_healthy_slots_raises_without_network_call(self):
        """If every slot is already disabled, the router must raise
        immediately without ever calling ``openrouter.stream_chat``.
        """
        from app.openrouter import OpenRouterAuthError
        from app.router import AllSlotsExhaustedError

        router = self._build(["k1", "k2"], ["m:free"], sleep=lambda _s: None)

        def fake_stream(_messages, **_kw):
            raise OpenRouterAuthError("nope", status=401)

        # First call disables every slot.
        with patch("app.openrouter.stream_chat", side_effect=fake_stream):
            with self.assertRaises(AllSlotsExhaustedError):
                list(router.stream_chat([{"role": "user", "content": "hi"}]))

        # Now stream_chat must short-circuit without calling upstream.
        called = {"n": 0}

        def counting(*_a, **_k):
            called["n"] += 1
            yield "should not be reached"

        with patch("app.openrouter.stream_chat", side_effect=counting):
            with self.assertRaises(AllSlotsExhaustedError):
                list(router.stream_chat([{"role": "user", "content": "hi"}]))
        self.assertEqual(called["n"], 0)

    def test_model_pin_filters_slots_by_model(self):
        """``stream_chat(model=...)`` must skip slots whose ``model_id``
        does not match, even when other healthy slots exist. This is the
        pin path the file-bearing turn takes so a vision model cannot be
        swapped for a text-only model mid-stream.
        """
        router = self._build(["k1", "k2"], ["vision:free", "text:free"])

        calls: list[tuple[str | None, str | None]] = []

        def fake_stream(_messages, *, model=None, api_key=None, **_kw):
            calls.append((api_key, model))
            yield "vision says hi"

        with patch("app.openrouter.stream_chat", side_effect=fake_stream):
            chunks = list(
                router.stream_chat(
                    [{"role": "user", "content": "see image"}],
                    model="vision:free",
                )
            )
        self.assertEqual(chunks, ["vision says hi"])
        # Both keys may be tried for vision:free, but no slot should ever
        # have been called with the text model.
        self.assertTrue(
            all(model == "vision:free" for _, model in calls),
            f"non-vision slot was called: {calls}",
        )
        self.assertGreaterEqual(len(calls), 1)

    def test_model_pin_with_no_matching_slot_raises(self):
        """If the requested ``model`` does not match any configured slot,
        ``stream_chat`` must raise :class:`AllSlotsExhaustedError` — the
        same error the unpinned path raises when every slot fails. We do
        *not* want a silent fallback to a different model.
        """
        from app.router import AllSlotsExhaustedError

        router = self._build(["k1"], ["text:free"])

        called = {"n": 0}

        def counting(*_a, **_k):
            called["n"] += 1
            yield "should not be reached"

        with patch("app.openrouter.stream_chat", side_effect=counting):
            with self.assertRaises(AllSlotsExhaustedError):
                list(
                    router.stream_chat(
                        [{"role": "user", "content": "see image"}],
                        model="vision:free",
                    )
                )
        self.assertEqual(called["n"], 0)

    def test_model_pin_rotates_keys_for_pinned_model(self):
        """When two keys are configured for the same pinned model, the
        router must still rotate across keys on a pre-stream failure. The
        pin restricts the model pool; the key rotation policy is
        unchanged.
        """
        from app.openrouter import OpenRouterClientError

        router = self._build(["k1", "k2"], ["vision:free"])

        calls: list[str] = []

        def fake_stream(_messages, *, model=None, api_key=None, **_kw):
            calls.append(api_key or "")
            if len(calls) == 1:
                raise OpenRouterClientError("bad", status=400, model=model)
            yield from ["done"]

        with patch("app.openrouter.stream_chat", side_effect=fake_stream):
            chunks = list(
                router.stream_chat(
                    [{"role": "user", "content": "see image"}],
                    model="vision:free",
                )
            )
        self.assertEqual(chunks, ["done"])
        self.assertEqual(calls, ["k1", "k2"])


# --- ModelRouter.stream_chat per-call tunables -----------------------------


class ModelRouterStreamChatTunableTests(unittest.TestCase):
    """Pin the per-call ``max_attempts`` and ``rate_limit_cooldown_seconds``
    kwargs on ``ModelRouter.stream_chat``.

    The vision caller (see ``web/chat_helpers.stream_vision_turn_with_fallback``)
    needs a tighter retry budget than the generic text path because
    the free-tier Nemotron vision model shares one upstream provider
    across every (key, model) slot. Long cooldowns and high attempt
    caps turn a single 429 into a multi-minute stall. These tests
    pin the contract that ``stream_chat`` honours the per-call
    overrides.
    """

    def _build(self, keys, models, **kwargs):
        from app.router import build_from_config

        return build_from_config(keys, models, **kwargs)

    def _always_rate_limited(self, _messages, *, model=None, api_key=None, **_kw):
        """Fake ``app.openrouter.stream_chat`` that always 429s.

        Body is left ``None`` so :py:meth:`ModelRouter._backoff_for`
        returns the router's ``backoff_seconds`` default. Tests pass
        ``backoff_seconds=0`` to keep the cooldown arithmetic
        deterministic.
        """
        from app.openrouter import OpenRouterRateLimitError

        def _gen():
            raise OpenRouterRateLimitError("throttled", status=429, model=model)
            yield  # pragma: no cover — make this a generator

        return _gen()

    def test_max_attempts_override_caps_attempts(self):
        """``max_attempts=2`` must limit the router to exactly two
        upstream calls, regardless of the constructor default.

        This is the cap the vision caller relies on: even on a
        pool of four slots, we want to give up after two tries so
        the user sees a fast failure instead of waiting through
        every cooldowned slot.
        """
        from app.router import AllSlotsExhaustedError

        # 4 slots, but max_attempts=2 means only the first two
        # should ever be called before we raise.
        router = self._build(
            ["k1", "k2", "k3", "k4"],
            ["m:free"],
            max_attempts=4,  # constructor default — would normally cap at 4
            sleep=lambda _s: None,
        )

        calls: list[str] = []

        def fake_stream(_messages, *, model=None, api_key=None, **_kw):
            calls.append(api_key or "")
            return self._always_rate_limited(
                _messages, model=model, api_key=api_key, **_kw
            )

        with patch("app.openrouter.stream_chat", side_effect=fake_stream):
            with self.assertRaises(AllSlotsExhaustedError):
                list(
                    router.stream_chat(
                        [{"role": "user", "content": "hi"}],
                        max_attempts=2,  # per-call override
                    )
                )

        self.assertEqual(
            len(calls),
            2,
            f"expected exactly 2 upstream calls (per-call max_attempts=2), "
            f"got {len(calls)}: {calls}",
        )

    def test_rate_limit_cooldown_override_is_honoured(self):
        """``rate_limit_cooldown_seconds=10`` must set ``cooldown_until``
        to ~10s ahead of ``now``, not the router's constructor default.

        This is the half of the vision tunables that prevents a 429
        from wedging the slot for a full minute. We assert against
        ``cooldown_until`` directly because that is the field the
        router uses on its next iteration to decide whether the
        slot is healthy.
        """
        import time as _time

        from app.openrouter import OpenRouterRateLimitError
        from app.router import AllSlotsExhaustedError

        # Constructor default is 60s; the per-call override is 10s.
        # We also zero out ``backoff_seconds`` so the upstream-hint
        # path in ``_set_rate_limit_cooldown`` cannot inflate the
        # window above the override.
        router = self._build(
            ["k1"],
            ["m:free"],
            rate_limit_cooldown_seconds=60.0,
            backoff_seconds=0.0,
            sleep=lambda _s: None,
        )

        def fake_stream(_messages, *, model=None, api_key=None, **_kw):
            # Body=None → _backoff_for returns 0.0 → window stays
            # at the override value (10.0).
            raise OpenRouterRateLimitError(
                "throttled", status=429, model=model, body=None
            )

        before = _time.monotonic()
        with patch("app.openrouter.stream_chat", side_effect=fake_stream):
            with self.assertRaises(AllSlotsExhaustedError):
                list(
                    router.stream_chat(
                        [{"role": "user", "content": "hi"}],
                        rate_limit_cooldown_seconds=10.0,
                    )
                )
        after = _time.monotonic()

        slot = router._slots[0]
        self.assertIsNotNone(
            slot.cooldown_until,
            "expected cooldown_until to be set after a 429",
        )
        # The cooldown window is recorded as now+override at the
        # moment _set_rate_limit_cooldown ran. We measure it from
        # both ``before`` and ``after`` so the test is robust to
        # which side of the call the monotonic clock ticks on.
        cooldown_remaining = slot.cooldown_until - before
        self.assertGreater(
            cooldown_remaining,
            9.0,
            f"cooldown should be ~10s, got {cooldown_remaining:.2f}s",
        )
        self.assertLess(
            cooldown_remaining,
            11.0,
            f"cooldown should not exceed the override by much, "
            f"got {cooldown_remaining:.2f}s",
        )
        # And it must not be the constructor default of ~60s.
        self.assertLess(
            cooldown_remaining,
            30.0,
            "cooldown looks like it used the constructor default "
            "(60s), not the per-call override",
        )
        # Sanity: the slot was actually marked in cooldown.
        self.assertTrue(
            slot.is_in_cooldown(_time.monotonic()),
            "slot should be in cooldown immediately after the 429",
        )
        # And ``after`` is strictly later, just so the assertion
        # above has a meaningful upper bound.
        self.assertGreaterEqual(after, before)

    def test_default_cooldown_used_when_no_override(self):
        """When no ``rate_limit_cooldown_seconds`` override is
        passed, ``stream_chat`` must fall back to the router's
        constructor value. This is the regression guard for the
        default path — we must not have broken the generic text
        caller.
        """
        import time as _time

        from app.openrouter import OpenRouterRateLimitError
        from app.router import AllSlotsExhaustedError

        # Constructor default is 60s; we pass nothing on the call.
        router = self._build(
            ["k1"],
            ["m:free"],
            rate_limit_cooldown_seconds=60.0,
            backoff_seconds=0.0,
            sleep=lambda _s: None,
        )

        def fake_stream(_messages, *, model=None, **_kw):
            raise OpenRouterRateLimitError("throttled", status=429, model=model, body=None)

        before = _time.monotonic()
        with patch("app.openrouter.stream_chat", side_effect=fake_stream):
            with self.assertRaises(AllSlotsExhaustedError):
                list(router.stream_chat([{"role": "user", "content": "hi"}]))

        slot = router._slots[0]
        self.assertIsNotNone(slot.cooldown_until)
        cooldown_remaining = slot.cooldown_until - before
        self.assertGreater(
            cooldown_remaining,
            55.0,
            f"default cooldown should be ~60s, got {cooldown_remaining:.2f}s",
        )

    def test_max_attempts_override_only_takes_effect_when_positive(self):
        """A ``max_attempts`` override of 0 or a negative number
        must not bypass the constructor cap. We treat the override
        as "set this explicitly" only when it is a positive
        integer.
        """
        from app.openrouter import OpenRouterRateLimitError
        from app.router import AllSlotsExhaustedError

        # 4 slots, constructor cap 2. The override of 0 must be
        # ignored so we still see 2 upstream calls (the constructor
        # cap), not 0 (which would short-circuit and never call
        # upstream) and not 4 (which would ignore the cap entirely).
        router = self._build(
            ["k1", "k2", "k3", "k4"],
            ["m:free"],
            max_attempts=2,
            sleep=lambda _s: None,
        )

        calls: list[str] = []

        def fake_stream(_messages, *, model=None, api_key=None, **_kw):
            calls.append(api_key or "")
            raise OpenRouterRateLimitError(
                "throttled", status=429, model=model, body=None
            )

        with patch("app.openrouter.stream_chat", side_effect=fake_stream):
            with self.assertRaises(AllSlotsExhaustedError):
                list(
                    router.stream_chat(
                        [{"role": "user", "content": "hi"}],
                        max_attempts=0,  # must be ignored → cap=2 wins
                    )
                )

        # Two upstream calls — matches the constructor cap. If the
        # override had been honoured, we'd see 0 calls. If the cap
        # had been bypassed entirely, we'd see up to 4.
        self.assertEqual(
            len(calls),
            2,
            f"max_attempts=0 must fall back to the constructor cap (2); "
            f"got {len(calls)} calls: {calls}",
        )
        # And the slots are merely in cooldown, not permanently
        # disabled (429 ≠ auth failure).
        self.assertTrue(
            all(not s.disabled for s in router._slots),
            "429s must not disable slots",
        )


if __name__ == "__main__":
    unittest.main()

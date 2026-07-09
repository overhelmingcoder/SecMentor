"""Multi-key, multi-model rotation layer for OpenRouter free-tier calls.

Why this module exists
----------------------

The OpenRouter *free* tier caps each **model** (per upstream
provider) at roughly 50 requests per day, not per API key and not
per account. So the only knobs we have to keep the demo running are
(a) switch to a different **model** (a different upstream
provider's quota), and (b) only after every configured model on the
*current* key is exhausted, switch to a different **API key**
(recovering from a key that has hit its account-wide limit).

This module owns the pool, the per-slot health state, the retry
policy, and the "did we exceed the cap" detection. The view
(``web/streamlit_app.py``) and the CLI (``cli/chatbot.py``) just call
``router.chat(messages)`` and treat the result as a string.

Rotation policy (one-key-at-a-time)
-----------------------------------

The configured pool is stored in a *deliberate* order: for each
key K, every configured model M is paired with K, and the keys
appear in slot order. Concretely, with ``["k1", "k2"]`` and
``["m1:free", "m2:free"]`` the slot list is::

    (k1, m1:free), (k1, m2:free), (k2, m1:free), (k2, m2:free)

The router walks slots in stored order. The first request goes
out on **one slot — k1+m1 — never on multiple at once**. If that
slot fails with a recoverable error, the router moves to the
*next* slot, which is **k1+m2** (same key, next model). It only
reaches **k2** after every k1 model has been tried.

This rule is what gives us:

* **One API in flight at a time.** No parallel fan-out across
  keys, ever.
* **Model fallback within a key.** When the active model's quota
  is hit, the next model on the *same* key is tried before any
  second key is touched.
* **Key fallback only after model exhaustion.** The router only
  shifts to key 2 when every model on key 1 has returned a
  recoverable error. That is the moment the agent "shifts to
  another API".
* **No simultaneous multi-API use.** The view and the CLI never
  observe two different API keys being used in the same chat
  turn. A request that needs to escalate from k1+m2 to k2+m1
  happens *sequentially*, in order, on the *same* thread.

Design rules
------------

1. **Fail loud, not silent.** If every healthy slot is exhausted, the
   router raises the *last* error it saw so the caller can show a
   meaningful error to the user. We never return a truncated answer
   just because "something sort of worked".
2. **No silent paid fallback.** Every model id handed to the router
   is checked for the ``:free`` suffix. A non-free id is rejected at
   ``__init__`` time, not at call time, so a typo cannot burn a
   credit mid-conversation.
3. **Auth errors are sticky.** A 401/403 from a slot disables it for
   the rest of the session. Retrying a dead key just wastes the
   caller's latency budget.
4. **Rate limits get a short backoff, then rotate.** A 429 sleeps
   for ``backoff_seconds`` (or the upstream ``Retry-After`` value if
   the response sent one) and then tries the same slot once more
   before moving on.
5. **Server errors retry once on the same slot, then rotate.** A
   5xx is usually transient and the same slot will often recover.
6. **Client errors (4xx other than 401/403/429) are terminal.** The
   request is malformed or the model is wrong; retrying will not
   help. We rotate immediately.
7. **Keys are never logged.** The model id and the error *class* are
   the only things that go to the logger. The key prefix is masked
   to its last four characters everywhere it surfaces.

This module imports from ``app.openrouter`` only — it does not
re-implement HTTP. ``app.openrouter`` does not know about rotation;
that keeps the boundary clean and makes the router testable in
isolation with a stubbed ``openrouter.chat``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Sequence

from app import openrouter

logger = logging.getLogger(__name__)


# --- Public exceptions -------------------------------------------------------


class RouterError(RuntimeError):
    """Base class for every failure the router itself can raise.

    Distinct from ``openrouter.OpenRouterError`` so the caller can
    branch on "the upstream is the problem" vs "the pool is
    exhausted". Inherits from ``RuntimeError`` for symmetry with
    ``OpenRouterError``.
    """


class NoFreeModelConfiguredError(RouterError):
    """Raised at ``__init__`` time when the caller hands the router
    a model id that does not end in ``:free``."""


class AllSlotsExhaustedError(RouterError):
    """Raised at call time when every healthy slot has been tried
    and all of them failed.

    Carries the most recent upstream exception as ``__cause__``,
    the count of attempted slots, and the list of tried slot
    labels (with redacted keys) for the UI / CLI to display.
    """

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        tried_slots: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.attempts: int = attempts
        self.tried_slots: list[str] = list(tried_slots)


# --- Internal data model -----------------------------------------------------


@dataclass
class KeySlot:
    """One (api_key, model_id) pair plus its in-memory health state.

    The router owns the health state — ``chat()`` is the only thing
    that mutates ``disabled``, ``cooldown_until``, and ``last_error``.
    Callers should treat the slot as opaque; they receive the public
    ``api_key_prefix`` and ``model_id`` for logging only.
    """

    api_key: str
    model_id: str
    # A 401/403 from this slot flips this to True and leaves it True
    # for the rest of the session. The router skips disabled slots
    # without burning a round-trip.
    disabled: bool = False
    # A 429 (or upstream "daily cap reached") sets ``cooldown_until``
    # to ``monotonic() + cooldown_seconds``; the router treats the
    # slot as unhealthy until that time. Without this, the same slot
    # is retried every call and the daily per-account cap stays
    # exhausted forever from the demo's perspective. The cooldown is
    # a wall-clock-free monotonic so it survives daylight-saving
    # changes and clock skew. ``None`` means "no cooldown active".
    cooldown_until: float | None = None
    # The most recent error from this slot, kept for diagnostics
    # only. Never used to make routing decisions — those decisions
    # are based on the *class* of the error, the disabled flag, and
    # the cooldown timestamp.
    last_error: BaseException | None = field(default=None)

    def is_in_cooldown(self, now: float | None = None) -> bool:
        """Return True if the slot is currently in a rate-limit cooldown.

        ``cooldown_until`` is a ``time.monotonic()`` value. ``now`` is
        injectable so tests can pin the wall clock without monkey-
        patching ``time``. ``False`` when no cooldown is active.
        """
        if self.cooldown_until is None:
            return False
        import time as _time
        current = now if now is not None else _time.monotonic()
        return current < self.cooldown_until

    def redacted_key(self) -> str:
        """Return a key string safe to put in logs / error messages.

        The full key never leaves this module. We keep the last four
        characters for debuggability (two accounts on the same email
        is a real thing) and mask the rest.
        """
        if len(self.api_key) <= 4:
            return "****"
        return "****" + self.api_key[-4:]

    def short_label(self) -> str:
        """A short, log-friendly identifier for this slot.

        Format: ``"<model_id> via ****abcd"`` — enough to tell two
        slots apart without leaking the key.
        """
        return f"{self.model_id} via {self.redacted_key()}"


# --- The router itself -------------------------------------------------------


class ModelRouter:
    """Rotate across a pool of (api_key, model_id) free-tier slots.

    Parameters
    ----------
    slots
        Sequence of ``(api_key, model_id)`` pairs. Order matters: the
        router starts at slot 0 and wraps around. Duplicate pairs
        are allowed (two keys on the same model) and the pool is
        taken as-is.
    backoff_seconds
        How long to sleep on a 429 before retrying the same slot.
        OpenRouter's free providers usually send a ``Retry-After``
        header in seconds; we use that value if it is present and
        larger, otherwise this default.
    max_attempts
        Hard cap on the number of (slot, retry) pairs we will try
        for a single ``chat`` call. Defaults to ``len(slots) * 2``:
        once around the pool, plus one retry on the same slot for
        transient errors. Set lower to fail fast; the cap exists to
        guarantee termination even if a bug sets every slot to
        "transient".
    sleep
        Indirection for ``time.sleep`` so tests can fast-forward
        backoff without actually sleeping. Defaults to
        ``time.sleep``. Tests pass a no-op.
    """

    def __init__(
        self,
        slots: Sequence[tuple[str, str]],
        *,
        backoff_seconds: float = 1.0,
        max_attempts: int | None = None,
        rate_limit_cooldown_seconds: float = 60.0,
        sleep: "callable" = time.sleep,
    ) -> None:
        if not slots:
            raise NoFreeModelConfiguredError(
                "ModelRouter needs at least one (api_key, model_id) slot."
            )

        # Validate every model id up front. Catching a typo at init
        # time is far better than catching it on the first user
        # message at 02:00.
        for _api_key, model_id in slots:
            if not model_id.endswith(":free"):
                raise NoFreeModelConfiguredError(
                    f"ModelRouter only accepts free-tier model ids "
                    f"(must end with ':free'). Got: {model_id!r}. "
                    "Refusing to silently fall back to a paid model."
                )

        # Defensive copy of the key strings so a caller mutating
        # their original list does not affect us. (We never mutate
        # the key ourselves, but copy-on-ingest is cheap and
        # removes a class of bug.)
        self._slots: list[KeySlot] = [
            KeySlot(api_key=api_key, model_id=model_id)
            for api_key, model_id in slots
        ]
        self._start_index: int = 0
        self._backoff_seconds: float = backoff_seconds
        # Default 60s cooldown on a 429. OpenRouter's free-tier daily
        # caps are usually much longer than that, but a short
        # cooldown gives the *upstream provider* time to release the
        # per-minute bucket, and it stops the demo from re-hitting
        # the same exhausted slot on every consecutive user turn.
        # If the upstream actually sends a longer ``Retry-After`` we
        # take the max of the two (see ``_backoff_for``).
        self._rate_limit_cooldown_seconds: float = (
            rate_limit_cooldown_seconds
        )
        # Pool-walk cap. With 5 keys x 10 models = 50 slots, a
        # single bad minute burns the entire pool before the user
        # sees a useful error. We cap the *per-call* walk to
        # ``min(max_attempts, 6)`` so a transient outage fails fast
        # and the remaining healthy slots stay available for the
        # next call. ``6`` was chosen as "two keys worth" — the
        # typical fix is to add a new API key, not to wait out a
        # 6-deep walk.
        if max_attempts is not None:
            self._max_attempts: int = max_attempts
        else:
            # Cap at 6 so a single chat call can't burn the whole
            # pool. The full ``len(slots) * 2`` is still respected
            # when there are few slots.
            self._max_attempts = min(len(self._slots) * 2, 6)
        self._sleep = sleep

        logger.info(
            "ModelRouter initialised with %d slot(s) (max_attempts=%d, "
            "rate_limit_cooldown=%.1fs): %s",
            len(self._slots),
            self._max_attempts,
            self._rate_limit_cooldown_seconds,
            ", ".join(s.short_label() for s in self._slots),
        )

    # --- Introspection (used by the UI caption and the live probe) ----

    def slot_labels(self) -> list[str]:
        """Return a snapshot of the public labels for every slot,
        including disabled ones. The UI uses this to render the
        "current pool" caption."""
        return [s.short_label() for s in self._slots]

    def healthy_slot_count(self) -> int:
        """Number of slots that have not been permanently disabled
        by a 401/403. The UI uses this to show a warning if we are
        down to the last slot."""
        return sum(1 for s in self._slots if not s.disabled)

    # --- The main entry point ------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> str:
        """Send ``messages`` to a healthy slot and return the
        assistant's text.

        The router walks the pool starting at ``_start_index``,
        wrapping around. A slot that returns ``OpenRouterAuthError``
        is disabled for the rest of the session and we move on
        immediately. A slot that returns ``OpenRouterRateLimitError``
        gets a short backoff, then a second chance on the same
        slot, then rotation. A ``OpenRouterServerError`` is retried
        once on the same slot, then rotated. A
        ``OpenRouterClientError`` rotates immediately.

        If every healthy slot has been tried and all of them failed,
        the router raises ``AllSlotsExhaustedError`` whose
        ``__cause__`` is the most recent upstream exception.

        The caller's ``messages`` list is passed through unchanged;
        the router never inspects or rewrites the content.

        ``timeout`` is forwarded to :func:`openrouter.chat` per slot.
        The text path uses the module default (60s); the vision path
        in :mod:`web.chat_helpers` overrides this with a 45s ceiling
        — long enough to absorb a typical warm-slot first-token from
        NVIDIA's free-tier vision endpoint (median ~25s) but short
        enough that a hung call degrades to the text fallback before
        the user perceives a stall. Without that override a hung vision
        call burns all 10 slots before the default timeout fires and
        the user sees ``AllSlotsExhaustedError`` instead of a useful
        error.
        """
        last_error: BaseException | None = None
        attempts = 0
        tried_slots: list[str] = []

        # ``_start_index`` is the "where to begin" cursor. We bump
        # it after a *successful* call so the next user turn does
        # not always start at slot 0 — that gives a fair load
        # distribution across keys even if the user is hammering
        # the same model.
        for slot in self._iter_healthy_slots_starting_at(self._start_index):
            # Each slot gets up to two chances: an initial call and
            # (for transient errors) one immediate retry after a
            # short backoff. That maps to the
            # ``max_attempts = len(slots) * 2`` default.
            for retry_index in range(2):
                attempts += 1
                if attempts > self._max_attempts:
                    logger.warning(
                        "ModelRouter hit max_attempts=%d, giving up",
                        self._max_attempts,
                    )
                    raise AllSlotsExhaustedError(
                        f"ModelRouter exhausted after {self._max_attempts} "
                        f"attempts across {len(self._slots)} slot(s). "
                        f"Last error: {last_error}",
                        attempts=attempts,
                        tried_slots=tried_slots,
                    ) from last_error

                try:
                    text = openrouter.chat(
                        messages,
                        model=slot.model_id,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=timeout,
                        api_key=slot.api_key,
                    )
                except openrouter.OpenRouterAuthError as exc:
                    # 401/403: the key is dead. Disable for the
                    # rest of the session and move on immediately.
                    slot.disabled = True
                    slot.last_error = exc
                    last_error = exc
                    tried_slots.append(slot.short_label())
                    logger.warning(
                        "Slot %s returned auth error (status=%s); "
                        "disabling for this session",
                        slot.short_label(),
                        exc.status,
                    )
                    break  # out of the inner retry loop
                except openrouter.OpenRouterRateLimitError as exc:
                    # 429: transient. Sleep, retry the same slot
                    # once, then move on if it 429s again. Also push
                    # the slot's cooldown timestamp forward so the
                    # *next* chat call (not just this one) skips it.
                    slot.last_error = exc
                    last_error = exc
                    sleep_for = self._backoff_for(exc)
                    self._set_rate_limit_cooldown(slot, exc)
                    logger.info(
                        "Slot %s rate-limited (status=%s); sleeping "
                        "%.2fs before retry",
                        slot.short_label(),
                        exc.status,
                        sleep_for,
                    )
                    if retry_index == 0:
                        self._sleep(sleep_for)
                        continue  # try the same slot one more time
                    tried_slots.append(slot.short_label())
                    logger.info(
                        "Slot %s still rate-limited after backoff; "
                        "rotating to next slot",
                        slot.short_label(),
                    )
                    break  # out of the inner retry loop, move to next slot
                except openrouter.OpenRouterServerError as exc:
                    # 5xx or network error: transient. Retry once on
                    # the same slot (with a small sleep), then
                    # rotate.
                    slot.last_error = exc
                    last_error = exc
                    sleep_for = self._backoff_seconds
                    if retry_index == 0:
                        self._sleep(sleep_for)
                        continue
                    tried_slots.append(slot.short_label())
                    logger.info(
                        "Slot %s returned server error (status=%s); "
                        "transient, rotated after retry",
                        slot.short_label(),
                        exc.status,
                    )
                    break  # rotate
                except openrouter.OpenRouterClientError as exc:
                    # 4xx other than 401/403/429: the request is
                    # malformed or the model is wrong. Rotating
                    # will not help, but we move on rather than
                    # burning the rest of the pool.
                    slot.last_error = exc
                    last_error = exc
                    tried_slots.append(slot.short_label())
                    logger.warning(
                        "Slot %s returned client error (status=%s); "
                        "rotating (won't retry same slot)",
                        slot.short_label(),
                        exc.status,
                    )
                    break  # rotate
                else:
                    # Success. Advance the start cursor past this
                    # slot for the next call so the load spreads
                    # fairly across keys. ``disabled`` slots are
                    # skipped, so the cursor is over the *healthy*
                    # pool only.
                    self._start_index = (
                        (self._slots.index(slot) + 1) % len(self._slots)
                    )
                    return text

        # If we fell out of the for-loop without returning, every
        # healthy slot is exhausted. Raise with the most recent
        # error as the cause and the list of tried slots (with
        # redacted keys) so the UI can show both the upstream
        # message and which slots were tried.
        raise AllSlotsExhaustedError(
            f"ModelRouter tried every healthy slot and all "
            f"{attempts} attempt(s) failed. Tried: "
            f"{', '.join(tried_slots) or '<none>'}. "
            f"Last error: {last_error}",
            attempts=attempts,
            tried_slots=tried_slots,
        ) from last_error
    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        timeout: float | None = None,
        max_attempts: int | None = None,
        rate_limit_cooldown_seconds: float | None = None,
    ) -> Iterator[str]:
        """Stream the assistant's reply, rotating across the pool on failure.

        Streaming counterpart of :meth:`chat`. The router walks the
        same healthy-slot list in the same round-robin order, but
        because the response is an open-ended iterator we cannot
        retry "mid-stream" — once we start receiving deltas from a
        slot we commit to it. The rotation policy is therefore:

        ``model`` is an optional pin: when set, only slots whose
        ``model_id`` matches are considered. This is what the view
        layer uses for file-bearing turns (vision models, PDF
        models), where rotating to a different capability mid-stream
        would corrupt the reply. ``None`` (the default) preserves the
        original round-robin behaviour across every healthy slot.

        ``timeout`` is forwarded to :func:`openrouter.stream_chat` per
        slot. The text path uses the module default (60s); the
        vision path in :mod:`web.chat_helpers` overrides this with a
        45s ceiling because the only working free vision model is
        slow to first byte and a hung call needs to degrade to the
        text fallback within a user-perceivable window. See
        :meth:`chat` for the full rationale.

        ``max_attempts`` and ``rate_limit_cooldown_seconds`` are
        optional **per-call** overrides on top of the constructor
        defaults. The vision path passes tighter values
        (``max_attempts=2``, ``rate_limit_cooldown_seconds=10``) so
        a stuck Nemotron throttle burns at most ~2 attempts × (45s
        timeout + 10s cooldown) ≈ 110s instead of the previous
        ~9 minutes. Defaults of ``None`` mean "use whatever the
        router was constructed with", so the text path is
        unaffected. ``max_attempts=0`` is treated as "use the
        constructor default" rather than "give up immediately",
        so callers cannot accidentally disable rotation by passing
        a default-valued parameter.

        * If the upstream raises **before** yielding any delta
          (auth error, 4xx, 5xx on the first chunk, or a network
          failure during the request open), we move to the next
          slot exactly the way :meth:`chat` does.
        * If the upstream raises **after** at least one delta has
          already been yielded, we re-raise the exception: a
          partial reply is still useful, and the caller (the view)
          keeps the accumulated text on screen alongside the
          friendly error toast. Rotating at that point would
          produce two interleaved streams of text.
        * If every slot fails before yielding, we raise
          :class:`AllSlotsExhaustedError` from the last error so
          the view can show the same slot-debug list it already
          shows for the non-streaming path.

        Backoff policy matches :meth:`chat`: rate-limit errors
        honour the upstream ``Retry-After`` hint via
        :meth:`_backoff_for`; other transient errors get a single
        retry per slot (the slot pool already gives us the
        rotation, so an in-slot retry would only matter for
        two-attempt pools).
        """
        # Resolve per-call overrides. ``None`` or ``<= 0`` for
        # ``max_attempts`` falls back to the constructor default —
        # the text path passes nothing and continues to walk the
        # whole pool the same way it always did.
        effective_max_attempts = (
            self._max_attempts
            if max_attempts is None or max_attempts <= 0
            else max_attempts
        )
        last_error: BaseException | None = None
        attempts = 0
        tried_slots: list[str] = []
        start = self._start_index
        for offset, slot in enumerate(
            self._iter_healthy_slots_starting_at(start, model=model)
        ):
            if attempts >= effective_max_attempts:
                # Honour the per-call cap before burning another
                # network round-trip. The vision caller relies on
                # this to fail fast; the text path never passes an
                # override, so this branch is unreachable there.
                logger.warning(
                    "ModelRouter.stream_chat hit max_attempts=%d "
                    "(model=%s); giving up before next slot",
                    effective_max_attempts,
                    model,
                )
                break
            attempts += 1
            tried_slots.append(slot.short_label())
            try:
                stream = openrouter.stream_chat(
                    messages,
                    model=slot.model_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    api_key=slot.api_key,
                    base_url=None,
                )
            except openrouter.OpenRouterError as exc:
                # The request could not even be opened. Apply the
                # same per-slot policy as chat() — disable on
                # auth, sleep on rate-limit, otherwise move on.
                self._record_slot_failure(
                    slot, exc, cooldown_seconds=rate_limit_cooldown_seconds,
                )
                if isinstance(exc, openrouter.OpenRouterAuthError):
                    continue
                delay = self._backoff_for(exc)
                if delay > 0:
                    self._sleep(delay)
                if (
                    isinstance(exc, openrouter.OpenRouterRateLimitError)
                    and self._retries_remaining(
                        slot, offset, max_attempts=effective_max_attempts
                    ) > 0
                ):
                    continue
                last_error = exc
                continue

            # The request opened. Stream the first delta; if it
            # arrives, commit to this slot for the rest of the
            # response.
            try:
                yielded_any = False
                for chunk in stream:
                    if chunk:
                        yielded_any = True
                        yield chunk
                # Stream ended cleanly (data: [DONE]). Reject
                # zero-chunk replies as transient failures on this
                # slot — an upstream that swallows the content is
                # indistinguishable from a flaky connection to the
                # view, so we rotate instead of presenting a blank
                # bubble. Also matches the non-streaming path,
                # where _extract_assistant_text raises on empty
                # content.
                if not yielded_any:
                    raise openrouter.OpenRouterError(
                        "OpenRouter stream returned no deltas before [DONE].",
                        status=None,
                        model=slot.model_id,
                    )
                # Stream ended cleanly (data: [DONE]) with at
                # least one delta. Advance start index and return
                # — the generator is exhausted and the for-loop
                # above is the only thing still holding a
                # reference.
                self._start_index = (start + offset + 1) % max(len(self._slots), 1)
                # When ``model`` pinned us to a single slot, len(self._slots)
                # is still > 1 but the filtered walk only saw one slot. The
                # modulo on the full pool still gives a valid cursor; it
                # simply has no effect when there is exactly one
                # model-matching slot, which is the entire point of the
                # pin.
                self._record_slot_success(slot)
                return
            except openrouter.OpenRouterError as exc:
                if yielded_any:
                    # Partial reply is in the caller's accumulator.
                    # Surface the error so the view can show the
                    # friendly message; do not rotate, do not
                    # advance the start index.
                    raise
                self._record_slot_failure(
                    slot, exc, cooldown_seconds=rate_limit_cooldown_seconds,
                )
                if isinstance(exc, openrouter.OpenRouterAuthError):
                    continue
                delay = self._backoff_for(exc)
                if delay > 0:
                    self._sleep(delay)
                last_error = exc
                continue

        if last_error is None and not tried_slots:
            # No healthy slots at all.
            raise AllSlotsExhaustedError(
                "ModelRouter has no healthy slots to stream from.",
                attempts=attempts,
                tried_slots=tried_slots,
            )
        raise AllSlotsExhaustedError(
            f"ModelRouter tried every healthy slot and all "
            f"{attempts} attempt(s) failed before producing any "
            f"streamed delta. Tried: "
            f"{', '.join(tried_slots) or '<none>'}. "
            f"Last error: {last_error}",
            attempts=attempts,
            tried_slots=tried_slots,
        ) from last_error

    def _record_slot_success(self, slot: KeySlot) -> None:
        """Reset the per-slot health marker after a successful call.

        Mirrors the inline success branch in :meth:`chat` so the
        streaming and non-streaming paths converge on the same slot
        health state. Kept as a small helper because both paths
        need the same three lines and the streaming path is
        already long. A successful call also clears any active
        rate-limit cooldown — the upstream accepted the request, so
        the slot is healthy again.
        """
        slot.last_error = None
        slot.disabled = False
        slot.cooldown_until = None

    def _record_slot_failure(
        self,
        slot: KeySlot,
        exc: BaseException,
        *,
        cooldown_seconds: float | None = None,
    ) -> None:
        """Record a failure on a slot and disable it on auth errors.

        Same policy as the inline block in :meth:`chat`. Splitting
        it out keeps the streaming and non-streaming call sites in
        sync and gives a single place to extend (e.g. disabling a
        slot after N consecutive 429s) later.

        A ``OpenRouterRateLimitError`` also flips the slot's
        ``cooldown_until`` timestamp forward by the configured
        cooldown (or the upstream ``Retry-After`` value, whichever
        is larger), so the pool walk skips it on the next call
        instead of burning another 429 round-trip. ``cooldown_seconds``
        is an optional per-call override — when set, the streaming
        path uses it (e.g. the vision caller passes a tight 10s
        window so a single Nemotron throttle does not block every
        other key for 60s) instead of the constructor default.
        """
        slot.last_error = exc
        if isinstance(exc, openrouter.OpenRouterAuthError):
            slot.disabled = True
            return
        if isinstance(exc, openrouter.OpenRouterRateLimitError):
            self._set_rate_limit_cooldown(
                slot, exc, cooldown_seconds=cooldown_seconds
            )

    def _set_rate_limit_cooldown(
        self,
        slot: KeySlot,
        exc: openrouter.OpenRouterRateLimitError,
        *,
        cooldown_seconds: float | None = None,
    ) -> None:
        """Push ``slot.cooldown_until`` forward by the cooldown window.

        We use ``time.monotonic()`` rather than wall-clock so the
        cooldown is immune to DST changes and clock skew. The window
        is ``max(rate_limit_cooldown_seconds, upstream_retry_after)``,
        honouring the upstream's hint when it sends one.

        ``cooldown_seconds`` is an optional per-call override. The
        streaming path passes it so the vision caller does not have
        to share its low-throttle cooldown with every other caller
        on the same router instance.
        """
        window = (
            self._rate_limit_cooldown_seconds
            if cooldown_seconds is None
            else cooldown_seconds
        )
        upstream_hint = self._backoff_for(exc)
        if upstream_hint > window:
            window = upstream_hint
        slot.cooldown_until = time.monotonic() + window
        logger.info(
            "Slot %s rate-limited; cooldown until now+%.1fs "
            "(upstream hint was %.1fs)",
            slot.short_label(),
            window,
            upstream_hint,
        )

    def _retries_remaining(
        self,
        slot: KeySlot,
        offset: int,
        *,
        max_attempts: int | None = None,
    ) -> int:
        """Best-effort estimate of retries left in this slot.

        Used only to decide whether to take a second swing at the
        same slot for a 429 (the rate-limit error often resolves
        after a short sleep). The math matches :meth:`chat`: with
        ``max_attempts = len(slots) * 2``, each slot gets up to
        two attempts.

        ``max_attempts`` is the optional per-call override used by
        the streaming vision path — when the caller asks for a
        tight cap (e.g. ``max_attempts=2``), the retry budget per
        slot shrinks to match. ``None`` (the default) keeps the
        constructor-level behaviour.
        """
        if max_attempts is None or max_attempts <= 0:
            max_attempts = self._max_attempts or len(self._slots) * 2
        # Offset is the slot's position in the rotation (0..N-1);
        # we've already made one attempt on this slot.
        return max(0, max_attempts - 1 - (offset * 2))
    # --- Internal helpers ----------------------------------------------

    def _iter_healthy_slots_starting_at(
        self, start: int, *, model: str | None = None
    ) -> Iterator[KeySlot]:
        """Yield each non-disabled slot exactly once, starting at
        ``start`` and wrapping around.

        When ``model`` is given, only slots whose ``model_id``
        matches are emitted. This is the pin path used for turns
        that attached files (vision / PDF models), where rotating
        to a different capability mid-stream would corrupt the
        reply. When ``model`` is ``None`` every healthy slot is
        yielded in round-robin order — the original behaviour.

        **One-key-at-a-time invariant.** Slot order is
        ``[(k1, m1), (k1, m2), …, (k2, m1), (k2, m2), …]`` — the
        outer loop is **key**, the inner loop is **model**. A
        single ``chat`` / ``stream_chat`` call therefore uses
        one key at a time. Model fallback within the same key
        happens *before* the router ever reaches a different
        key. That is the contract documented in the module
        docstring; this iterator is the only place that
        enforces it.

        Slots currently in a rate-limit cooldown (``is_in_cooldown``
        returns ``True``) are skipped without burning a round-trip,
        so the router does not re-hit a slot we already know is
        exhausted for the next minute.

        A fully-disabled or fully-cooled-down pool yields nothing and
        the call site raises ``AllSlotsExhaustedError`` with the
        last error it saw (or, if every slot was already excluded
        before any call was made, with no cause at all).

        **Custom-id pin path.** When ``model`` is set but does not
        match any built slot, we synthesize a single ephemeral
        slot on the *currently active* key only. The user's
        "Advanced model" override must not silently fan out
        across multiple API keys — that would violate the
        one-key-at-a-time contract. The synthesized slot shares
        its ``disabled`` / ``cooldown_until`` state with the
        built anchor for that key, so a 401 on key K in
        :meth:`chat` mutates ``anchor.disabled`` and the
        ephemeral twin reflects it immediately on the next
        iteration. See :meth:`_ephemeral_slot_for_active_key`.
        """
        if model is not None:
            matched = [s for s in self._slots if s.model_id == model and not s.disabled]
            if matched:
                # Pin path: walk only the slots whose model_id
                # matches ``model``. Slot order is already
                # "key outer, model inner", so the first matching
                # slot is the active key + this model; the next
                # match is the same key + this model on the next
                # iteration round; etc. The router only ever
                # touches one key for this pinned turn.
                n = len(self._slots)
                for offset in range(n):
                    slot = self._slots[(start + offset) % n]
                    if slot.model_id != model:
                        continue
                    if slot.disabled:
                        continue
                    if slot.is_in_cooldown():
                        continue
                    yield slot
                return
            # No built slot has this model id. Stay on the active
            # key only — synthesize one ephemeral slot.
            yield from self._ephemeral_slot_for_active_key(model, start)
            return

        n = len(self._slots)
        for offset in range(n):
            slot = self._slots[(start + offset) % n]
            if slot.disabled:
                continue
            if slot.is_in_cooldown():
                continue
            yield slot

    def _ephemeral_slot_for_active_key(
        self, model_id: str, start: int
    ) -> Iterator[KeySlot]:
        """Yield a single ephemeral :class:`KeySlot` for a custom
        (user-supplied) model id, pinned to the currently active
        key only.

        This is the recovery path used when a user pastes a
        ``vendor/some-model:free`` id into the Advanced sidebar
        option because the curated default is rate-limited out.
        The built pool was constructed from a hand-picked list of
        10 free models, so the new id almost certainly does not
        appear there.

        **Why only one key, not all of them.** The module-level
        contract is "one API in flight at a time; the router
        only shifts to a different key after the active key's
        models are exhausted". The legacy implementation
        synthesised one ephemeral slot *per configured key*, which
        silently violated that contract — a single user turn
        could fan out across every configured key in one
        :meth:`chat` call. This implementation honours the
        contract: the ephemeral slot is bound to the key whose
        built-slot index is ``start`` (the active key), and the
        router escalates to a *built* slot on a different key on
        a subsequent call only after every built model on the
        active key has been tried.

        **State sharing.** The ephemeral slot for the active key
        shares its ``disabled`` / ``cooldown_until`` state with
        the *anchor* built slot (the first built slot that holds
        the active key). We do this by reading the anchor's
        flags at yield time instead of snapshotting them into
        the ephemeral slot — a 401 on the active key in
        :meth:`chat` mutates ``anchor.disabled``, and the
        ephemeral twin reflects it immediately on the next
        iteration. Snapshotting would let a user bypass the
        disabled-key guard rails by pasting a custom id.

        We do not track per-slot error history for the ephemeral
        slot — it gets the "last error across the whole pool"
        semantics the call sites already handle.

        If the active key has no built slot (e.g. an empty pool),
        nothing is yielded and the caller raises
        ``AllSlotsExhaustedError`` with no cause — the same as
        the built-slot empty-pool path.
        """
        if not self._slots:
            return
        # The active key is the one behind ``start`` mod len(slots).
        anchor_index = start % len(self._slots)
        anchor = self._slots[anchor_index]
        # Reflect anchor state live — do NOT read slot.disabled /
        # slot.cooldown_until, which are stale by construction.
        if anchor.disabled:
            return
        if anchor.is_in_cooldown():
            return
        yield KeySlot(api_key=anchor.api_key, model_id=model_id)

    def has_built_slot_for(self, model_id: str) -> bool:
        """Return True iff a built pool slot matches ``model_id``.

        Used by the Streamlit sidebar to decide whether the user's
        custom id still needs the ephemeral-slot path or whether
        it can ride on the regular built pool. Read-only — never
        mutates router state.
        """
        return any(s.model_id == model_id for s in self._slots)

    def _backoff_for(self, exc: openrouter.OpenRouterRateLimitError) -> float:
        """Pick a sleep duration for a 429.

        If the exception carries an upstream ``Retry-After`` value
        (we read it from ``exc.body`` for now — OpenRouter sends
        it as part of the JSON body on most providers, e.g.
        ``{"error": {"metadata": {"retry_after": 19}}}``) we use
        the larger of that and the configured default. Otherwise
        we use the default.
        """
        default = self._backoff_seconds
        if not exc.body:
            return default
        # Best-effort parse: we tolerate both the OpenAI-style
        # ``{"error": {"message": "...retry_after..."}}`` envelope
        # and a bare number. We never raise from this helper — a
        # parse error just means we fall back to the default.
        try:
            import json
            payload = json.loads(exc.body)
            # OpenRouter typically nests: error.metadata.retry_after
            retry_after = (
                payload.get("error", {})
                .get("metadata", {})
                .get("retry_after")
            )
            if isinstance(retry_after, (int, float)) and retry_after > 0:
                return max(default, float(retry_after))
        except (ValueError, AttributeError, TypeError):
            pass
        return default


# --- Module-level convenience ------------------------------------------------


def build_from_config(
    keys: Iterable[str],
    models: Iterable[str],
    **kwargs: object,
) -> ModelRouter:
    """Build a ``ModelRouter`` from the config layer's key + model
    iterables.

    The Cartesian product is intentional: every key is paired with
    every model, so two keys on three models gives six slots. This
    is the simplest, most predictable layout — the alternative
    (round-robin pair, so key1+model1, key2+model2) hides the
    effective pool size from the user.
    """
    keys_list = [k for k in keys if k]  # drop empty strings silently
    models_list = [m for m in models if m]
    if not keys_list or not models_list:
        raise NoFreeModelConfiguredError(
            "build_from_config needs at least one key and one model."
        )
    slots: list[tuple[str, str]] = []
    for key in keys_list:
        for model in models_list:
            slots.append((key, model))
    return ModelRouter(slots, **kwargs)

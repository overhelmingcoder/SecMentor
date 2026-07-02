"""Recon orchestrator (Phase 15).

This module is the single dispatch point for the recon subsystem. The
public API is intentionally tiny:

- :class:`ReconReport` — aggregate result. One field per tool plus a
  ``total_ms`` wall-clock and a ``target/display/host/scope_token``
  envelope. The renderer (:mod:`app.recon.report`) and the chat-history
  rehydrate path (:mod:`web.chat_helpers`) both read this exact shape,
  so any rename here is a coordinated edit across three modules.
- :class:`ToolResult` — the per-tool value the orchestrator returns.
  Every transport is wrapped so a tool failure surfaces as
  ``ok=False, error=<str>`` rather than an unhandled exception —
  the report must render even when the WHOIS server is down.
- :func:`run_recon` — the synchronous, total entry point. Validates
  the scope token, normalizes the target, runs every tool, returns a
  fully-populated :class:`ReconReport`.
- :func:`stream_recon` — the generator used by the Streamlit view.
  Yields ``(name, value)`` tuples in *completion* order so the UI can
  render a live progress line per tool, and finishes with a
  ``("report", ReconReport)`` sentinel that the consumer treats as
  the final aggregate.

The orchestrator never raises on a tool failure. The only exceptions
that propagate out of :func:`run_recon` are the pre-dispatch
validations — :class:`ValueError` for an empty target or an unknown
scope token, and :class:`TargetBlockedError` for a target on the
safety blocklist. Anything past that gate is captured into the
``ToolResult.error`` field and rendered as a clean row in the report.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Generator

from .. import config
from . import crt_sh, dns_lookup, ipinfo, urlinfo, whois
from .safety import TargetBlockedError, assert_target_allowed


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolResult:
    """The per-tool value the orchestrator produces.

    ``tool``       : canonical name (``"dns"``, ``"ipinfo"``,
                     ``"urlinfo"``, ``"whois"``, ``"crt_sh"``). The
                     renderer keys off this string; the chat-history
                     rehydrate path also uses it to pick the typed
                     dataclass for ``value``.
    ``ok``         : True iff the transport succeeded. A successful
                     transport with an empty result (e.g. crt.sh
                     returning ``[]``) is still ``ok=True`` — the
                     emptiness is in ``value`` and the renderer knows
                     how to render it as a "no certificates" notice.
    ``value``      : the transport-specific dataclass (``DNSResult``,
                     ``IPInfoResult``, etc.) on success, ``None`` on
                     failure. The renderer ``isinstance``-dispatches
                     on this field.
    ``error``      : human-readable error string on failure, ``None``
                     on success. The crt.sh kill-switch is the one
                     exception: the orchestrator writes
                     :data:`crt_sh.DISABLED_SENTINEL` here and sets
                     ``ok=True`` so the renderer can show the
                     soft-empty "crt.sh disabled" copy.
    ``duration_ms``: wall-clock for this tool only. Used by the
                     Streamlit progress line and the report summary
                     table.
    """

    tool: str
    ok: bool
    value: Any
    error: str | None
    duration_ms: int


@dataclass(frozen=True)
class ReconReport:
    """Aggregate recon report.

    The dataclass is frozen so a report cannot be mutated by a renderer
    bug — the chat-history path serialises a report and then re-hydrates
    it from JSON, and a frozen source means a re-hydrated copy cannot
    bleed into a later call.

    ``target``     : the raw user input. Kept on the report so the
                     audit log can show exactly what was queried.
    ``display``    : the refanged, trimmed user-facing form. The
                     report header shows this; the user recognises the
                     form they typed.
    ``host``       : the normalized hostname the transports queried.
                     This is what the safety rail and the transports
                     actually used.
    ``scope_token``: the engagement scope that authorised this run.
                     Must be in :data:`config._RECON_SCOPE_TOKENS`.
    ``total_ms``   : wall-clock for the whole fan-out, in
                     milliseconds. The summary table in the markdown
                     report renders this as "Total duration".
    ``dns``        : :class:`ToolResult` from :mod:`dns_lookup`.
    ``ipinfo``     : :class:`ToolResult` from :mod:`ipinfo`.
    ``urlinfo``    : :class:`ToolResult` from :mod:`urlinfo`.
    ``whois``      : :class:`ToolResult` from :mod:`whois`.
    ``crt_sh``     : :class:`ToolResult` from :mod:`crt_sh`.
    """

    target: str
    display: str
    host: str
    scope_token: str
    total_ms: int
    dns: ToolResult
    ipinfo: ToolResult
    urlinfo: ToolResult
    whois: ToolResult
    crt_sh: ToolResult

    def tool_results(self) -> list[ToolResult]:
        """Return the per-tool results in the canonical render order.

        Order is fixed: dns, ipinfo, urlinfo, whois, crt_sh. The
        renderer relies on this order for the markdown section order
        and the JSON ``tools`` list order. The chat-history rehydrate
        path uses a ``dict`` keyed by ``tool`` and so is order-agnostic
        — the order here is purely a presentation contract.
        """
        return [self.dns, self.ipinfo, self.urlinfo, self.whois, self.crt_sh]

    def all_ok(self) -> bool:
        """Return True iff every tool in the report succeeded.

        ``all_ok`` is the single check the test suite (and the
        Streamlit status box) uses to decide whether the report is
        "clean" or "partial". A crt.sh kill-switch result counts as
        ``ok`` (the orchestrator sets ``ok=True`` for that branch)
        because the kill-switch is an operator decision, not a
        transport failure.
        """
        return all(r.ok for r in self.tool_results())


# ---------------------------------------------------------------------------
# Scope-token validation
# ---------------------------------------------------------------------------


def _assert_scope_token(scope_token: str) -> str:
    """Return the scope token if it's whitelisted, else raise ``ValueError``.

    The whitelist lives in :mod:`app.config` so the operator has one
    place to extend it. The check is case-sensitive — the slash
    command already lowercases the token before it reaches the
    orchestrator, so a mixed-case token from any other caller surfaces
    as a clean ``ValueError`` rather than silently bypassing the gate.

    Empty and whitespace-only tokens are rejected with a clear message
    that includes the bad value, so a misconfigured caller (or a UI
    bug that drops the token) surfaces as a debuggable error rather
    than a default-allow.
    """
    if not scope_token or not scope_token.strip():
        raise ValueError("recon scope_token must not be empty")
    if scope_token not in config._RECON_SCOPE_TOKENS:
        raise ValueError(
            f"recon scope_token {scope_token!r} is not in the "
            f"allow-list ({sorted(config._RECON_SCOPE_TOKENS)})"
        )
    return scope_token


# ---------------------------------------------------------------------------
# Per-tool wrappers
# ---------------------------------------------------------------------------
#
# Every wrapper has the same shape: take a normalized host, call the
# transport, return a :class:`ToolResult`. The orchestrator's contract
# is "tool failures are part of the report", so the wrappers swallow
# the transport's own exception class and write a populated ``error``
# string into the ToolResult.
#
# Keeping each transport behind its own private function means the
# test suite can patch ``app.recon.orchestrator._run_dns`` (etc.)
# individually to model a flaky tool without mocking the underlying
# ``socket`` / ``urllib`` boundary. The :class:`CrtShOrchestratorDisabledTests`
# suite relies on this — it patches ``_run_crt_sh`` away and
# ``crt_sh.lookup`` to assert the kill-switch is honoured.


def _run_dns(host: str) -> ToolResult:
    """Resolve the host and wrap the result in a ToolResult.

    :func:`dns_lookup.resolve` raises :class:`dns_lookup.DNSError`
    on any failure (NXDOMAIN, no-records, timeout, OSError). The
    wrapper here catches that specific class plus the broad
    defensive ``Exception`` net, times the call, and tags the
    ``tool`` field — the failure surfaces as a populated
    ``ToolResult.error`` rather than as an unhandled exception.
    """
    start = time.perf_counter()
    try:
        result = dns_lookup.resolve(host)
    except dns_lookup.DNSError as exc:
        duration = int((time.perf_counter() - start) * 1000)
        return ToolResult(
            tool="dns", ok=False, value=None,
            error=str(exc), duration_ms=duration,
        )
    except Exception as exc:  # pragma: no cover - defensive
        # The transport is documented as raising only DNSError, so
        # this branch is purely defensive against a future regression
        # (e.g. someone adds a new exception class without catching
        # it here).
        duration = int((time.perf_counter() - start) * 1000)
        return ToolResult(
            tool="dns", ok=False, value=None,
            error=f"dns unexpected: {exc}", duration_ms=duration,
        )
    duration = int((time.perf_counter() - start) * 1000)
    return ToolResult(
        tool="dns", ok=result.ok, value=result,
        error=result.error, duration_ms=duration,
    )


def _run_ipinfo(host: str) -> ToolResult:
    """Look up IP metadata for the first A record of ``host``.

    The IP-info transport requires an IP literal, not a hostname, so
    we first resolve the host (via :mod:`dns_lookup`) and pick the
    first IPv4 — falling back to the first IPv6 if there are no A
    records. If DNS returns nothing, we surface the failure as a
    ToolResult without making a second network call.
    """
    start = time.perf_counter()
    try:
        dns = dns_lookup.resolve(host)
        target_ip: str | None = None
        if dns.ipv4:
            target_ip = dns.ipv4[0]
        elif dns.ipv6:
            target_ip = dns.ipv6[0]
        if not target_ip:
            duration = int((time.perf_counter() - start) * 1000)
            return ToolResult(
                tool="ipinfo", ok=False, value=None,
                error="no IP address to look up (DNS returned nothing)",
                duration_ms=duration,
            )
        info = ipinfo.lookup(target_ip)
    except dns_lookup.DNSError as exc:
        duration = int((time.perf_counter() - start) * 1000)
        return ToolResult(
            tool="ipinfo", ok=False, value=None,
            error=f"ipinfo skipped: {exc}", duration_ms=duration,
        )
    except ipinfo.IPInfoError as exc:
        duration = int((time.perf_counter() - start) * 1000)
        return ToolResult(
            tool="ipinfo", ok=False, value=None,
            error=str(exc), duration_ms=duration,
        )
    except Exception as exc:  # pragma: no cover - defensive
        duration = int((time.perf_counter() - start) * 1000)
        return ToolResult(
            tool="ipinfo", ok=False, value=None,
            error=f"ipinfo unexpected: {exc}", duration_ms=duration,
        )
    duration = int((time.perf_counter() - start) * 1000)
    return ToolResult(
        tool="ipinfo", ok=True, value=info,
        error=None, duration_ms=duration,
    )


def _run_urlinfo(host: str) -> ToolResult:
    """Probe the target URL in-process and wrap the result.

    The probe wants a URL, not a hostname, so we hand it the bare
    host — :func:`urlinfo.probe` prepends ``https://`` if no scheme
    is present. That keeps the recon result consistent with what a
    browser would see when the user types ``example.com`` into the
    address bar.
    """
    start = time.perf_counter()
    try:
        result = urlinfo.probe(host)
    except urlinfo.URLInfoError as exc:
        duration = int((time.perf_counter() - start) * 1000)
        return ToolResult(
            tool="urlinfo", ok=False, value=None,
            error=str(exc), duration_ms=duration,
        )
    except Exception as exc:  # pragma: no cover - defensive
        duration = int((time.perf_counter() - start) * 1000)
        return ToolResult(
            tool="urlinfo", ok=False, value=None,
            error=f"urlinfo unexpected: {exc}", duration_ms=duration,
        )
    duration = int((time.perf_counter() - start) * 1000)
    return ToolResult(
        tool="urlinfo", ok=True, value=result,
        error=None, duration_ms=duration,
    )


def _run_whois(host: str) -> ToolResult:
    """Run the two-hop WHOIS lookup and wrap the result.

    WHOIS requires a domain (no scheme, no path) — :func:`whois.lookup`
    documents that and we just hand it the normalized host, which
    :func:`normalize_target` already stripped down to a bare
    hostname.
    """
    start = time.perf_counter()
    try:
        result = whois.lookup(host)
    except whois.WHOISError as exc:
        duration = int((time.perf_counter() - start) * 1000)
        return ToolResult(
            tool="whois", ok=False, value=None,
            error=str(exc), duration_ms=duration,
        )
    except Exception as exc:  # pragma: no cover - defensive
        duration = int((time.perf_counter() - start) * 1000)
        return ToolResult(
            tool="whois", ok=False, value=None,
            error=f"whois unexpected: {exc}", duration_ms=duration,
        )
    duration = int((time.perf_counter() - start) * 1000)
    return ToolResult(
        tool="whois", ok=True, value=result,
        error=None, duration_ms=duration,
    )


def _run_crt_sh(host: str) -> ToolResult:
    """Certificate-transparency lookup, with kill-switch support.

    The orchestrator honours two env-var knobs:

    - :data:`config.RECON_CRT_SH_ENABLED` (default ``True``): the
      live kill-switch. When ``False`` we never call
      :func:`crt_sh.lookup` — we return a synthetic
      :class:`CrtShResult` from :func:`crt_sh.disabled_result`,
      tagged with :data:`crt_sh.DISABLED_SENTINEL` in the ``error``
      field so the renderer can show the "crt.sh disabled" copy.
    - :data:`config.RECON_FALLBACK_SUBDOMAINS` (default ``()``): an
      operator-curated list that the kill-switch path merges into the
      synthetic result, so the report still has something useful to
      show when crt.sh is offline.

    When the kill-switch is on, the wrapper behaves like every other
    transport: call the lookup, catch :class:`CrtShError`, return a
    populated ToolResult.
    """
    start = time.perf_counter()
    if not config.RECON_CRT_SH_ENABLED:
        # Kill-switch path: no network call, no CrtShError. The
        # synthetic result is shaped exactly like a real
        # ``CrtShResult`` so the renderer's ``isinstance`` check
        # dispatches to the crt.sh code path without a second
        # branch. The sentinel in ``error`` is the only signal the
        # renderer uses to swap the copy.
        synthetic = crt_sh.disabled_result(
            host, fallback_hosts=config.RECON_FALLBACK_SUBDOMAINS,
        )
        duration = int((time.perf_counter() - start) * 1000)
        return ToolResult(
            tool="crt_sh", ok=True, value=synthetic,
            error=crt_sh.DISABLED_SENTINEL, duration_ms=duration,
        )
    try:
        result = crt_sh.lookup(host)
    except crt_sh.CrtShError as exc:
        duration = int((time.perf_counter() - start) * 1000)
        return ToolResult(
            tool="crt_sh", ok=False, value=None,
            error=str(exc), duration_ms=duration,
        )
    except Exception as exc:  # pragma: no cover - defensive
        duration = int((time.perf_counter() - start) * 1000)
        return ToolResult(
            tool="crt_sh", ok=False, value=None,
            error=f"crt_sh unexpected: {exc}", duration_ms=duration,
        )
    duration = int((time.perf_counter() - start) * 1000)
    return ToolResult(
        tool="crt_sh", ok=True, value=result,
        error=None, duration_ms=duration,
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


#: The five per-tool wrappers the orchestrator fans out to, in
#: canonical render order. Defined as a list of (tool-name,
#: attribute-name) tuples so the dispatch loop resolves the
#: callable via :func:`getattr` *at submit time* — this indirection
#: is what makes the test suite's
#: ``mock.patch.multiple("app.recon.orchestrator", _run_dns=...)``
#: patches actually take effect. If the list held the callables
#: directly, the executor would have captured the *original*
#: references at import time and the patches would silently
#: no-op. Resolving via ``getattr(sys.modules[__name__], name)`` at
#: call time is the standard Python idiom for "patchable
#: dispatch" and is the only reason the test suite can inject
#: fake ``ToolResult`` objects without monkey-patching the executor.
_TOOL_DISPATCH: list[tuple[str, str]] = [
    ("dns", "_run_dns"),
    ("ipinfo", "_run_ipinfo"),
    ("urlinfo", "_run_urlinfo"),
    ("whois", "_run_whois"),
    ("crt_sh", "_run_crt_sh"),
]


def _resolve_tool(attr_name: str) -> Any:
    """Look up a per-tool wrapper by its module attribute name.

    Resolved at dispatch time (not import time) so test patches
    on ``app.recon.orchestrator._run_<x>`` are honoured. The
    fallback to a direct local lookup covers the case where a
    caller imported the module and is dispatching outside the
    standard entry points.
    """
    return globals()[attr_name]


def _build_report(
    target: str,
    display: str,
    host: str,
    scope_token: str,
    results: dict[str, ToolResult],
    total_ms: int,
) -> ReconReport:
    """Assemble a :class:`ReconReport` from a name-keyed result map.

    The streaming loop builds a ``dict`` because tools complete in
    non-deterministic order, but the dataclass needs them as named
    fields. This helper is the single place that translates between
    the two shapes.
    """
    return ReconReport(
        target=target,
        display=display,
        host=host,
        scope_token=scope_token,
        total_ms=total_ms,
        dns=results["dns"],
        ipinfo=results["ipinfo"],
        urlinfo=results["urlinfo"],
        whois=results["whois"],
        crt_sh=results["crt_sh"],
    )


def run_recon(target: str, *, scope_token: str) -> ReconReport:
    """Synchronous entry point — runs every tool, returns the report.

    The function is total with respect to the transports: any tool
    failure surfaces as ``ok=False`` on its :class:`ToolResult`, not
    as an exception here. The two exceptions that *do* propagate are
    the pre-dispatch gates:

    - :class:`ValueError` for an empty target or an unknown scope
      token. The :mod:`web.chat_helpers` slash-command parser already
      fills both in correctly, so reaching this branch means a
      programmatic caller passed garbage.
    - :class:`TargetBlockedError` for a host on the safety
      blocklist (loopback, RFC1918, internal TLDs, etc.). The
      Streamlit view catches this *before* dispatching to
      :func:`run_recon`, so it should never fire from the UI — but
      we honour the blocklist here as well, so a CLI caller or a
      test can never bypass the safety rail by going around the view.

    The five tools are run **in parallel** with a
    :class:`ThreadPoolExecutor`. The orchestrator's wall-clock is the
    span from the first dispatch to the last completion, which is
    noticeably shorter than the sum of the per-tool durations when
    one tool is slow (crt.sh). The ThreadPoolExecutor is a private
    implementation detail — the public contract is "we run all five
    tools".
    """
    if not target or not target.strip():
        raise ValueError("recon target must not be empty")
    _assert_scope_token(scope_token)
    # ``assert_target_allowed`` both validates the host and returns a
    # cleaned form. We feed it the trimmed target — the normalize
    # module's own ``normalize_target`` is a richer pipeline (handles
    # defang + IDN), but the safety rail is the gate the test suite
    # pins, so we go through the same surface the Streamlit view
    # uses.
    safe_host = assert_target_allowed(target.strip())

    # Capture the user-facing display form. We keep the original
    # ``target`` for the audit log; the refanged form is what the
    # report header shows.
    from .normalize import refang
    display = refang(target).strip() or target

    start = time.perf_counter()
    results: dict[str, ToolResult] = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_tool = {
            executor.submit(_resolve_tool(attr), safe_host): name
            for name, attr in _TOOL_DISPATCH
        }
        for future in as_completed(future_to_tool):
            name = future_to_tool[future]
            try:
                results[name] = future.result()
            except Exception as exc:  # pragma: no cover - defensive
                # The per-tool wrappers are designed to be total; this
                # branch only fires on a bug in a wrapper. We still
                # want a populated ToolResult in the report rather
                # than a crash, so we synthesise an error row.
                results[name] = ToolResult(
                    tool=name, ok=False, value=None,
                    error=f"{name} unexpected: {exc}", duration_ms=0,
                )
    total_ms = int((time.perf_counter() - start) * 1000)
    return _build_report(
        target=target,
        display=display,
        host=safe_host,
        scope_token=scope_token,
        results=results,
        total_ms=total_ms,
    )


def stream_recon(
    target: str, *, scope_token: str,
) -> Generator[tuple[str, Any], None, None]:
    """Streaming entry point — yields per-tool results as they finish.

    The contract is: yield one ``(tool_name, ToolResult)`` tuple per
    tool in *completion* order (not dispatch order), then yield a
    final ``("report", ReconReport)`` sentinel that the consumer
    treats as the aggregate. The Streamlit view uses the per-tool
    yields to render a live progress line and the final sentinel to
    render the markdown report.

    The pre-dispatch gates (empty target, unknown scope token,
    blocked target) raise the same exceptions :func:`run_recon`
    raises — the consumer catches :class:`TargetBlockedError` and
    :class:`ValueError` *before* iterating the generator, so the
    generator itself does not need to re-validate. We re-validate
    anyway as a defensive measure (CLI callers may not have run the
    same pre-checks), and so the failure mode is consistent between
    :func:`run_recon` and :func:`stream_recon`.
    """
    if not target or not target.strip():
        raise ValueError("recon target must not be empty")
    _assert_scope_token(scope_token)
    safe_host = assert_target_allowed(target.strip())

    from .normalize import refang
    display = refang(target).strip() or target

    start = time.perf_counter()
    results: dict[str, ToolResult] = {}
    # The order we yield per-tool results in is completion order, so
    # the executor returns futures that complete in arbitrary order.
    # We collect them in a dict and yield in the order ``as_completed``
    # produces.
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_tool = {
            executor.submit(_resolve_tool(attr), safe_host): name
            for name, attr in _TOOL_DISPATCH
        }
        for future in as_completed(future_to_tool):
            name = future_to_tool[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - defensive
                result = ToolResult(
                    tool=name, ok=False, value=None,
                    error=f"{name} unexpected: {exc}", duration_ms=0,
                )
            results[name] = result
            yield (name, result)
    total_ms = int((time.perf_counter() - start) * 1000)
    report = _build_report(
        target=target,
        display=display,
        host=safe_host,
        scope_token=scope_token,
        results=results,
        total_ms=total_ms,
    )
    yield ("report", report)

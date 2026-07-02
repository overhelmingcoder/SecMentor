"""crt.sh certificate-transparency lookup for the recon subsystem.

Phase 15 — query the crt.sh JSON API for every certificate ever
issued for a given host. Certificate transparency logs are a gold
mine for subdomain enumeration: a certificate is published the
moment it is issued, so the list is a near-complete picture of
the host's surface area.

Endpoint
--------

GET ``https://crt.sh/?q=<host>&output=json``

The response is a JSON array of objects, one per certificate. The
fields we care about are:

- ``common_name``        : the CN (often includes wildcards like
                           ``*.example.com``).
- ``name_value``         : the SAN list, newline-separated. This
                           is the field that reveals subdomains.
- ``issuer_name``        : the issuing CA.
- ``not_before``         : ISO 8601 issuance date.
- ``not_after``          : ISO 8601 expiry date.

We deduplicate the SAN list across all returned certificates
because crt.sh returns one row per certificate, and the same
subdomain shows up in every renewal. The deduplication is on the
fully-qualified, lowercased hostname.

Failure modes
-------------

- crt.sh is a hobby project; it has long cold-start latencies
  (sometimes 30s+) and occasionally returns 503. The orchestrator
  gives it a generous timeout (the shared
  :data:`app.config.RECON_HTTP_TIMEOUT_SECONDS`).
- A non-200 from crt.sh is a hard error; a non-200 from the
  *target* is not relevant here (we never touch the target).
- A 200 with an empty array is not an error — it just means crt.sh
  has no record of the host. The result is an empty ``hosts`` set.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Final, Iterable

from .. import config

_BASE_URL: Final[str] = "https://crt.sh"
#: Per-tool timeout for the crt.sh HTTP query. Defaults to the
#: shared ``RECON_HTTP_TIMEOUT_SECONDS`` (15s) but can be raised
#: via ``RECON_CRT_SH_TIMEOUT_SECONDS`` in the operator's ``.env``
#: when crt.sh is responding slowly — it's a hobby project with
#: known cold-start spikes.
_TIMEOUT: Final[float] = config.RECON_CRT_SH_TIMEOUT_SECONDS
_USER_AGENT: Final[str] = "SecMentor-Stage1-Recon/0.1 (+local)"
#: crt.sh is a hobby project with documented transient failures
#: (502/503/timeout on cold queries). The operator's kill-switch
#: ``RECON_CRT_SH_ENABLED`` exists for outages long enough that
#: retrying is futile, but for the short blips a single retry with
#: a small backoff clears the failure without forcing the operator
#: to flip the switch. Two attempts (one retry) at 1s backoff keeps
#: the total worst-case latency at ``_TIMEOUT + 1s + _TIMEOUT`` —
#: well under the orchestrator's 60s budget and short enough that
#: crt.sh being genuinely broken surfaces as a ``CrtShError``
#: promptly instead of stalling the recon turn. The crt.sh project
#: itself recommends "retry, we get busy" in its FAQ; one retry is
#: enough to clear the common cold-query blip without paying the
#: latency tax twice for genuine outages.
_MAX_ATTEMPTS: Final[int] = 2
_BACKOFF_SECONDS: Final[tuple[float, ...]] = (1.0,)

#: Sentinel value the orchestrator writes into
#: :attr:`ToolResult.error` when :data:`app.config.RECON_CRT_SH_ENABLED`
#: is ``False``. The renderer checks for this exact string to decide
#: whether to render the "crt.sh disabled" copy versus the
#: "crt.sh returned no certificates" copy. Centralising the string
#: here means the orchestrator and renderer never drift apart, and
#: tests can assert on the canonical value.
DISABLED_SENTINEL: Final[str] = (
    "crt.sh disabled (set RECON_CRT_SH_ENABLED=true in .env to re-enable)"
)

#: The "q" parameter on crt.sh accepts a domain *or* a literal with
#: a leading ``%`` (e.g. ``%.example.com``) to match subdomains. We
#: always send the bare host — crt.sh's default match is
#: "contains", which catches the subdomains anyway. If we want
#: strict-match semantics later, the operator can add the ``%``
#: in :mod:`app.recon.orchestrator` — out of scope for Phase 15.


class CrtShError(RuntimeError):
    """Raised when the crt.sh query fails.

    The error message includes the host and the underlying
    exception class so the operator can tell apart a timeout from
    a JSON parse failure from a non-200 status.
    """


@dataclass(frozen=True)
class CrtShResult:
    """Parsed crt.sh record.

    ``hosts``     : sorted, deduplicated, lowercased list of FQDNs
                   seen across all returned certificates (wildcard
                   entries like ``*.example.com`` are preserved —
                   the wildcard is meaningful signal in CT logs and
                   the renderer shows it). This is the only field
                   the report cares about for Phase 15 — the rest
                   is audit-trail detail.
    ``cert_count``: number of certificates crt.sh returned (NOT
                   the host count — one host typically has many
                   certs).
    ``issuers``   : sorted, deduplicated list of issuing CAs.
    ``raw``       : the first N rows of the JSON array, kept for
                   the audit log so an operator can re-derive
                   fields this dataclass does not enumerate.
                   Capped at :data:`_RAW_KEEP` to bound memory.
    """

    hosts: list[str]
    cert_count: int
    issuers: list[str]
    raw: list[dict[str, Any]] = field(default_factory=list)

#: Cap on the raw rows we keep. crt.sh can return thousands of
#: rows for a popular domain; we only need a sample for the audit
#: log. 100 is enough to spot the issuing CA and the first SANs.
_RAW_KEEP: Final[int] = 100


def _coerce_str(value: Any) -> str:
    """Coerce a JSON value to ``str`` with a sane empty-string fallback.

    crt.sh returns ``None`` for missing string fields; everything
    else is already a string or a list of strings.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _extract_hosts(rows: Iterable[dict[str, Any]]) -> set[str]:
    """Union of all hostnames mentioned in a list of crt.sh rows.

    The ``name_value`` field is newline-separated SANs and may
    contain bare hostnames, wildcard labels (``*.example.com``),
    and the CN repeated. We **preserve** the leading ``*.`` because
    wildcards are meaningful signal in certificate-transparency
    logs — the renderer shows them and a security review of the
    report would want to know ``*.example.com`` was issued. We
    only lowercase + strip whitespace + dedupe via the set.

    The ``common_name`` field is more dangerous: CAs sometimes put a
    human-readable label there (e.g. ``"as207960 test intermediate -
    example.com"`` for an intermediate certificate's CN) rather than
    a hostname. We require the cleaned entry to *not* contain a space
    — a real hostname never has spaces, and this single-character
    filter removes the CA-label false positives without rejecting
    any legitimate subdomain.
    """
    hosts: set[str] = set()
    for row in rows:
        for field_name in ("name_value", "common_name"):
            raw = _coerce_str(row.get(field_name))
            if not raw:
                continue
            for entry in raw.splitlines():
                h = entry.strip().lower()
                # Reject empty, whitespace-bearing (CA labels), and
                # entries with no dot (would be invalid FQDNs).
                # Wildcard labels (``*.example.com``) are kept as-is.
                if not h or " " in h or "." not in h:
                    continue
                hosts.add(h)
    return hosts


def _extract_issuers(rows: Iterable[dict[str, Any]]) -> set[str]:
    """Set of unique issuer names across all rows.

    Empty / non-string values are skipped; order is not preserved
    here — the dataclass sorts at the boundary.
    """
    issuers: set[str] = set()
    for row in rows:
        issuer = _coerce_str(row.get("issuer_name")).strip()
        if issuer:
            issuers.add(issuer)
    return issuers


def disabled_result(
    host: str, *, fallback_hosts: Iterable[str] = ()
) -> CrtShResult:
    """Build a synthetic :class:`CrtShResult` for the "crt.sh disabled" path.

    When :data:`app.config.RECON_CRT_SH_ENABLED` is ``False`` the
    orchestrator calls this helper instead of :func:`lookup`. No HTTP
    call is made, no :class:`CrtShError` is raised, and the report
    renders cleanly — ``ok=True``, ``cert_count=0``, an empty
    issuers list, and a hosts tuple built from
    :data:`app.config.RECON_FALLBACK_SUBDOMAINS` (passed through as
    ``fallback_hosts`` so the orchestrator can override the value at
    runtime, e.g. in tests).

    The :class:`CrtShResult` is the same dataclass the live
    :func:`lookup` returns, so the renderer's type discrimination
    (``isinstance(result.value, CrtShResult)``) does not need a
    second branch — the renderer just sees "no certs" and shows a
    single soft copy line instead of three ``_(none)_`` rows.

    Args:
        host: The host the orchestrator was about to query. Kept on
            the function signature for parity with :func:`lookup`
            and for future logging; the returned result is *not*
            tagged with the host — the report already has the host
            in its target/host section.
        fallback_hosts: Optional iterable of hostnames that should
            appear in the ``hosts`` field. Whitespace, empty
            entries, and duplicates are stripped, and the surviving
            entries are lowercased and sorted so the result is
            deterministic.

    Returns:
        A :class:`CrtShResult` with ``cert_count=0``,
        ``issuers=[]``, ``hosts=sorted(set(...))``, and an empty
        ``raw`` list.
    """
    seen: set[str] = set()
    cleaned: list[str] = []
    for entry in fallback_hosts:
        if not isinstance(entry, str):
            continue
        h = entry.strip().lower()
        if not h or "." not in h or h in seen:
            continue
        seen.add(h)
        cleaned.append(h)
    return CrtShResult(
        hosts=sorted(cleaned),
        cert_count=0,
        issuers=[],
        raw=[],
    )


def lookup(host: str) -> CrtShResult:
    """Query crt.sh for every certificate ever issued for ``host``.

    The argument is the *hostname*, not a URL. crt.sh does not
    understand paths, ports, or schemes. The function is total:
    any failure raises :class:`CrtShError` and the orchestrator
    records the tool as ``status="error"``.
    """
    if not host or not host.strip():
        raise CrtShError("crt.sh lookup requires a non-empty host")
    target = host.strip()
    # crt.sh's default match is "contains" — a query for
    # "example.com" also returns rows for "*.example.com" and for
    # subdomains. We keep that behaviour.
    url = f"{_BASE_URL}/?q={urllib.parse.quote(target)}&output=json"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    body: bytes | None = None
    last_error: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                status = getattr(resp, "status", None) or resp.getcode()
                if status != 200:
                    raise CrtShError(
                        f"crt.sh returned HTTP {status} for {target}"
                    )
                body = resp.read()
            last_error = None
            break
        except urllib.error.HTTPError as exc:
            # 502 / 503 / 504 are transient — retry. Anything else
            # (404, 400) is a client error and retrying won't help.
            if exc.code in {502, 503, 504} and attempt + 1 < _MAX_ATTEMPTS:
                last_error = exc
                time.sleep(_BACKOFF_SECONDS[attempt])
                continue
            raise CrtShError(
                f"crt.sh HTTPError {exc.code} for {target}: {exc.reason}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            # URLError covers DNS/connect failures, TimeoutError fires
            # when the socket read exceeds _TIMEOUT, OSError covers the
            # rest. All three are transient on crt.sh's infrastructure.
            if attempt + 1 < _MAX_ATTEMPTS:
                last_error = exc
                time.sleep(_BACKOFF_SECONDS[attempt])
                continue
            raise CrtShError(
                f"crt.sh {type(exc).__name__} for {target}: {exc}"
            ) from exc
    if body is None:
        # Defensive: every branch above either sets body or raises.
        # If we got here the loop exited with no body and no exception
        # — that should be impossible, but fail loudly rather than
        # silently produce an empty report.
        raise CrtShError(
            f"crt.sh lookup exhausted {_MAX_ATTEMPTS} attempts for "
            f"{target}: {type(last_error).__name__ if last_error else 'unknown'}"
        )
    try:
        payload = json.loads(body.decode("utf-8", errors="replace"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise CrtShError(
            f"crt.sh returned non-JSON body for {target}: {exc}"
        ) from exc
    if not isinstance(payload, list):
        raise CrtShError(
            f"crt.sh returned non-array JSON for {target}: "
            f"type {type(payload).__name__}"
        )
    # crt.sh returns rows as dicts; tolerate any non-dict items by
    # dropping them (rather than failing the whole call) — the
    # API is stable but a future version may add a metadata block.
    rows = [r for r in payload if isinstance(r, dict)]
    hosts = _extract_hosts(rows)
    issuers = _extract_issuers(rows)
    return CrtShResult(
        hosts=sorted(hosts),
        cert_count=len(rows),
        issuers=sorted(issuers),
        raw=rows[:_RAW_KEEP],
    )
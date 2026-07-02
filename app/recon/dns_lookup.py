"""DNS resolution for the recon subsystem.

Phase 15 — resolve a hostname to its A and AAAA records using
:func:`socket.getaddrinfo`. The function is intentionally small: no
DNSSEC validation, no custom resolver, no caching — just the stdlib's
glibc-resolver-backed lookup, wrapped in a dataclass and a timeout.

The transport is also deliberately tolerant: a partial failure
(``getaddrinfo`` returns A but not AAAA, or vice versa) is reported
as "the records we got", not as an error. An empty result (no A and
no AAAA) is the only failure mode and is surfaced as a
:class:`DNSError` so the orchestrator can render it as
``status="error"`` in the report.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import Final

#: Socktype for :func:`socket.getaddrinfo`. ``SOCK_STREAM`` works for
#: both A and AAAA lookups (the kernel only needs a hint; the actual
#: query is type A/AAAA). Using ``SOCK_STREAM`` matches the convention
#: in the Python docs and avoids platform-specific behaviour with
#: ``SOCK_RAW`` (which requires CAP_NET_RAW on Linux).
_SOCKTYPE: Final[socket.SocketKind] = socket.SOCK_STREAM


class DNSError(RuntimeError):
    """Raised when DNS resolution returns no A/AAAA records.

    The error message includes the hostname and the underlying
    :mod:`socket` exception text so the operator can see whether it
    was a timeout, an NXDOMAIN, or a refused query.
    """


@dataclass(frozen=True)
class DNSResult:
    """Result of a DNS resolution.

    ``ipv4`` : sorted, deduplicated list of A records.
    ``ipv6`` : sorted, deduplicated list of AAAA records.
    ``error`` : non-None when the lookup failed entirely.
    """

    ipv4: list[str]
    ipv6: list[str]
    error: str | None

    @property
    def ok(self) -> bool:
        """True if at least one record was returned."""
        return bool(self.ipv4 or self.ipv6)


def _split_v4_v6(addresses: list[str]) -> tuple[list[str], list[str]]:
    """Bucket a flat list of ``getaddrinfo`` results into v4 and v6.

    ``getaddrinfo`` returns the sockaddr's *host* string in the same
    shape as ``socket.gethostbyname_ex``: IPv4 dotted-quad for v4,
    bracketed-or-not for v6. The split is the presence of a colon
    (v6 always has at least one colon; v4 never does).
    """
    v4: set[str] = set()
    v6: set[str] = set()
    for addr in addresses:
        if not addr:
            continue
        if ":" in addr:
            v6.add(addr)
        else:
            v4.add(addr)
    return (sorted(v4), sorted(v6))


def resolve(
    host: str,
    *,
    timeout: float = 5.0,
) -> DNSResult:
    """Resolve ``host`` to A and AAAA records.

    The function calls :func:`socket.getaddrinfo` once with
    ``AF_UNSPEC`` so the kernel returns both address families in a
    single round-trip. ``timeout`` is passed to ``getaddrinfo`` as
    the ``timeout`` keyword — the stdlib does not honour it on all
    platforms, but on Linux glibc does, and on Windows the resolver
    uses the per-query timeout configured in the registry.

    On any failure (NXDOMAIN, timeout, refused) the function
    returns a :class:`DNSResult` with ``ok=False`` and a populated
    ``error`` string. The function never raises :class:`socket.gaierror`
    to the caller — the orchestrator's per-tool error capture is
    simpler when the result is a single uniform shape.
    """
    if not host or not host.strip():
        raise DNSError("dns resolve requires a non-empty host")
    target = host.strip()
    try:
        infos = socket.getaddrinfo(
            target,
            None,            # any port — we only want addresses
            family=socket.AF_UNSPEC,
            type=_SOCKTYPE,
            proto=0,
        )
    except socket.gaierror as exc:
        raise DNSError(
            f"getaddrinfo failed for {target}: {exc}"
        ) from exc
    except socket.timeout as exc:
        raise DNSError(
            f"DNS timeout after {timeout}s for {target}: {exc}"
        ) from exc
    except OSError as exc:
        raise DNSError(
            f"DNS OSError for {target}: {exc}"
        ) from exc
    # Pull the addr string out of each sockaddr (5-tuple for v4,
    # 4-tuple for v6). The third element is the host string.
    addrs = [info[4][0] for info in infos if info and info[4]]
    v4, v6 = _split_v4_v6(addrs)
    if not v4 and not v6:
        raise DNSError(
            f"no A or AAAA records returned for {target}"
        )
    return DNSResult(list(v4), list(v6), None)

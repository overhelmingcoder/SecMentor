"""Target safety rail for the recon subsystem.

Phase 15 — every recon call MUST pass through :func:`assert_target_allowed`
before any transport is invoked. The check is deliberately
*pessimistic*: when in doubt, block. The point of the recon feature
is to gather public OSINT, not to attack infrastructure, and the
reputation cost of an unauthorized scan is permanent.

Three classes of target are rejected:

1. **IP literals** that are private, loopback, link-local, multicast,
   or otherwise non-routable on the public Internet.
   - ``0.0.0.0/8``      — "this network" (RFC 1122)
   - ``10.0.0.0/8``     — RFC 1918 private
   - ``100.64.0.0/10``  — CGNAT (RFC 6598)
   - ``127.0.0.0/8``    — loopback
   - ``169.254.0.0/16`` — link-local
   - ``172.16.0.0/12``  — RFC 1918 private
   - ``192.0.0.0/24``   — IETF protocol assignments (RFC 6890)
   - ``192.168.0.0/16`` — RFC 1918 private
   - ``198.18.0.0/15``  — benchmarking (RFC 2544)
   - ``198.51.100.0/24``— TEST-NET-2 (RFC 5737)
   - ``203.0.113.0/24`` — TEST-NET-3 (RFC 5737)
   - ``224.0.0.0/4``    — multicast
   - ``240.0.0.0/4``    — reserved
   - ``::1/128``        — IPv6 loopback
   - ``fc00::/7``       — IPv6 ULA
   - ``fe80::/10``      — IPv6 link-local

2. **Hostnames** that end in an internal TLD or a non-DNS suffix:
   - ``.local`` (mDNS), ``.internal``, ``.lan``, ``.intranet``,
     ``.corp``, ``.home``, ``.private`` — RFC 6762 + common corp use
   - ``.arpa`` — reverse DNS, never a forward target
   - ``.onion`` — Tor hidden service; out of scope for this tool
   - ``.localhost`` — RFC 6761 reserved

3. **Explicitly-typed private literals** that didn't make it through
   the IP check (e.g. an IPv6 with a zone, a private range in
   non-canonical form). We delegate that to the IP check above.

The function is pure and synchronous — no I/O, no DNS, no
configuration. It only reads its arguments. The orchestrator calls
it once per dispatch and treats any raised :class:`TargetBlockedError`
as a hard failure that is logged with ``status="blocked"``.
"""

from __future__ import annotations

import ipaddress
from typing import Final

# --- Internal TLDs / non-DNS suffixes ---------------------------------------
# Lowercase, with the leading dot. A hostname matches if its last
# label(s) equal one of these. We are conservative: any suffix that
# is *commonly* used inside a private network is blocked, even if it
# is technically routable (e.g. ".local" is mDNS and not on the
# public DNS).
_INTERNAL_TLDS: Final[frozenset[str]] = frozenset({
    ".local",
    ".localhost",
    ".internal",
    ".intranet",
    ".lan",
    ".corp",
    ".home",
    ".private",
    ".arpa",
    ".onion",
})

#: Hostnames that are reserved by the IANA / IETF and must not be
#: scanned regardless of suffix match. Lowercase.
_RESERVED_NAMES: Final[frozenset[str]] = frozenset({
    "localhost",
    "ip6-localhost",
    "ip6-loopback",
})


class TargetBlockedError(ValueError):
    """Raised when a recon target is on the blocklist.

    Inherits from ``ValueError`` so a single ``except ValueError``
    catches it alongside the empty-target / unparseable-target
    cases in :mod:`app.recon.normalize`. The ``reason`` attribute
    is the human-readable explanation; the audit log stores it as
    the ``result_excerpt``.
    """

    def __init__(self, target: str, reason: str) -> None:
        super().__init__(f"recon target blocked: {target} ({reason})")
        self.target = target
        self.reason = reason


def _is_blocked_ip(host: str) -> str | None:
    """Return a reason string if the host is a blocked IP literal, else None.

    Uses :mod:`ipaddress` so we get the full RFC 6890 / 5735 / 6598
    coverage for free. The function accepts both IPv4 and IPv6
    literals; a hostname (no dots-as-numbers) is reported as
    not-an-IP and the caller moves on to the suffix check.
    """
    # Strip an IPv6 zone if present ("fe80::1%eth0" → "fe80::1").
    bare = host.split("%", 1)[0]
    try:
        ip = ipaddress.ip_address(bare)
    except ValueError:
        return None
    # ``is_global`` is True only for IPs that are routable on the
    # public Internet. Everything else — private, loopback, link-local,
    # multicast, reserved, unspecified — is False. That single check
    # covers every blocklist entry above except the TEST-NET ranges
    # and the benchmarking range, which ``is_global`` also flags.
    if not ip.is_global:
        return f"IP literal {ip} is not globally routable"
    return None


def _is_internal_suffix(host: str) -> str | None:
    """Return a reason string if the host ends in a blocked TLD, else None.

    Matches against :data:`_INTERNAL_TLDS` (with leading dot) and
    :data:`_RESERVED_NAMES` (exact match). The comparison is
    case-insensitive — the caller is expected to have already
    lowercased the host.
    """
    if host in _RESERVED_NAMES:
        return f"reserved hostname {host!r}"
    for suffix in _INTERNAL_TLDS:
        if host.endswith(suffix):
            return f"hostname ends in reserved suffix {suffix!r}"
    return None


def assert_target_allowed(target: str) -> str:
    """Validate a normalized target. Returns the host on success.

    The argument is expected to be the *normalized* host (lowercase,
    no scheme, no path) as produced by :func:`app.recon.normalize.normalize_target`.
    A non-normalized input still works, but the check is more
    reliable on the normalized form.

    The return value is the host string, normalized once more, so
    callers can do ``host = assert_target_allowed(t)`` and not have
    to re-normalize.
    """
    if not target or not target.strip():
        raise TargetBlockedError(str(target), "empty target")
    host = target.strip().lower().rstrip(".")
    if not host:
        raise TargetBlockedError(target, "empty after normalization")
    if reason := _is_blocked_ip(host):
        raise TargetBlockedError(host, reason)
    if reason := _is_internal_suffix(host):
        raise TargetBlockedError(host, reason)
    return host

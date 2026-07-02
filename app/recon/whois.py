"""WHOIS lookup for the recon subsystem.

Phase 15 — speak WHOIS directly over TCP/43. We do not use the
``python-whois`` PyPI package because it is unmaintained and pulls
in a heavy dependency tree; the WHOIS protocol is a single line
of text out and a multi-line response back, which fits in ~50
lines of stdlib.

Two-step lookup
---------------

1. Query ``whois.iana.org:43`` for the domain. The response contains
   a ``refer:`` line pointing to the *registrar* WHOIS server
   (e.g. ``whois.markmonitor.com``).
2. Open a second TCP connection to that registrar and ask the same
   question. The registrar's response has the registration record.

We do NOT do a third step to chase down the *registrant* contact
(``abusecompose@...`` etc.) — that requires RDAP, which is a
different protocol and out of scope for Phase 15.

Failure modes
-------------

- Connection refused / timeout / DNS error: :class:`WHOISError`.
- IANA returns no ``refer:`` line (rare — some TLDs delegate to a
  whois server via a different mechanism): the function returns
  the raw IANA text as ``body`` and leaves ``registrar_server=None``.
- Registrar returns a referral of its own (some country-code TLDs
  nest two or three levels deep): we stop after the first hop and
  record the second-level referral in ``body`` for the operator.
"""

from __future__ import annotations

import re
import socket
from dataclasses import dataclass
from typing import Final

#: Default WHOIS server for the first hop. IANA is the canonical
#: bootstrap — every TLD publishes its registrar there.
_ANA_SERVER: Final[str] = "whois.iana.org"
_ANA_PORT: Final[int] = 43

#: Per-hop timeout in seconds. WHOIS servers are usually fast; a
#: 10-second ceiling is generous. We pick 10s rather than the
#: shared HTTP timeout because a hung WHOIS server is rarer than
#: a slow web page and a tighter ceiling speeds up the report.
_TIMEOUT: Final[float] = 10.0

#: Read buffer size for the WHOIS response. WHOIS responses are
#: small (1-20 KB) so a 4 KB buffer is plenty; the loop reads
#: until EOF so the size only matters for syscall count.
_RECV_BUF: Final[int] = 4096

#: Pattern for the ``refer:`` line in an IANA response. The line is
#: always lower-case ``refer:`` followed by whitespace and the
#: hostname of the registrar WHOIS server. Case-insensitive match
#: because some older IANA mirrors use ``Refer:``.
_REFER_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*refer\s*:\s*([^\s]+)",
    re.IGNORECASE | re.MULTILINE,
)


class WHOISError(RuntimeError):
    """Raised when a WHOIS query fails.

    The error message includes the server that failed and the
    underlying exception class so the operator can tell apart a
    connection refused from a timeout from a DNS error.
    """


@dataclass(frozen=True)
class WHOISResult:
    """Parsed WHOIS record.

    ``iana_body``       : raw text returned by the IANA bootstrap
                          server (kept for the audit log and for
                          fields the parser does not enumerate).
    ``registrar_server``: hostname of the registrar WHOIS server,
                          or None if the IANA response had no
                          ``refer:`` line.
    ``registrar_body``  : raw text returned by the registrar, or
                          None if no second hop was performed.
    """

    iana_body: str
    registrar_server: str | None
    registrar_body: str | None

    @property
    def body(self) -> str:
        """The most informative body available — registrar first, IANA fallback."""
        return self.registrar_body or self.iana_body


def _query(server: str, port: int, query: str) -> str:
    """Open a TCP connection, send ``query\\r\\n``, return the response body.

    The function is the single network primitive for the WHOIS
    module. It is private because every caller wraps it in a
    :class:`WHOISError` with a server-specific message; the bare
    ``OSError`` is not useful at the orchestrator level.
    """
    with socket.create_connection((server, port), timeout=_TIMEOUT) as sock:
        sock.sendall(f"{query}\r\n".encode("ascii", errors="replace"))
        chunks: list[bytes] = []
        while True:
            buf = sock.recv(_RECV_BUF)
            if not buf:
                break
            chunks.append(buf)
    return b"".join(chunks).decode("utf-8", errors="replace")


def _extract_refer(body: str) -> str | None:
    """Pull the registrar hostname out of an IANA response.

    Returns ``None`` if no ``refer:`` line is found. The match is
    case-insensitive and tolerates trailing whitespace.
    """
    match = _REFER_RE.search(body)
    if not match:
        return None
    host = match.group(1).strip().lower()
    return host or None


def lookup(domain: str) -> WHOISResult:
    """Run the two-hop WHOIS lookup for ``domain``.

    The argument is the *domain*, not a URL — WHOIS does not
    understand paths, ports, or schemes. The caller is responsible
    for handing us the post-normalize host (e.g. ``example.com``,
    not ``https://example.com/foo``).
    """
    if not domain or not domain.strip():
        raise WHOISError("whois lookup requires a non-empty domain")
    target = domain.strip().lower()
    try:
        iana_body = _query(_ANA_SERVER, _ANA_PORT, target)
    except socket.gaierror as exc:
        raise WHOISError(
            f"whois IANA DNS error for {target}: {exc}"
        ) from exc
    except socket.timeout as exc:
        raise WHOISError(
            f"whois IANA timeout after {_TIMEOUT}s for {target}"
        ) from exc
    except OSError as exc:
        raise WHOISError(
            f"whois IANA connection error for {target}: {exc}"
        ) from exc
    registrar_server = _extract_refer(iana_body)
    registrar_body: str | None = None
    if registrar_server:
        try:
            registrar_body = _query(registrar_server, _ANA_PORT, target)
        except (socket.gaierror, socket.timeout, OSError) as exc:
            # The IANA hop succeeded — that's the most useful
            # result. A failed registrar hop is recorded in
            # ``registrar_body`` as None and the function returns
            # rather than raising, so the orchestrator can still
            # render the IANA body in the report.
            registrar_body = None
            # We deliberately swallow the exception; the error
            # string would be lost otherwise. Stash it in a
            # structured comment line that the renderer can
            # surface if it wants.
            iana_body = (
                iana_body
                + f"\n\n# registrar hop to {registrar_server} failed: {exc}\n"
            )
    return WHOISResult(
        iana_body=iana_body,
        registrar_server=registrar_server,
        registrar_body=registrar_body,
    )
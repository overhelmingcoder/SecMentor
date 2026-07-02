"""Target normalization for the recon subsystem.

Phase 15 — every recon call goes through :func:`normalize_target` first,
so the downstream transports always see a canonical ``host`` (lowercase,
no scheme, no port, no path, no defang, no IDN encoding). This is the
single place that knows about URL parsing, IDN round-trips, and
"defanging" — every other module trusts the shape :class:`NormalizedTarget`
guarantees.

Two-step pipeline
-----------------

1. :func:`refang` rewrites the common social-media / mail-client
   defangs back to live syntax:
   - ``hxxp://`` / ``hxxps://`` → ``http://`` / ``https://``
   - ``[.]`` and ``(.)`` and ``{.}`` → ``.``
   - ``[@]`` → ``@``
   - leading ``http://`` and ``https://`` are stripped once present
2. URL parse → if it has a host, use the host. If it's a bare word
   (no scheme, no slash), treat it as a hostname. Then ``idna`` encode
   unicode → ASCII so DNS / WHOIS / ipinfo all see bytes they
   understand.

The output is a frozen dataclass with three fields: the *display* form
(``display`` — what the user typed, lightly cleaned), the *canonical
host* (``host`` — what the transports use), and the *original input*
(``raw`` — kept for the audit log).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Final

# --- Defang rewrites --------------------------------------------------------
# Order matters: longer tokens first so ``hxxps://`` is matched before
# a single ``hxxp://`` slice would leave a stray ``s://``.
_DEFANG_SCHEMES: Final[tuple[tuple[str, str], ...]] = (
    ("hxxps://", "https://"),
    ("hxxp://", "http://"),
    ("[.]", "."),
    ("(.)", "."),
    ("{.}", "."),
    ("[dot]", "."),
    ("(dot)", "."),
    ("{dot}", "."),
    ("[:]", ":"),
    ("[@]", "@"),
    ("(at)", "@"),
    ("{at}", "@"),
)

# A *very* conservative scan: strip leading/trailing whitespace, then
# drop a single optional scheme. We deliberately do NOT use the result
# of this scan to feed urllib — the refang step above already
# canonicalized the common defangs.
_SCHEME_PREFIX: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:https?|ftp)://",
    re.IGNORECASE,
)


def refang(text: str) -> str:
    """Rewrite common defanged syntax back to live syntax.

    The function is intentionally narrow: it handles the patterns that
    show up in phishing emails, Twitter posts, and PDF threat-intel
    reports, and ignores anything more exotic. The output is a
    *string*, not a parsed URL — the next step in the pipeline parses.

    The function is idempotent: ``refang(refang(x)) == refang(x)``
    because every substitution replaces a defang token with a token
    that does not itself match any defang pattern.
    """
    if not text:
        return text
    out = unicodedata.normalize("NFKC", text)
    # Pre-pass: collapse the *bracket* defangs (``[.]``, ``[:]``,
    # ``[@]``) BEFORE the scheme-token rewrite. Otherwise a defanged
    # URL like ``hxxps[:]//example[.]com`` would survive the
    # ``hxxps://`` rewrite (because the literal ``hxxps://`` substring
    # is broken up by the brackets) and we'd be left with a hybrid
    # like ``hxxps://example.com`` that is still not fanged.
    for needle, replacement in _DEFANG_SCHEMES:
        if not needle.startswith("hxxp"):
            out = out.replace(needle, replacement)
    # Now do the scheme-token rewrite. Collapse the ``xx`` in
    # ``hxxp`` / ``hxxps`` to ``tt``, preserving the trailing
    # optional ``s`` and optional ``:`` (the regex captures it via
    # the ``(s?)`` group and the replacement re-emits it as ``\1``).
    # The IGNORECASE flag covers ``hXXp`` / ``HXXPS`` / etc. The
    # rewrite is unambiguous because the bracket defangs were
    # already collapsed above — there is no way for a stray
    # ``hxxp`` substring to survive without us wanting to rewrite it.
    out = re.sub(r"hxxp(s?)", r"http\1", out, flags=re.IGNORECASE)
    return out


# --- IDN handling -----------------------------------------------------------
def _idna_encode(host: str) -> str:
    """Best-effort IDN → ASCII (punycode) encode.

    Returns the input unchanged if it is already ASCII or if ``idna``
    encoding fails (malformed label, empty string, etc.). The
    downstream transports get ASCII so DNS, ipinfo, and crt.sh all
    speak the same wire format.
    """
    if not host:
        return host
    if host.isascii():
        return host
    try:
        # ``idna`` is in the stdlib; ``encodings.idna`` is the same
        # module. We import lazily so this module remains importable
        # even if a future refactor splits IDN handling out.
        return host.encode("idna").decode("ascii")
    except UnicodeError:
        return host


def _strip_scheme_and_path(raw: str) -> str:
    """Pull the host portion out of a URL-or-bare-word string.

    Strips a single optional ``http://`` / ``https://`` / ``ftp://``
    prefix, then drops any path / query / fragment. A bare word (no
    slash) is returned verbatim. The result is **not** lowercased —
    that is the caller's job so the display form keeps its case.
    """
    s = raw.strip()
    s = _SCHEME_PREFIX.sub("", s, count=1)
    # If there's a path / query / fragment, drop everything from the
    # first '/', '?', or '#' onwards. The first character of the host
    # is the start of the string at this point.
    for sep in ("/", "?", "#"):
        idx = s.find(sep)
        if idx != -1:
            s = s[:idx]
    # Drop a trailing :port if and only if the port is numeric. This
    # keeps the rare case ``host:8080`` from being misparsed — it
    # is rare but the safety module rejects ports anyway, so we just
    # drop the suffix here.
    if ":" in s:
        # Don't trip on the last colon of an IPv6 literal "::1"; that
        # case is handled by the safety module, not here.
        prefix, _, maybe_port = s.rpartition(":")
        if maybe_port.isdigit():
            s = prefix
    return s


@dataclass(frozen=True)
class NormalizedTarget:
    """Canonical form of a recon target.

    ``raw``     : exact input from the user (kept for the audit log).
    ``display`` : lightly cleaned user-facing form (refanged, trimmed).
    ``host``    : canonical hostname — what every transport queries.

    The three-field split is deliberate: the audit log wants ``raw`` so
    an operator can see exactly what the user typed, the report header
    wants ``display`` so the user sees the form they recognize, and
    the transports want ``host`` so they never have to re-parse.
    """

    raw: str
    display: str
    host: str


def normalize_target(text: str) -> NormalizedTarget:
    """Normalize a user-supplied recon target.

    The function is total: any string input produces a
    :class:`NormalizedTarget` with a non-empty ``host`` UNLESS the
    input is empty or whitespace-only, in which case ``ValueError`` is
    raised. There is no other failure mode — the function does not
    touch the network, does not import the recon transports, and does
    not require a valid TLD.
    """
    if not text or not text.strip():
        raise ValueError("recon target must not be empty")
    raw = text
    display = refang(raw).strip()
    # strip scheme + path so the host is what we hand to transports
    candidate = _strip_scheme_and_path(display)
    # IDN round-trip. _idna_encode returns ASCII (or the original
    # unicode if encode fails); we lowercase here so the safety
    # module and the transports can do case-insensitive compares.
    host_ascii = _idna_encode(candidate).lower().strip().rstrip(".")
    if not host_ascii:
        raise ValueError(f"recon target has no host: {raw!r}")
    return NormalizedTarget(raw=raw, display=display.strip(), host=host_ascii)

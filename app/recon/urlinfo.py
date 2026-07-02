"""In-process URL probe for the recon subsystem.

Phase 15 (PR-E rewire) — fetch a lightweight HTTP-fingerprint record
for a URL **directly**, with no third-party proxy. We do a ``HEAD``
request against the target, follow redirects, fall back to ``GET`` if
the server rejects ``HEAD`` (some sites return 405), and read at most
``_TITLE_BODY_LIMIT`` bytes of the response body to extract the
``<title>`` tag.

Why we changed approach
-----------------------

The previous design proxied through ``urlinfo.io``, a small
single-operator service with no SLA. From many regions (and from
this dev environment) ``urlinfo.io`` times out at the 15 s
``RECON_HTTP_TIMEOUT_SECONDS`` wall-clock, which made the whole
recon turn feel broken even though the other four tools were fine.

Probing the target URL directly is faster, has no third-party
dependency, and is what every other security recon tool does (nmap,
httpx, Nuclei templates). The cost is one extra outbound connection
to the *target*, which the safety rail has already approved.

Endpoint
--------

``GET https://<host>/`` (or whatever the caller passes) — we never
hit a third-party endpoint. The response yields the data the report
needs: the final URL after redirects, the HTTP status, the page
title, the server banner, the content type, and the content length.
We do **not** parse the body beyond a small title scan.

Failure modes
-------------

- The target is unreachable (DNS, connection refused, timeout):
  the function raises :class:`URLInfoError` and the orchestrator
  marks the tool as ``status="error"``.
- The target responds with a non-2xx status: we still return a
  populated :class:`URLInfoResult` with ``http_status`` set to the
  real status — a 404 is useful intel, not a transport error.
- The server returns a non-HTML body: ``title`` and ``content_type``
  fall back to empty strings; ``content_length`` reflects the actual
  payload size.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Final, Optional

from urllib.error import HTTPError

from .. import config

_TIMEOUT: Final[float] = config.RECON_HTTP_TIMEOUT_SECONDS
_USER_AGENT: Final[str] = "SecMentor-Stage1-Recon/0.1 (+local)"

#: Hard cap on the number of bytes of the body we'll read while
#: hunting for a ``<title>`` tag. 64 KiB is more than enough for the
#: ``<head>`` of any reasonable HTML page; we never download the
#: whole body.
_TITLE_BODY_LIMIT: Final[int] = 64 * 1024

#: A loose, deliberately non-strict ``<title>...</title>`` regex.
#: We use ``re.IGNORECASE | re.DOTALL`` so multi-line titles and
#: mixed-case tags both match. We are NOT building a full HTML
#: parser — we only need the first title on the page for the
#: report.
_TITLE_RE: Final[re.Pattern[str]] = re.compile(
    r"<title\b[^>]*>(.*?)</title>",
    re.IGNORECASE | re.DOTALL,
)


class URLInfoError(RuntimeError):
    """Raised when the in-process URL probe fails.

    A 4xx / 5xx response from the *target* is **not** an error — it
    is recorded in :attr:`URLInfoResult.http_status`. The only
    transport-level failures that raise are network / DNS / timeout
    errors and malformed redirects.
    """


@dataclass(frozen=True)
class URLInfoResult:
    """Parsed URL probe record.

    ``requested_url`` : the URL we asked about.
    ``final_url``     : the URL after redirects (may equal requested).
    ``http_status``   : the status code we ultimately got, or 0 if
                        the probe didn't see a response.
    ``title``         : the page <title>, or "" if absent / non-HTML.
    ``server``        : the Server: response header, or "".
    ``content_type``  : the Content-Type: response header, or "".
    ``content_length``: the Content-Length: header, or 0 if absent.
    """

    requested_url: str
    final_url: str
    http_status: int
    title: str
    server: str
    content_type: str
    content_length: int

    @property
    def redirected(self) -> bool:
        """True if the final URL differs from the requested URL."""
        return bool(self.final_url) and self.final_url != self.requested_url


def _coerce_int(value: Any, default: int = 0) -> int:
    """Best-effort coercion of a header / value to ``int``.

    Used for the ``Content-Length`` and HTTP status code. Booleans
    are guarded first because ``bool`` is a subclass of ``int``.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


def _coerce_str(value: Any) -> str:
    """Best-effort coercion of a header value to ``str``.

    Returns the empty string for ``None`` and the str() of any other
    type so a missing header is rendered as ``""`` rather than
    crashing the renderer.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _read_title(body: bytes) -> str:
    """Extract the page title from a (possibly truncated) body.

    Decodes as UTF-8 with ``errors="replace"`` so a malformed byte
    sequence does not blow up the probe. Returns ``""`` for empty
    bodies or when no ``<title>`` tag is found.
    """
    if not body:
        return ""
    text = body.decode("utf-8", errors="replace")
    m = _TITLE_RE.search(text)
    if not m:
        return ""
    # Collapse all whitespace runs (including the newlines that
    # ``re.DOTALL`` lets through) into single spaces and strip.
    return re.sub(r"\s+", " ", m.group(1)).strip()


def _safe_redirect_url(current: str, location: str) -> Optional[str]:
    """Resolve a ``Location:`` header against ``current``.

    Returns the absolute URL the redirect points to, or ``None`` if
    the location is malformed. We use :func:`urllib.parse.urljoin`
    so a relative ``Location: /path`` becomes absolute against
    ``current``'s scheme + host.
    """
    if not location:
        return None
    try:
        joined = urllib.parse.urljoin(current, location)
    except (ValueError, TypeError):
        return None
    # urljoin returns the *base* unchanged when both inputs are
    # empty / unparseable; require an actual scheme + host to
    # consider the redirect valid.
    parsed = urllib.parse.urlparse(joined)
    if not parsed.scheme or not parsed.netloc:
        return None
    return joined


def _build_opener() -> urllib.request.OpenerDirector:
    """Build a urllib opener that records the final URL after redirects.

    We override ``http_error_302`` (and the rest of the 3xx family)
    with a no-op so :func:`urllib.request.urlopen` follows redirects
    transparently and the returned ``resp.url`` is the final URL
    after the chain.
    """

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def http_error_301(  # type: ignore[override]
            self, req, fp, code, msg, headers
        ):
            return None  # let the default handler chain follow it

        def http_error_302(  # type: ignore[override]
            self, req, fp, code, msg, headers
        ):
            return None

        def http_error_303(self, req, fp, code, msg, headers):  # type: ignore[override]
            return None

        def http_error_307(self, req, fp, code, msg, headers):  # type: ignore[override]
            return None

        def http_error_308(self, req, fp, code, msg, headers):  # type: ignore[override]
            return None

    return urllib.request.build_opener(_NoRedirect())


def _do_request(url: str) -> tuple[Any, bytes]:
    """Issue the probe request, falling back HEAD -> GET on 405/501.

    Strategy
    --------
    1. Try ``HEAD`` first. Many servers support it, and it never
       transfers a body — fast and cheap.
    2. If the server rejects ``HEAD`` with 405/501, retry as
       ``GET`` so we still get headers + body for the title.
    3. If the HEAD response is 2xx and has a text-y Content-Type,
       follow up with a bounded ``GET`` to grab a body sample for
       the title scan. We never download the full page.

    Returns ``(response, body)``. ``body`` is empty if we never
    issued a GET; otherwise it is at most
    :data:`_TITLE_BODY_LIMIT` bytes.

    Raises :class:`URLInfoError` on transport failures.
    """
    opener = _build_opener()
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    req.get_method = lambda: "HEAD"  # type: ignore[method-assign]
    try:
        resp = opener.open(req, timeout=_TIMEOUT)
    except urllib.error.HTTPError as exc:
        # 405 Method Not Allowed (or 501 Not Implemented) -> the
        # server doesn't support HEAD. Fall back to a GET so we
        # still get the headers + a body for the title.
        if exc.code in (405, 501):
            req2 = urllib.request.Request(
                url, headers={"User-Agent": _USER_AGENT}
            )
            resp = opener.open(req2, timeout=_TIMEOUT)
            body = resp.read(_TITLE_BODY_LIMIT)
            return resp, body
        # Any other HTTP error: surface as a transport failure.
        raise URLInfoError(
            f"urlinfo HTTPError {exc.code} for {url}: {exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise URLInfoError(
            f"urlinfo URLError for {url}: {exc.reason}"
        ) from exc
    except OSError as exc:
        raise URLInfoError(
            f"urlinfo OSError for {url}: {exc}"
        ) from exc
    # HEAD succeeded with a 2xx response. If the content type
    # looks like HTML, issue a follow-up bounded GET so we can
    # extract the <title>. Otherwise (images, PDFs, etc.) skip
    # the body to save bandwidth.
    headers = getattr(resp, "headers", None)
    ctype = _coerce_str(headers.get("Content-Type")) if headers else ""
    if "text/html" in ctype.lower() or "application/xhtml" in ctype.lower():
        try:
            req2 = urllib.request.Request(
                url,
                headers={
                    "User-Agent": _USER_AGENT,
                    "Range": f"bytes=0-{_TITLE_BODY_LIMIT - 1}",
                },
            )
            resp2 = opener.open(req2, timeout=_TIMEOUT)
            body = resp2.read(_TITLE_BODY_LIMIT)
        except (urllib.error.URLError, OSError, HTTPError) as exc:
            # Body is best-effort; never let a GET failure kill the
            # probe — we already have a valid HEAD response.
            body = b""
        return resp, body
    # HEAD succeeded but the body is non-HTML: skip the body.
    return resp, b""


def probe(url: str) -> URLInfoResult:
    """Probe ``url`` in-process and return a :class:`URLInfoResult`.

    The argument is the *URL the user asked about*, not the
    hostname. The function prepends ``https://`` if no scheme is
    present so a bare ``example.com`` works the same as
    ``https://example.com``.

    On any transport failure (DNS, connection refused, timeout,
    malformed redirect) the function raises :class:`URLInfoError`
    with a message that includes the URL and the underlying
    exception class. The orchestrator catches the error, marks the
    tool as ``status="error"`` in the report, and logs the failure.
    """
    if not url or not url.strip():
        raise URLInfoError("urlinfo probe requires a non-empty URL")
    target = url.strip()
    if "://" not in target:
        # A bare host like ``example.com`` becomes
        # ``https://example.com/`` so the path is well-defined and
        # the report shows a canonical URL.
        target = f"https://{target}/"
    else:
        # If the caller passed a scheme but no path, add one for
        # the same reason. ``urllib.parse.urlparse("https://h")``
        # gives ``path=""``, which is what we want to fix.
        parsed_target = urllib.parse.urlparse(target)
        if parsed_target.scheme and not parsed_target.path:
            target = urllib.parse.urlunparse(parsed_target._replace(path="/"))
    try:
        resp, body = _do_request(target)
    except URLInfoError:
        raise
    # ``resp.url`` is the final URL after redirects (urllib sets
    # it). If the response is a bare object that only exposes
    # ``geturl()`` (some addinfourl implementations), fall back to
    # that. If neither is present / neither parses, use the
    # requested target unchanged.
    final_url = _coerce_str(getattr(resp, "url", None)) or _coerce_str(
        getattr(resp, "geturl", lambda: "")()
    ) or target
    # Defensive: if the opener gave us a relative final URL,
    # resolve it against the requested one.
    if final_url and not urllib.parse.urlparse(final_url).netloc:
        final_url = _safe_redirect_url(target, final_url) or target
    headers = getattr(resp, "headers", None)
    return URLInfoResult(
        requested_url=target,
        final_url=final_url,
        http_status=_coerce_int(getattr(resp, "status", None) or resp.getcode(), default=0),
        title=_read_title(body),
        server=_coerce_str(headers.get("Server")) if headers else "",
        content_type=_coerce_str(headers.get("Content-Type")) if headers else "",
        content_length=_coerce_int(
            headers.get("Content-Length") if headers else None, default=0
        ),
    )
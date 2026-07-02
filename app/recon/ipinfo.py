"""IP metadata lookup for the recon subsystem.

Phase 15 (PR-E rewire) — fetch the public-IP metadata record for an IP
literal returned by :mod:`app.recon.dns_lookup`. The default upstream
is `ipapi.co` (no token, no rate-limit problem for our volume), with
`ipinfo.io` kept as a paid-tier fallback for operators who configure
``IPINFO_TOKEN``.

Endpoint selection
------------------

- **Default (no token):** GET ``https://ipapi.co/<ip>/json/`` — full
  record (city / region / country / ASN / org / postal / timezone /
  lat / lng). The free tier of ipapi.co is sufficient for low-volume
  recon (a few requests per minute per source IP) and does not
  require an API key.
- **With ``IPINFO_TOKEN`` configured:** GET
  ``https://ipinfo.io/<ip>?token=<token>`` — full ipinfo.io record.
  We fall back to this when the operator has paid for ipinfo.io and
  prefers its data shape.

Why we changed providers
------------------------

ipinfo.io's no-token ``/lite`` endpoint was deprecated and now returns
``404`` from many regions. ipapi.co has a stable free tier, no token,
and returns an equivalent record for the fields we render.

The HTTP request uses the stdlib :mod:`urllib.request` so we don't
take a new dependency on ``requests`` for a single GET. The request
is wrapped in a timeout from :data:`app.config.RECON_HTTP_TIMEOUT_SECONDS`.

The result is a frozen dataclass with a fixed shape; missing fields
are returned as empty strings or empty tuples so the renderer does
not have to special-case ``None``.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Final

from .. import config

#: Base URL for the default provider (ipapi.co). No token required.
_DEFAULT_BASE_URL: Final[str] = "https://ipapi.co"
_DEFAULT_PATH_TEMPLATE: Final[str] = "/{ip}/json/"

#: Paid-tier fallback: ipinfo.io with a token.
_PAID_BASE_URL: Final[str] = "https://ipinfo.io"
_TIMEOUT: Final[float] = config.RECON_HTTP_TIMEOUT_SECONDS

#: User-Agent the request advertises. Both providers log this and
#: contact the operator on abuse reports, so we identify ourselves
#: honestly.
_USER_AGENT: Final[str] = "SecMentor-Stage1-Recon/0.1 (+local)"


class IPInfoError(RuntimeError):
    """Raised when an ipinfo.io lookup fails.

    The error message includes the URL and the underlying exception
    class name so the operator can tell apart a network failure from
    a JSON parse failure from a non-200 HTTP status.
    """


@dataclass(frozen=True)
class IPInfoResult:
    """Parsed ipinfo.io record.

    All fields are strings; missing fields are empty strings (not
    ``None``) so the renderer can use ``f"{r.city}"`` without a
    guard. ``raw`` holds the unparsed JSON for the audit log and
    for fields this dataclass does not enumerate.
    """

    ip: str
    hostname: str
    city: str
    region: str
    country: str
    loc: str          # "lat,lng"
    org: str          # "AS15169 Google LLC"
    postal: str
    timezone: str
    # Lite-mode fields are present on the full record too, but the
    # values are the same shape (single string). We do not split
    # ``loc`` into (lat, lng) — the renderer does that.
    raw: dict[str, Any]

    @property
    def asn(self) -> str:
        """Return just the AS number (``"AS15169"``) from ``org``.

        Returns the empty string if ``org`` is empty or does not
        start with ``AS`` (some lite records omit the ASN entirely).
        """
        if not self.org:
            return ""
        head, _, _ = self.org.partition(" ")
        return head if head.startswith("AS") else ""


def _build_url(ip: str) -> str:
    """Build the provider URL for the configured mode.

    With ``IPINFO_TOKEN`` configured we use ipinfo.io's full endpoint
    and URL-encode the token (the spec allows ``+``, ``/``, ``=`` in
    the payload). Without a token we use ipapi.co's public JSON
    endpoint, which does not require authentication for low volume.
    """
    token = (config.IPINFO_TOKEN or "").strip()
    if token:
        return f"{_PAID_BASE_URL}/{ip}?token={urllib.parse.quote(token)}"
    # ipapi.co shape: /<ip>/json/ — trailing slash matters (some
    # CDN edges 301-redirect otherwise).
    return f"{_DEFAULT_BASE_URL}{_DEFAULT_PATH_TEMPLATE.format(ip=ip)}"


def _adapt_ipapi_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Translate ipapi.co's JSON shape into the flat keys ``_parse_payload`` reads.

    ipapi.co returns::

        {"ip": ..., "city": ..., "region": ..., "country": "US",
         "country_name": "United States", "postal": ..., "latitude": 12.34,
         "longitude": -56.78, "timezone": ..., "asn": "AS15169",
         "org": "Google LLC", ...}

    Our dataclass wants ``loc = "lat,lng"``, a single ``country`` code
    (which ipapi.co already provides as the 2-letter ISO code), and
    an ``org`` that already includes the ``ASxxxxx`` prefix when the
    upstream supplies both fields. We build the ``loc`` string and
    leave the rest as-is.
    """
    out = dict(payload)
    lat = payload.get("latitude")
    lng = payload.get("longitude")
    if isinstance(lat, (int, float)) and isinstance(lng, (int, float)):
        out["loc"] = f"{lat},{lng}"
    # ipapi.co already returns the 2-letter ISO code as ``country``,
    # so no rename is needed for ``country``. ``asn`` is a separate
    # key on ipapi.co; ``_parse_payload`` reads only ``org``, and our
    # adapter below folds ``asn`` into ``org`` for the dataclass
    # contract.
    asn = payload.get("asn")
    org = payload.get("org")
    if asn and org and "asn" not in out.get("org", "").lower():
        # Combine so the dataclass' ``.asn`` property (which reads the
        # leading ``ASxxxxx`` token from ``org``) keeps working.
        out["org"] = f"{asn} {org}"
    elif asn and not org:
        out["org"] = str(asn)
    return out


def _parse_payload(ip: str, payload: dict[str, Any]) -> IPInfoResult:
    """Map a raw JSON dict into an :class:`IPInfoResult`.

    The upstream may be ipapi.co (which we adapt via
    :func:`_adapt_ipapi_payload`) or ipinfo.io (flat keys). Both
    shapes funnel into the same dataclass.

    The lite / minimal endpoint of either provider returns a smaller
    dict (no ``org``, no ``postal``, no ``timezone``); missing keys
    become empty strings so the dataclass is always fully populated.
    """
    def s(key: str) -> str:
        v = payload.get(key, "")
        return v if isinstance(v, str) else ""

    return IPInfoResult(
        ip=s("ip") or ip,
        hostname=s("hostname"),
        city=s("city"),
        region=s("region"),
        country=s("country"),
        loc=s("loc"),
        org=s("org"),
        postal=s("postal"),
        timezone=s("timezone"),
        raw=dict(payload),
    )


def lookup(ip: str) -> IPInfoResult:
    """Fetch and parse the ipinfo.io record for ``ip``.

    The function is total: any failure (network, JSON parse,
    non-200 status) raises :class:`IPInfoError` with a message that
    includes the URL and the underlying exception class. The
    orchestrator catches the error, marks the tool as
    ``status="error"`` in the report, and logs the failure.
    """
    if not ip or not ip.strip():
        raise IPInfoError("ipinfo lookup requires a non-empty IP")
    url = _build_url(ip.strip())
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            status = getattr(resp, "status", None) or resp.getcode()
            if status != 200:
                raise IPInfoError(
                    f"ipinfo {ip} returned HTTP {status} from {url}"
                )
            body = resp.read()
    except urllib.error.HTTPError as exc:
        raise IPInfoError(
            f"ipinfo {ip} HTTPError {exc.code} from {url}: {exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise IPInfoError(
            f"ipinfo {ip} URLError from {url}: {exc.reason}"
        ) from exc
    except OSError as exc:
        raise IPInfoError(f"ipinfo {ip} OSError from {url}: {exc}") from exc
    try:
        payload = json.loads(body.decode("utf-8", errors="replace"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise IPInfoError(
            f"ipinfo {ip} returned non-JSON body: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise IPInfoError(
            f"ipinfo {ip} returned non-object JSON of type "
            f"{type(payload).__name__}"
        )
    # Detect ipapi.co's response by the presence of ``latitude`` /
    # ``longitude`` (ipinfo.io uses a single ``loc`` string instead).
    # When we see the ipapi.co shape, run it through the adapter so
    # downstream code only ever sees the flat ipinfo.io-style keys.
    if "latitude" in payload and "longitude" in payload and "loc" not in payload:
        payload = _adapt_ipapi_payload(payload)
    return _parse_payload(ip, payload)

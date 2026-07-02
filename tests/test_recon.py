"""Unit tests for the Phase 15 recon subsystem.

These tests pin the *public contract* of every recon module so a
refactor that silently changes a field name or a signature trips
the test suite. Network access is stubbed at the stdlib boundary
(``socket``, ``urllib.request``) so the tests are hermetic.

Layout
------

- :class:`ParseReconCommandTests`     — ``/recon`` grammar
- :class:`RefangTests`                 — defang rewriting
- :class:`NormalizeTargetTests`        — normalization pipeline
- :class:`SafetyTests`                 — blocklist enforcement
- :class:`DNSResolveTests`             — ``getaddrinfo`` mocked
- :class:`UrlInfoProbeTests`           — in-process URL probe
- :class:`IpInfoLookupTests`           — ipapi.co / ipinfo.io lookup
- :class:`WhoisLookupTests`            — two-hop TCP/43
- :class:`CrtShLookupTests`            — crt.sh JSON endpoint
- :class:`OrchestratorTests`           — end-to-end with mocked transports
- :class:`RenderMarkdownTests`         — markdown renderer
- :class:`RenderJsonTests`             — JSON renderer
- :class:`ReconStorageTests`           — storage repo functions
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import os
import socket
import sqlite3
import tempfile
import unittest
import urllib.error
from pathlib import Path
from typing import Any, Optional
from unittest import mock

from app import config, storage
from app.recon import (
    NormalizedTarget,
    ReconReport,
    TargetBlockedError,
    assert_target_allowed,
    normalize_target,
    refang,
    render_report_json,
    render_report_markdown,
    run_recon,
)
from app.recon import crt_sh, dns_lookup, ipinfo, normalize, orchestrator, report
from app.recon import safety
from app.recon import urlinfo, whois
from app.recon.orchestrator import ToolResult

# Pull the parse_recon_command helper out of web.chat_helpers without
# importing streamlit. chat_helpers is the *pure* module that holds the
# grammar; the Streamlit view imports from it.
from web.chat_helpers import (
    DEFAULT_RECON_SCOPE_TOKEN,
    ReconCommand,
    parse_recon_command,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _FakeSocket:
    """Minimal context-manager socket double for the WHOIS tests.

    The WHOIS module calls ``socket.create_connection((host, port))``
    and immediately enters the returned object (``with sock: ...``).
    We hand it a callable that returns ``self`` so the ``with`` block
    binds to us, and we expose ``sendall`` / ``recv`` as recording
    stubs. Two of these chained together model the IANA hop and the
    registrar hop.
    """

    def __init__(self, response: bytes) -> None:
        self.response = response
        self.sent: list[bytes] = []

    # Context manager protocol --------------------------------------------
    def __enter__(self) -> "_FakeSocket":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        return None

    # WHOIS API surface ----------------------------------------------------
    def sendall(self, data: bytes) -> None:
        self.sent.append(data)

    def recv(self, _n: int) -> bytes:
        return self.response


def _make_open_side_effect(body: bytes, status: int = 200) -> mock.MagicMock:
    """Build a ``urllib.request.urlopen`` mock that returns *body* + *status*."""

    fake_response = mock.MagicMock()
    fake_response.read.return_value = body
    fake_response.status = status
    fake_response.__enter__.return_value = fake_response
    fake_response.__exit__.return_value = False
    return fake_response


# Sample factories ----------------------------------------------------------


def _fake_dns_ok() -> dns_lookup.DNSResult:
    return dns_lookup.DNSResult(
        ipv4=["93.184.216.34"],
        ipv6=["2606:2800:220:1:248:1893:25c8:1946"],
        error=None,
    )


def _fake_ipinfo_ok() -> ipinfo.IPInfoResult:
    return ipinfo.IPInfoResult(
        ip="93.184.216.34",
        hostname="example.com",
        city="Norwell",
        region="Massachusetts",
        country="US",
        loc="42.1508,-70.9495",
        org="AS15169 Google LLC",
        postal="02061",
        timezone="America/New_York",
        raw={"ip": "93.184.216.34"},
    )


def _fake_urlinfo_ok(
    requested_url: str = "https://example.com/",
    final_url: str = "https://example.com/",
    http_status: int = 200,
) -> urlinfo.URLInfoResult:
    return urlinfo.URLInfoResult(
        requested_url=requested_url,
        final_url=final_url,
        http_status=http_status,
        title="Example Domain",
        server="ECS (sec/9704)",
        content_type="text/html; charset=UTF-8",
        content_length=1256,
    )


def _fake_whois_ok() -> whois.WHOISResult:
    return whois.WHOISResult(
        iana_body=(
            "refer: whois.verisign-grs.com\n"
            "domain: EXAMPLE.COM\n"
            "status: active\n"
        ),
        registrar_server="whois.verisign-grs.com",
        registrar_body=(
            "Domain Name: EXAMPLE.COM\n"
            "Registry Domain ID: 2336799_DOMAIN_COM-VRSN\n"
        ),
    )


def _fake_crt_ok() -> crt_sh.CrtShResult:
    return crt_sh.CrtShResult(
        hosts=["example.com", "www.example.com"],
        cert_count=2,
        issuers=["DigiCert"],
        raw=[],
    )


def _sample_report() -> ReconReport:
    """Build a :class:`ReconReport` with every tool succeeding."""
    return ReconReport(
        target="example.com",
        display="example.com",
        host="example.com",
        scope_token="lab",
        total_ms=42,
        dns=ToolResult(
            tool="dns", ok=True, value=_fake_dns_ok(),
            error=None, duration_ms=3,
        ),
        ipinfo=ToolResult(
            tool="ipinfo", ok=True, value=_fake_ipinfo_ok(),
            error=None, duration_ms=12,
        ),
        urlinfo=ToolResult(
            tool="urlinfo", ok=True, value=_fake_urlinfo_ok(),
            error=None, duration_ms=11,
        ),
        whois=ToolResult(
            tool="whois", ok=True, value=_fake_whois_ok(),
            error=None, duration_ms=9,
        ),
        crt_sh=ToolResult(
            tool="crt_sh", ok=True, value=_fake_crt_ok(),
            error=None, duration_ms=7,
        ),
    )


# ---------------------------------------------------------------------------
# /recon grammar
# ---------------------------------------------------------------------------


class ParseReconCommandTests(unittest.TestCase):
    """Pin the slash-command parser contract."""

    def test_lowercase_prefix_with_target(self) -> None:
        cmd = parse_recon_command("/recon example.com")
        self.assertIsInstance(cmd, ReconCommand)
        assert cmd is not None
        self.assertEqual(cmd.target, "example.com")
        self.assertEqual(cmd.scope_token, DEFAULT_RECON_SCOPE_TOKEN)

    def test_uppercase_prefix_rejected(self) -> None:
        # Grammar is case-sensitive: only lowercase /recon is recognised.
        self.assertIsNone(parse_recon_command("/RECON example.com"))

    def test_target_with_explicit_scope(self) -> None:
        cmd = parse_recon_command("/recon example.com scope=lab")
        assert cmd is not None
        self.assertEqual(cmd.target, "example.com")
        self.assertEqual(cmd.scope_token, "lab")

    def test_target_with_url_and_scope(self) -> None:
        cmd = parse_recon_command(
            "/recon https://api.github.com/users/torvalds scope=ctf"
        )
        assert cmd is not None
        self.assertEqual(cmd.target, "https://api.github.com/users/torvalds")
        self.assertEqual(cmd.scope_token, "ctf")

    def test_bare_prefix_rejected(self) -> None:
        # /recon with no target is not a recon command; falls through.
        self.assertIsNone(parse_recon_command("/recon"))

    def test_bare_prefix_with_whitespace_rejected(self) -> None:
        self.assertIsNone(parse_recon_command("/recon   "))

    def test_other_command_rejected(self) -> None:
        self.assertIsNone(parse_recon_command("/help"))
        self.assertIsNone(parse_recon_command("/model gpt-4"))

    def test_non_string_rejected(self) -> None:
        self.assertIsNone(parse_recon_command(None))  # type: ignore[arg-type]
        self.assertIsNone(parse_recon_command(123))  # type: ignore[arg-type]

    def test_target_with_internal_spaces(self) -> None:
        # Trailing space-split joins all non-scope tokens with spaces.
        cmd = parse_recon_command("/recon two words scope=lab")
        assert cmd is not None
        self.assertEqual(cmd.target, "two words")
        self.assertEqual(cmd.scope_token, "lab")

    def test_scope_token_lowercased(self) -> None:
        cmd = parse_recon_command("/recon example.com scope=LAB")
        assert cmd is not None
        self.assertEqual(cmd.scope_token, "lab")

    def test_scope_only_with_empty_value(self) -> None:
        # "scope=" with no value: token becomes empty string, target keeps
        # the literal "scope=" text. Pin current behaviour.
        cmd = parse_recon_command("/recon example.com scope=")
        assert cmd is not None
        self.assertEqual(cmd.target, "example.com scope=")
        self.assertEqual(cmd.scope_token, "")

    def test_leading_whitespace_allowed(self) -> None:
        cmd = parse_recon_command("  /recon example.com  ")
        assert cmd is not None
        self.assertEqual(cmd.target, "example.com")

    def test_unknown_scope_token_still_parsed(self) -> None:
        # Parser does NOT validate scope_token; the orchestrator does that
        # so the audit log can record the rejected value.
        cmd = parse_recon_command("/recon example.com scope=does_not_exist")
        assert cmd is not None
        self.assertEqual(cmd.scope_token, "does_not_exist")

    def test_reconcommand_is_frozen(self) -> None:
        # Frozen dataclass: a parser bug cannot mutate the value.
        cmd = parse_recon_command("/recon example.com")
        assert cmd is not None
        with self.assertRaises(dataclasses.FrozenInstanceError):
            cmd.target = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Refang
# ---------------------------------------------------------------------------


class RefangTests(unittest.TestCase):
    """Defang → fang rewriting contract."""

    def test_rewrites_hxxp(self) -> None:
        self.assertEqual(refang("hxxp://example.com"), "http://example.com")

    def test_rewrites_hXXp(self) -> None:
        self.assertEqual(refang("hXXp://example.com"), "http://example.com")

    def test_rewrites_hxxps(self) -> None:
        self.assertEqual(refang("hxxps://example.com"), "https://example.com")

    def test_rewrites_bracket_dot(self) -> None:
        self.assertEqual(
            refang("hxxp://example[.]com"),
            "http://example.com",
        )

    def test_rewrites_bracket_colon(self) -> None:
        self.assertEqual(
            refang("hxxps[:]//example[.]com/path"),
            "https://example.com/path",
        )

    def test_idempotent_for_clean_input(self) -> None:
        # A clean URL is returned verbatim — refang is safe to call twice.
        self.assertEqual(
            refang("https://example.com/path?x=1"),
            "https://example.com/path?x=1",
        )

    def test_does_not_strip_internal_spaces(self) -> None:
        # Spaces are NOT refang material; only the documented patterns
        # are rewritten. This is the current contract; pin it.
        self.assertEqual(
            refang("example . com"),
            "example . com",
        )

    def test_does_not_strip_zero_width(self) -> None:
        # Zero-width spaces are not handled by refang. Pinning this so a
        # future change is intentional.
        self.assertEqual(
            refang("example\u200b.com"),
            "example\u200b.com",
        )


# ---------------------------------------------------------------------------
# Normalize
# ---------------------------------------------------------------------------


class NormalizeTargetTests(unittest.TestCase):
    """Normalization pipeline: scheme stripping, IDN, lowercasing."""

    def test_bare_domain(self) -> None:
        n = normalize_target("example.com")
        self.assertEqual(n.host, "example.com")
        self.assertEqual(n.display, "example.com")
        self.assertEqual(n.raw, "example.com")

    def test_full_url_preserves_www_lowercases_host(self) -> None:
        n = normalize_target("https://www.Example.com:443/foo?bar=1")
        # `www.` is preserved — it is a real subdomain, not noise.
        self.assertEqual(n.host, "www.example.com")
        # Display keeps original case and path.
        self.assertEqual(
            n.display,
            "https://www.Example.com:443/foo?bar=1",
        )
        self.assertEqual(
            n.raw,
            "https://www.Example.com:443/foo?bar=1",
        )

    def test_uppercase_subdomain_lowercased(self) -> None:
        n = normalize_target("API.github.com")
        self.assertEqual(n.host, "api.github.com")
        # Display keeps the user-supplied case.
        self.assertEqual(n.display, "API.github.com")

    def test_path_only_kept_in_display(self) -> None:
        n = normalize_target("example.com/path?x=1")
        self.assertEqual(n.host, "example.com")
        self.assertEqual(n.display, "example.com/path?x=1")

    def test_ip_literal_passes_through(self) -> None:
        n = normalize_target("8.8.8.8")
        self.assertEqual(n.host, "8.8.8.8")

    def test_idn_passthrough_preserved(self) -> None:
        n = normalize_target("xn--bcher-kva.example")
        self.assertEqual(n.host, "xn--bcher-kva.example")

    def test_empty_string_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_target("")

    def test_whitespace_only_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_target("   ")


# ---------------------------------------------------------------------------
# Safety rail
# ---------------------------------------------------------------------------


class SafetyTests(unittest.TestCase):
    """``assert_target_allowed`` blocklist enforcement."""

    def test_loopback_ipv4_blocked(self) -> None:
        with self.assertRaises(TargetBlockedError):
            assert_target_allowed("127.0.0.1")

    def test_rfc1918_10_blocked(self) -> None:
        with self.assertRaises(TargetBlockedError):
            assert_target_allowed("10.1.2.3")

    def test_rfc1918_172_16_blocked(self) -> None:
        with self.assertRaises(TargetBlockedError):
            assert_target_allowed("172.16.5.4")

    def test_rfc1918_192_168_blocked(self) -> None:
        with self.assertRaises(TargetBlockedError):
            assert_target_allowed("192.168.1.1")

    def test_cgnat_blocked(self) -> None:
        with self.assertRaises(TargetBlockedError):
            assert_target_allowed("100.64.1.1")

    def test_link_local_ipv4_blocked(self) -> None:
        with self.assertRaises(TargetBlockedError):
            assert_target_allowed("169.254.169.254")

    def test_loopback_ipv6_blocked(self) -> None:
        with self.assertRaises(TargetBlockedError):
            assert_target_allowed("::1")

    def test_link_local_ipv6_blocked(self) -> None:
        with self.assertRaises(TargetBlockedError):
            assert_target_allowed("fe80::1")

    def test_reserved_ipv4_blocked(self) -> None:
        # 240.0.0.0/4 is in the explicit blocklist.
        with self.assertRaises(TargetBlockedError):
            assert_target_allowed("240.0.0.1")

    def test_multicast_not_blocked(self) -> None:
        # 224.0.0.0/4 is NOT in the current explicit blocklist. This is
        # a known gap — pinning it so a future tightening is intentional.
        # The caller (orchestrator) is responsible for not running
        # multicast targets in the first place.
        try:
            assert_target_allowed("224.0.0.1")
        except TargetBlockedError as exc:  # pragma: no cover
            self.fail(f"multicast unexpectedly blocked: {exc}")

    def test_public_ipv4_allowed(self) -> None:
        # Must not raise.
        assert_target_allowed("8.8.8.8")

    def test_hostname_allowed(self) -> None:
        # Hostnames without an IP interpretation are passed through; the
        # safety rail does NOT do DNS lookups.
        assert_target_allowed("example.com")

    def test_hostname_with_subdomain_allowed(self) -> None:
        assert_target_allowed("api.github.com")

    def test_error_message_includes_host(self) -> None:
        with self.assertRaises(TargetBlockedError) as ctx:
            assert_target_allowed("127.0.0.1")
        self.assertIn("127.0.0.1", str(ctx.exception))

    def test_error_message_mentions_reason(self) -> None:
        with self.assertRaises(TargetBlockedError) as ctx:
            assert_target_allowed("10.0.0.1")
        self.assertIn("not globally routable", str(ctx.exception))


# ---------------------------------------------------------------------------
# DNS
# ---------------------------------------------------------------------------


class DNSResolveTests(unittest.TestCase):
    """``dns_lookup.resolve`` against a mocked ``getaddrinfo``."""

    def test_resolve_returns_ipv4_and_ipv6(self) -> None:
        fake_records = [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0)),
            (socket.AF_INET6, socket.SOCK_STREAM, 0, "",
             ("2606:2800:220:1:248:1893:25c8:1946", 0, 0, 0)),
        ]
        with mock.patch.object(
            dns_lookup.socket, "getaddrinfo", return_value=fake_records
        ):
            result = dns_lookup.resolve("example.com")
        self.assertTrue(result.ok)
        self.assertEqual(result.ipv4, ["93.184.216.34"])
        self.assertEqual(
            result.ipv6,
            ["2606:2800:220:1:248:1893:25c8:1946"],
        )
        self.assertIsNone(result.error)

    def test_resolve_only_ipv4(self) -> None:
        fake_records = [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("8.8.8.8", 0)),
        ]
        with mock.patch.object(
            dns_lookup.socket, "getaddrinfo", return_value=fake_records
        ):
            result = dns_lookup.resolve("example.com")
        self.assertTrue(result.ok)
        self.assertEqual(result.ipv4, ["8.8.8.8"])
        self.assertEqual(result.ipv6, [])

    def test_resolve_empty_raises(self) -> None:
        with mock.patch.object(
            dns_lookup.socket, "getaddrinfo", return_value=[]
        ):
            with self.assertRaises(dns_lookup.DNSError):
                dns_lookup.resolve("nx.example.invalid")

    def test_resolve_dedupes_ipv4(self) -> None:
        fake_records = [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("1.1.1.1", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("1.1.1.1", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("1.1.1.1", 0)),
        ]
        with mock.patch.object(
            dns_lookup.socket, "getaddrinfo", return_value=fake_records
        ):
            result = dns_lookup.resolve("example.com")
        self.assertEqual(result.ipv4, ["1.1.1.1"])

    def test_resolve_sorts_ipv4(self) -> None:
        fake_records = [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("9.9.9.9", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("1.1.1.1", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("8.8.8.8", 0)),
        ]
        with mock.patch.object(
            dns_lookup.socket, "getaddrinfo", return_value=fake_records
        ):
            result = dns_lookup.resolve("example.com")
        self.assertEqual(result.ipv4, ["1.1.1.1", "8.8.8.8", "9.9.9.9"])

    def test_resolve_propagates_gaierror(self) -> None:
        with mock.patch.object(
            dns_lookup.socket,
            "getaddrinfo",
            side_effect=socket.gaierror("no such host"),
        ):
            with self.assertRaises(dns_lookup.DNSError):
                dns_lookup.resolve("nx.example.invalid")

    def test_dnsresult_ok_property(self) -> None:
        empty = dns_lookup.DNSResult(ipv4=[], ipv6=[], error="x")
        self.assertFalse(empty.ok)
        full = dns_lookup.DNSResult(ipv4=["1.1.1.1"], ipv6=[], error=None)
        self.assertTrue(full.ok)


# ---------------------------------------------------------------------------
# urlinfo (in-process HTTP probe)
# ---------------------------------------------------------------------------


def _make_http_response(
    *,
    final_url: str,
    status: int,
    headers: Optional[dict] = None,
    body: bytes = b"",
) -> mock.MagicMock:
    """Build a ``urllib.request.urlopen`` mock for the in-process probe.

    Mirrors the attribute surface of an ``http.client.HTTPResponse``:
    ``.status``, ``.getcode()``, ``.geturl()`` -> ``final_url``,
    ``.headers`` (mapping), and ``.read(n)`` -> ``body``. The MagicMock
    context-manager protocol is also wired up so ``with opener.open(...)``
    works under :func:`urllib.request.build_opener`.
    """
    headers = headers or {}
    fake_response = mock.MagicMock()
    fake_response.status = status
    fake_response.getcode.return_value = status
    fake_response.geturl.return_value = final_url
    # ``urllib.request.urlopen`` returns an ``addinfourl`` whose
    # ``.url`` attribute is the final URL after redirects. Set it
    # directly so production code that prefers ``resp.url`` over
    # ``resp.geturl()`` still sees a real string.
    fake_response.url = final_url
    # MagicMock's ``headers`` is a MagicMock; ``headers.get`` returns
    # another MagicMock by default, which our ``_coerce_str`` would
    # str()-ify into something like ``"<MagicMock id=...>"``. Build a
    # real dict-backed mock so ``headers.get(name)`` returns either
    # the configured value or ``None``.
    fake_response.headers = headers
    fake_response.read.return_value = body
    fake_response.__enter__.return_value = fake_response
    fake_response.__exit__.return_value = False
    return fake_response


class UrlInfoProbeTests(unittest.TestCase):
    """``urlinfo.probe`` against a mocked in-process ``urlopen``.

    Phase 15 (PR-E rewire): we no longer proxy through ``urlinfo.io``.
    The probe hits the target URL directly, so the mocks below model
    an HTTP response from the *target*, not from a third-party
    aggregator.
    """

    def test_probe_happy_path(self) -> None:
        # HEAD response: 200 + HTML content-type -> the probe
        # follows up with a bounded GET for the title scan.
        head_resp = _make_http_response(
            final_url="https://example.com/",
            status=200,
            headers={
                "Server": "ECS (sec/9704)",
                "Content-Type": "text/html; charset=UTF-8",
                "Content-Length": "1256",
            },
            body=b"",
        )
        body_resp = _make_http_response(
            final_url="https://example.com/",
            status=200,
            headers={"Content-Type": "text/html; charset=UTF-8"},
            body=(
                b"<!doctype html><html><head>"
                b"<title>Example Domain</title></head>"
                b"<body>hi</body></html>"
            ),
        )
        with mock.patch("urllib.request.build_opener") as build_opener:
            opener = mock.MagicMock()
            opener.open.side_effect = [head_resp, body_resp]
            build_opener.return_value = opener
            result = urlinfo.probe("https://example.com/")
        self.assertEqual(result.requested_url, "https://example.com/")
        self.assertEqual(result.final_url, "https://example.com/")
        self.assertEqual(result.http_status, 200)
        self.assertEqual(result.title, "Example Domain")
        self.assertEqual(result.server, "ECS (sec/9704)")
        self.assertEqual(
            result.content_type, "text/html; charset=UTF-8"
        )
        self.assertEqual(result.content_length, 1256)
        self.assertFalse(result.redirected)

    def test_redirected_property(self) -> None:
        resp = _make_http_response(
            final_url="https://www.example.com/",
            status=200,
            headers={"Server": "ECS", "Content-Length": "0"},
            body=b"",
        )
        with mock.patch("urllib.request.build_opener") as build_opener:
            opener = mock.MagicMock()
            opener.open.return_value = resp
            build_opener.return_value = opener
            result = urlinfo.probe("https://example.com/")
        self.assertTrue(result.redirected)

    def test_probe_handles_missing_fields(self) -> None:
        # No headers, no body: all optional fields should fall back
        # to "" / 0, and status should be 0 (no response seen).
        resp = _make_http_response(
            final_url="https://example.com/",
            status=200,
            headers={},
            body=b"",
        )
        with mock.patch("urllib.request.build_opener") as build_opener:
            opener = mock.MagicMock()
            opener.open.return_value = resp
            build_opener.return_value = opener
            result = urlinfo.probe("https://example.com/")
        self.assertEqual(result.title, "")
        self.assertEqual(result.server, "")
        self.assertEqual(result.content_type, "")
        self.assertEqual(result.content_length, 0)

    def test_probe_handles_404(self) -> None:
        # The target returns 404. We do not raise; we record the
        # real status code in the result so the report can show it.
        # Simulate urllib raising HTTPError on the initial call, then
        # the GET fallback also returning 404.
        err = urllib.error.HTTPError(
            url="https://nx.invalid/",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=None,
        )
        resp = _make_http_response(
            final_url="https://nx.invalid/",
            status=404,
            headers={"Content-Length": "0"},
            body=b"",
        )
        with mock.patch("urllib.request.build_opener") as build_opener:
            opener = mock.MagicMock()
            opener.open.side_effect = [err, resp]
            build_opener.return_value = opener
            with self.assertRaises(urlinfo.URLInfoError):
                urlinfo.probe("https://nx.invalid/")

    def test_probe_network_error_raises(self) -> None:
        with mock.patch("urllib.request.build_opener") as build_opener:
            opener = mock.MagicMock()
            opener.open.side_effect = urllib_error()
            build_opener.return_value = opener
            with self.assertRaises(urlinfo.URLInfoError):
                urlinfo.probe("https://nx.invalid/")

    def test_probe_head_405_falls_back_to_get(self) -> None:
        # Some servers reject HEAD with 405. The probe should retry
        # with GET and still extract the title from the body.
        err = urllib.error.HTTPError(
            url="https://example.com/",
            code=405,
            msg="Method Not Allowed",
            hdrs={},
            fp=None,
        )
        resp = _make_http_response(
            final_url="https://example.com/",
            status=200,
            headers={
                "Server": "nginx/1.25",
                "Content-Type": "text/html",
                "Content-Length": "120",
            },
            body=b"<html><head><title>Heads-up</title></head></html>",
        )
        with mock.patch("urllib.request.build_opener") as build_opener:
            opener = mock.MagicMock()
            opener.open.side_effect = [err, resp]
            build_opener.return_value = opener
            result = urlinfo.probe("https://example.com/")
        self.assertEqual(result.http_status, 200)
        self.assertEqual(result.title, "Heads-up")
        self.assertEqual(opener.open.call_count, 2)

    def test_probe_no_title_in_body(self) -> None:
        # Non-HTML body -> empty title, no error. (Replaces the
        # pre-rewire "invalid JSON" test: we no longer parse JSON.)
        resp = _make_http_response(
            final_url="https://example.com/feed",
            status=200,
            headers={"Content-Type": "application/atom+xml"},
            body=b"<feed>...</feed>",
        )
        with mock.patch("urllib.request.build_opener") as build_opener:
            opener = mock.MagicMock()
            opener.open.return_value = resp
            build_opener.return_value = opener
            result = urlinfo.probe("https://example.com/feed")
        self.assertEqual(result.title, "")
        self.assertEqual(result.content_type, "application/atom+xml")

    def test_probe_content_length_as_string_coerced(self) -> None:
        # urllib headers are str-valued; the dataclass field is int.
        resp = _make_http_response(
            final_url="https://example.com/",
            status=200,
            headers={"Content-Length": "1256"},
            body=b"",
        )
        with mock.patch("urllib.request.build_opener") as build_opener:
            opener = mock.MagicMock()
            opener.open.return_value = resp
            build_opener.return_value = opener
            result = urlinfo.probe("https://example.com/")
        self.assertEqual(result.content_length, 1256)

    def test_probe_bare_host_gets_https_scheme(self) -> None:
        resp = _make_http_response(
            final_url="https://example.com/",
            status=200,
            headers={},
            body=b"",
        )
        with mock.patch("urllib.request.build_opener") as build_opener:
            opener = mock.MagicMock()
            opener.open.return_value = resp
            build_opener.return_value = opener
            result = urlinfo.probe("example.com")
        self.assertEqual(result.requested_url, "https://example.com/")

    def test_probe_empty_url_raises(self) -> None:
        with self.assertRaises(urlinfo.URLInfoError):
            urlinfo.probe("")


def urllib_error() -> Exception:
    """Build a ``urllib.error.URLError`` without importing at module top."""
    from urllib.error import URLError
    return URLError("no network in tests")


# ---------------------------------------------------------------------------
# ipinfo.io
# ---------------------------------------------------------------------------


class IpInfoLookupTests(unittest.TestCase):
    """``ipinfo.lookup`` against a mocked ``urlopen``."""

    def test_lookup_full_payload(self) -> None:
        payload = {
            "ip": "8.8.8.8",
            "hostname": "dns.google",
            "city": "Mountain View",
            "region": "California",
            "country": "US",
            "loc": "37.4056,-122.0775",
            "org": "AS15169 Google LLC",
            "postal": "94043",
            "timezone": "America/Los_Angeles",
        }
        with mock.patch(
            "urllib.request.urlopen",
            return_value=_make_open_side_effect(
                json.dumps(payload).encode("utf-8")
            ),
        ):
            result = ipinfo.lookup("8.8.8.8")
        self.assertEqual(result.ip, "8.8.8.8")
        self.assertEqual(result.hostname, "dns.google")
        self.assertEqual(result.org, "AS15169 Google LLC")
        self.assertEqual(result.country, "US")
        self.assertEqual(result.timezone, "America/Los_Angeles")
        self.assertEqual(result.asn, "AS15169")  # extracted from org

    def test_lookup_lite_payload(self) -> None:
        # Lite endpoint omits org/hostname/etc.
        payload = {
            "ip": "8.8.8.8",
            "city": "Mountain View",
            "country": "US",
        }
        with mock.patch(
            "urllib.request.urlopen",
            return_value=_make_open_side_effect(
                json.dumps(payload).encode("utf-8")
            ),
        ):
            result = ipinfo.lookup("8.8.8.8")
        self.assertEqual(result.ip, "8.8.8.8")
        self.assertEqual(result.hostname, "")
        self.assertEqual(result.org, "")
        self.assertEqual(result.asn, "")

    def test_lookup_asn_extraction(self) -> None:
        # The asn property pulls the leading "ASxxxxx" token from org.
        result = ipinfo.IPInfoResult(
            ip="1.1.1.1",
            hostname="",
            city="",
            region="",
            country="",
            loc="",
            org="AS13335 Cloudflare, Inc.",
            postal="",
            timezone="",
            raw={},
        )
        self.assertEqual(result.asn, "AS13335")

    def test_lookup_asn_empty_when_no_org(self) -> None:
        result = ipinfo.IPInfoResult(
            ip="1.1.1.1",
            hostname="",
            city="",
            region="",
            country="",
            loc="",
            org="",
            postal="",
            timezone="",
            raw={},
        )
        self.assertEqual(result.asn, "")

    def test_lookup_network_error_raises(self) -> None:
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=urllib_error(),
        ):
            with self.assertRaises(ipinfo.IPInfoError):
                ipinfo.lookup("8.8.8.8")

    def test_lookup_invalid_json_raises(self) -> None:
        with mock.patch(
            "urllib.request.urlopen",
            return_value=_make_open_side_effect(b"<html>500</html>"),
        ):
            with self.assertRaises(ipinfo.IPInfoError):
                ipinfo.lookup("8.8.8.8")


# ---------------------------------------------------------------------------
# WHOIS
# ---------------------------------------------------------------------------


class WhoisLookupTests(unittest.TestCase):
    """``whois.lookup`` against mocked ``socket.create_connection``."""

    def test_lookup_two_hop(self) -> None:
        iana_resp = (
            b"refer: whois.verisign-grs.com\r\n"
            b"domain: EXAMPLE.COM\r\n"
            b"status: active\r\n"
        )
        registrar_resp = (
            b"Domain Name: EXAMPLE.COM\r\n"
            b"Registry Domain ID: 2336799_DOMAIN_COM-VRSN\r\n"
        )
        # Two fake sockets chained: the IANA hop first, then the registrar
        # hop. ``create_connection`` is called once per hop.
        iana_sock = _FakeSocket(iana_resp)
        registrar_sock = _FakeSocket(registrar_resp)
        with mock.patch.object(
            whois.socket,
            "create_connection",
            side_effect=[iana_sock, registrar_sock],
        ):
            result = whois.lookup("example.com")
        self.assertIn("whois.verisign-grs.com", result.registrar_server)
        self.assertIn("EXAMPLE.COM", result.iana_body)
        self.assertIn("EXAMPLE.COM", result.registrar_body)

    def test_lookup_no_refer(self) -> None:
        # IANA response with no `refer:` line — registrar_server stays empty.
        iana_resp = b"domain: EXAMPLE.COM\r\nstatus: active\r\n"
        with mock.patch.object(
            whois.socket,
            "create_connection",
            side_effect=[_FakeSocket(iana_resp)],
        ):
            result = whois.lookup("example.com")
        self.assertEqual(result.registrar_server, "")
        self.assertEqual(result.registrar_body, "")

    def test_lookup_iana_sends_query(self) -> None:
        # The query sent to IANA ends with CRLF.
        iana_resp = b"refer: whois.verisign-grs.com\r\n"
        iana_sock = _FakeSocket(iana_resp)
        with mock.patch.object(
            whois.socket,
            "create_connection",
            side_effect=[iana_sock],
        ):
            whois.lookup("example.com")
        self.assertEqual(iana_sock.sent, [b"example.com\r\n"])

    def test_lookup_iana_connection_error(self) -> None:
        from socket import error as SocketError
        with mock.patch.object(
            whois.socket,
            "create_connection",
            side_effect=SocketError("connection refused"),
        ):
            with self.assertRaises(whois.WHOISError):
                whois.lookup("example.com")

    def test_lookup_registrar_connection_error(self) -> None:
        from socket import error as SocketError
        iana_resp = b"refer: whois.verisign-grs.com\r\n"
        with mock.patch.object(
            whois.socket,
            "create_connection",
            side_effect=[
                _FakeSocket(iana_resp),
                SocketError("connection refused"),
            ],
        ):
            with self.assertRaises(whois.WHOISError):
                whois.lookup("example.com")

    def test_whoisresult_fields(self) -> None:
        # Constructor accepts the three documented fields.
        r = whois.WHOISResult(
            iana_body="iana",
            registrar_server="whois.example",
            registrar_body="registrar",
        )
        self.assertEqual(r.iana_body, "iana")
        self.assertEqual(r.registrar_server, "whois.example")
        self.assertEqual(r.registrar_body, "registrar")


# ---------------------------------------------------------------------------
# crt.sh
# ---------------------------------------------------------------------------


class CrtShLookupTests(unittest.TestCase):
    """``crt_sh.lookup`` against a mocked ``urlopen``."""

    def test_lookup_extracts_hosts_and_issuers(self) -> None:
        payload = [
            {
                "id": 1,
                "name_value": "example.com\nwww.example.com",
                "issuer_name": "DigiCert TLS RSA SHA256 2020 CA1",
            },
            {
                "id": 2,
                "name_value": "*.example.com",
                "issuer_name": "Let's Encrypt Authority X3",
            },
        ]
        with mock.patch(
            "urllib.request.urlopen",
            return_value=_make_open_side_effect(
                json.dumps(payload).encode("utf-8")
            ),
        ):
            result = crt_sh.lookup("example.com")
        self.assertEqual(result.cert_count, 2)
        self.assertIn("example.com", result.hosts)
        self.assertIn("www.example.com", result.hosts)
        self.assertIn("*.example.com", result.hosts)
        self.assertIn("DigiCert TLS RSA SHA256 2020 CA1", result.issuers)
        self.assertIn("Let's Encrypt Authority X3", result.issuers)

    def test_lookup_empty_payload(self) -> None:
        with mock.patch(
            "urllib.request.urlopen",
            return_value=_make_open_side_effect(b"[]"),
        ):
            result = crt_sh.lookup("nx.invalid")
        self.assertEqual(result.cert_count, 0)
        self.assertEqual(result.hosts, [])
        self.assertEqual(result.issuers, [])

    def test_lookup_network_error_raises(self) -> None:
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=urllib_error(),
        ):
            with self.assertRaises(crt_sh.CrtShError):
                crt_sh.lookup("nx.invalid")

    def test_lookup_invalid_json_raises(self) -> None:
        with mock.patch(
            "urllib.request.urlopen",
            return_value=_make_open_side_effect(b"<html>500</html>"),
        ):
            with self.assertRaises(crt_sh.CrtShError):
                crt_sh.lookup("nx.invalid")

    def test_lookup_skips_none_name_value(self) -> None:
        # crt.sh returns None for missing name_value; the parser must
        # tolerate it without crashing.
        payload = [
            {"id": 1, "name_value": None, "issuer_name": "DigiCert"},
            {"id": 2, "name_value": "example.com", "issuer_name": None},
        ]
        with mock.patch(
            "urllib.request.urlopen",
            return_value=_make_open_side_effect(
                json.dumps(payload).encode("utf-8")
            ),
        ):
            result = crt_sh.lookup("example.com")
        self.assertIn("example.com", result.hosts)
        self.assertEqual(result.issuers, ["DigiCert"])

    def test_lookup_hosts_are_lowercased_and_sorted(self) -> None:
        payload = [
            {"id": 1, "name_value": "WWW.Example.COM\nb.example.com",
             "issuer_name": "X"},
            {"id": 2, "name_value": "a.example.com", "issuer_name": "X"},
        ]
        with mock.patch(
            "urllib.request.urlopen",
            return_value=_make_open_side_effect(
                json.dumps(payload).encode("utf-8")
            ),
        ):
            result = crt_sh.lookup("example.com")
        self.assertEqual(
            result.hosts,
            ["a.example.com", "b.example.com", "www.example.com"],
        )


class CrtShDisabledTests(unittest.TestCase):
    """``crt_sh.disabled_result`` synthetic helper.

    When ``RECON_CRT_SH_ENABLED`` is false, the orchestrator must not
    hit the network at all. ``disabled_result`` produces a well-formed
    ``CrtShResult`` from a manually-curated fallback list so the
    recon report still renders something useful.
    """

    def test_disabled_no_fallback_returns_empty(self) -> None:
        result = crt_sh.disabled_result("example.com")
        self.assertEqual(result.hosts, ())
        self.assertEqual(result.cert_count, 0)
        self.assertEqual(result.issuers, ())
        self.assertEqual(result.raw, [])

    def test_disabled_fallback_lowercases_strips_wildcards(self) -> None:
        result = crt_sh.disabled_result(
            "example.com",
            fallback_hosts=[
                "API.Example.com",
                " *.example.com ",
                "mail.example.com",
            ],
        )
        self.assertEqual(
            result.hosts,
            ("api.example.com", "example.com", "mail.example.com"),
        )
        self.assertEqual(result.cert_count, 0)
        self.assertEqual(result.issuers, ())

    def test_disabled_fallback_dedupes_and_drops_dotsless(self) -> None:
        result = crt_sh.disabled_result(
            "example.com",
            fallback_hosts=[
                "a.example.com",
                "A.example.com",  # dup (case-insensitive)
                "localhost",      # no dot -> dropped
                "",
                "   ",
                "b.example.com",
            ],
        )
        self.assertEqual(result.hosts, ("a.example.com", "b.example.com"))

    def test_disabled_fallback_ignores_non_strings(self) -> None:
        result = crt_sh.disabled_result(
            "example.com",
            fallback_hosts=["ok.example.com", 42, None, "z.example.com"],
        )
        self.assertEqual(result.hosts, ("ok.example.com", "z.example.com"))

    def test_disabled_returns_frozen_style_tuple(self) -> None:
        # CrtShResult.hosts is documented as tuple; confirm it round-trips
        # through equality comparison without coercion surprises.
        result = crt_sh.disabled_result(
            "example.com", fallback_hosts=["x.example.com"]
        )
        self.assertIsInstance(result.hosts, tuple)
        self.assertEqual(result, crt_sh.disabled_result(
            "example.com", fallback_hosts=["x.example.com"]
        ))


class CrtShConfigParserTests(unittest.TestCase):
    """The ``_parse_bool_env`` and ``_parse_fallback_hosts`` helpers.

    These parse the env-var knobs that gate the crt.sh kill-switch and
    the manual subdomain fallback. Pinning the vocabulary protects
    downstream config (``.env``, ``.env.example``) from silent breakage.
    """

    def test_parse_bool_truthy_vocabulary(self) -> None:
        for token in ("1", "true", "TRUE", "True", "yes", "YES", "on", "On"):
            with mock.patch.dict(os.environ, {"X_FLAG": token}, clear=False):
                self.assertIs(
                    config._parse_bool_env("X_FLAG", False), True,
                    msg=f"token={token!r}",
                )

    def test_parse_bool_falsy_vocabulary(self) -> None:
        for token in ("0", "false", "FALSE", "no", "NO", "off", "OFF"):
            with mock.patch.dict(os.environ, {"X_FLAG": token}, clear=False):
                self.assertIs(
                    config._parse_bool_env("X_FLAG", True), False,
                    msg=f"token={token!r}",
                )

    def test_parse_bool_unknown_returns_default(self) -> None:
        # Empty / whitespace / nonsense -> default (not crash, not False).
        with mock.patch.dict(os.environ, {"X_FLAG": ""}, clear=False):
            self.assertIs(config._parse_bool_env("X_FLAG", True), True)
            self.assertIs(config._parse_bool_env("X_FLAG", False), False)
        with mock.patch.dict(os.environ, {"X_FLAG": "maybe"}, clear=False):
            self.assertIs(config._parse_bool_env("X_FLAG", True), True)

    def test_parse_bool_unset_returns_default(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "X_FLAG"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertIs(config._parse_bool_env("X_FLAG", True), True)
            self.assertIs(config._parse_bool_env("X_FLAG", False), False)

    def test_parse_fallback_hosts_dedupes_lowercases(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"RECON_FALLBACK_SUBDOMAINS":
             "api.example.com, MAIL.example.com, api.example.com, "
             "*.blog.example.com, localhost, , web.example.com"},
            clear=False,
        ):
            self.assertEqual(
                config._parse_fallback_hosts(),
                (
                    "api.example.com",
                    "blog.example.com",  # wildcard prefix stripped
                    "mail.example.com",
                    "web.example.com",
                ),
            )

    def test_parse_fallback_hosts_empty(self) -> None:
        with mock.patch.dict(
            os.environ, {"RECON_FALLBACK_SUBDOMAINS": ""}, clear=False
        ):
            self.assertEqual(config._parse_fallback_hosts(), ())

    def test_parse_fallback_hosts_unset(self) -> None:
        env = {k: v for k, v in os.environ.items()
               if k != "RECON_FALLBACK_SUBDOMAINS"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(config._parse_fallback_hosts(), ())


class CrtShOrchestratorDisabledTests(unittest.TestCase):
    """``_run_crt_sh`` short-circuits when the kill-switch is engaged.

    The orchestrator must NOT call ``crt_sh.lookup`` (no network) and
    must report ``ok=True`` so the renderer can show the soft-empty
    copy instead of a scary error block. The ``ToolResult.error``
    field is populated with :data:`crt_sh.DISABLED_SENTINEL` so the
    renderer can tell the kill-switch path apart from a genuine
    "crt.sh returned no certificates for this host" — the two used
    to look identical from the report alone.
    """

    def _make_orchestrator_doubles(self) -> dict:
        """Stub every transport; only crt_sh.lookup must NOT be invoked."""
        return {
            "_run_dns": mock.MagicMock(return_value=ToolResult(
                tool="dns", ok=True, value=_fake_dns_ok(),
                error=None, duration_ms=1,
            )),
            "_run_ipinfo": mock.MagicMock(return_value=ToolResult(
                tool="ipinfo", ok=True, value=None,
                error=None, duration_ms=1,
            )),
            "_run_whois": mock.MagicMock(return_value=ToolResult(
                tool="whois", ok=True, value=None,
                error=None, duration_ms=1,
            )),
            "_run_urlinfo": mock.MagicMock(return_value=ToolResult(
                tool="urlinfo", ok=True, value=None,
                error=None, duration_ms=1,
            )),
            # crt_sh.lookup should never run; if it does, blow up loudly.
            "crt_sh.lookup": mock.MagicMock(
                side_effect=AssertionError(
                    "crt_sh.lookup must NOT be called when disabled"
                ),
            ),
        }

    def test_orchestrator_skips_lookup_when_disabled(self) -> None:
        doubles = self._make_orchestrator_doubles()
        with mock.patch.dict(os.environ,
                             {"RECON_CRT_SH_ENABLED": "false"},
                             clear=False), \
             mock.patch.multiple("app.recon.orchestrator", **{
                 k: v for k, v in doubles.items()
                 if k.startswith("_run_")
             }), \
             mock.patch.object(crt_sh, "lookup", doubles["crt_sh.lookup"]), \
             mock.patch.object(config, "RECON_CRT_SH_ENABLED", False), \
             mock.patch.object(config, "RECON_FALLBACK_SUBDOMAINS", ()):
            result = orchestrator._run_crt_sh("example.com")
        self.assertEqual(result.tool, "crt_sh")
        self.assertTrue(result.ok)
        # The orchestrator tags the disabled ToolResult with the
        # sentinel so the renderer can render the "crt.sh disabled"
        # copy (with the env-var hint) instead of the "crt.sh
        # returned no certificates" copy. Pinned here so the
        # sentinel and the renderer never drift apart.
        self.assertEqual(result.error, crt_sh.DISABLED_SENTINEL)
        self.assertIsInstance(result.value, crt_sh.CrtShResult)
        self.assertEqual(result.value.hosts, ())
        # lookup must not have been called — explicit assertion via the
        # side_effect would already have raised, but double-check the
        # call count for clarity.
        doubles["crt_sh.lookup"].assert_not_called()

    def test_orchestrator_returns_fallback_hosts_when_disabled(self) -> None:
        doubles = self._make_orchestrator_doubles()
        fallback = ("api.example.com", "mail.example.com")
        with mock.patch.dict(os.environ,
                             {"RECON_CRT_SH_ENABLED": "false"},
                             clear=False), \
             mock.patch.multiple("app.recon.orchestrator", **{
                 k: v for k, v in doubles.items()
                 if k.startswith("_run_")
             }), \
             mock.patch.object(crt_sh, "lookup", doubles["crt_sh.lookup"]), \
             mock.patch.object(config, "RECON_CRT_SH_ENABLED", False), \
             mock.patch.object(config, "RECON_FALLBACK_SUBDOMAINS", fallback):
            result = orchestrator._run_crt_sh("example.com")
        self.assertTrue(result.ok)
        # Even with a populated fallback, the disabled ToolResult
        # still carries the sentinel — the sentinel is the
        # "this was short-circuited" signal, not the "this has no
        # hosts" signal.
        self.assertEqual(result.error, crt_sh.DISABLED_SENTINEL)
        self.assertEqual(result.value.hosts, fallback)
        doubles["crt_sh.lookup"].assert_not_called()

    def test_orchestrator_calls_lookup_when_enabled(self) -> None:
        """The inverse: when the kill-switch is OFF, lookup MUST run.

        Pinned so a future refactor can't accidentally disable crt.sh
        for everyone by default.
        """
        fake_result = crt_sh.CrtShResult(
            hosts=("x.example.com",), cert_count=1,
            issuers=(), raw=[],
        )
        with mock.patch.object(
            crt_sh, "lookup", return_value=fake_result
        ), mock.patch.object(config, "RECON_CRT_SH_ENABLED", True), \
             mock.patch.object(config, "RECON_FALLBACK_SUBDOMAINS", ()):
            result = orchestrator._run_crt_sh("example.com")
        self.assertTrue(result.ok)
        self.assertEqual(result.value, fake_result)


class CrtShReportSoftEmptyTests(unittest.TestCase):
    """The renderer must show soft-empty copy when crt.sh is disabled."""

    def test_disabled_no_fallback_renders_kill_switch_copy(self) -> None:
        """``crt_sh.DISABLED_SENTINEL`` + empty hosts -> kill-switch copy.

        This is the path the user complained about: the report used to
        collapse to a single one-line "crt.sh skipped" notice. The new
        copy is richer — it explains *why* the lookup was skipped (the
        env-var kill-switch) and *what* the operator can do about it
        (flip ``RECON_CRT_SH_ENABLED`` or populate
        ``RECON_FALLBACK_SUBDOMAINS``).
        """
        synthetic = crt_sh.disabled_result("example.com")
        tr = ToolResult(
            tool="crt_sh", ok=True, value=synthetic,
            error=crt_sh.DISABLED_SENTINEL, duration_ms=0,
        )
        out = report._format_crt_sh(tr)
        # The new copy must surface both knobs the operator can flip.
        self.assertIn("RECON_CRT_SH_ENABLED", out)
        self.assertIn("RECON_FALLBACK_SUBDOMAINS", out)
        # Old one-liner is gone.
        self.assertNotIn("crt.sh skipped", out)
        # No scary error block.
        self.assertNotIn("**Error:**", out)

    def test_disabled_with_fallback_renders_hosts_and_hint(self) -> None:
        """``crt_sh.DISABLED_SENTINEL`` + populated hosts -> hosts + hint.

        This is the graceful-degrade case the kill-switch was designed
        for: the orchestrator put a fallback list in
        ``RECON_FALLBACK_SUBDOMAINS``, the synthetic ``CrtShResult``
        carries those hosts, and the renderer surfaces them as a real
        list (not a "skipped" one-liner) while still telling the
        operator crt.sh was offline.
        """
        synthetic = crt_sh.disabled_result(
            "example.com",
            fallback_hosts=["api.example.com", "mail.example.com"],
        )
        tr = ToolResult(
            tool="crt_sh", ok=True, value=synthetic,
            error=crt_sh.DISABLED_SENTINEL, duration_ms=12,
        )
        out = report._format_crt_sh(tr)
        # The fallback hosts still render.
        self.assertIn("api.example.com", out)
        self.assertIn("mail.example.com", out)
        # And the operator still knows crt.sh was the source.
        self.assertIn("RECON_CRT_SH_ENABLED", out)

    def test_empty_response_distinguishable_from_disabled(self) -> None:
        """ok=True, error=None, no hosts -> "no certificates" copy.

        Previously the "genuinely empty" response and the "kill-switch"
        response were collapsed into the same one-line copy. After the
        sentinel-tag refactor the renderer must distinguish them: the
        "no certificates" copy must NOT mention the env-var knob, and
        the kill-switch copy must. Pinned here so a future renderer
        refactor can't silently merge them again.
        """
        synthetic = crt_sh.CrtShResult(
            hosts=(), cert_count=0, issuers=(), raw=[],
        )
        tr = ToolResult(
            tool="crt_sh", ok=True, value=synthetic, error=None,
            duration_ms=5,
        )
        out = report._format_crt_sh(tr)
        # Genuinely-empty copy: short, neutral, no env-var hint.
        self.assertIn("no certificates", out)
        self.assertNotIn("RECON_CRT_SH_ENABLED", out)
        self.assertNotIn("crt.sh skipped", out)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class OrchestratorTests(unittest.TestCase):
    """End-to-end ``run_recon`` with every transport stubbed."""

    def test_happy_path_all_tools_ok(self) -> None:
        with mock.patch.multiple(
            "app.recon.orchestrator",
            _run_dns=mock.MagicMock(
                return_value=ToolResult(
                    tool="dns", ok=True, value=_fake_dns_ok(),
                    error=None, duration_ms=3,
                )
            ),
            _run_ipinfo=mock.MagicMock(
                return_value=ToolResult(
                    tool="ipinfo", ok=True, value=_fake_ipinfo_ok(),
                    error=None, duration_ms=12,
                )
            ),
            _run_urlinfo=mock.MagicMock(
                return_value=ToolResult(
                    tool="urlinfo", ok=True,
                    value=_fake_urlinfo_ok(), error=None, duration_ms=11,
                )
            ),
            _run_whois=mock.MagicMock(
                return_value=ToolResult(
                    tool="whois", ok=True, value=_fake_whois_ok(),
                    error=None, duration_ms=9,
                )
            ),
            _run_crt_sh=mock.MagicMock(
                return_value=ToolResult(
                    tool="crt_sh", ok=True, value=_fake_crt_ok(),
                    error=None, duration_ms=7,
                )
            ),
        ):
            report = run_recon("example.com", scope_token="lab")
        self.assertEqual(report.host, "example.com")
        self.assertEqual(report.target, "example.com")
        self.assertEqual(report.scope_token, "lab")
        self.assertTrue(report.all_ok())
        for tr in report.tool_results():
            self.assertTrue(tr.ok, msg=f"{tr.tool} should be ok")

    def test_partial_failure_recorded(self) -> None:
        with mock.patch.multiple(
            "app.recon.orchestrator",
            _run_dns=mock.MagicMock(
                return_value=ToolResult(
                    tool="dns", ok=True, value=_fake_dns_ok(),
                    error=None, duration_ms=3,
                )
            ),
            _run_ipinfo=mock.MagicMock(
                return_value=ToolResult(
                    tool="ipinfo", ok=False, value=None,
                    error="ipinfo outage", duration_ms=12,
                )
            ),
            _run_urlinfo=mock.MagicMock(
                return_value=ToolResult(
                    tool="urlinfo", ok=True,
                    value=_fake_urlinfo_ok(), error=None, duration_ms=11,
                )
            ),
            _run_whois=mock.MagicMock(
                return_value=ToolResult(
                    tool="whois", ok=True, value=_fake_whois_ok(),
                    error=None, duration_ms=9,
                )
            ),
            _run_crt_sh=mock.MagicMock(
                return_value=ToolResult(
                    tool="crt_sh", ok=True, value=_fake_crt_ok(),
                    error=None, duration_ms=7,
                )
            ),
        ):
            report = run_recon("example.com", scope_token="lab")
        self.assertFalse(report.all_ok())
        self.assertTrue(report.dns.ok)
        self.assertFalse(report.ipinfo.ok)
        self.assertEqual(report.ipinfo.error, "ipinfo outage")

    def test_invalid_scope_token_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            run_recon("example.com", scope_token="does_not_exist")
        self.assertIn("does_not_exist", str(ctx.exception))

    def test_empty_scope_token_raises(self) -> None:
        with self.assertRaises(ValueError):
            run_recon("example.com", scope_token="")

    def test_blocked_target_raises(self) -> None:
        # Loopback targets are blocked before any tool is invoked.
        with self.assertRaises(TargetBlockedError):
            run_recon("127.0.0.1", scope_token="lab")

    def test_blocked_target_does_not_call_transports(self) -> None:
        with mock.patch.multiple(
            "app.recon.orchestrator",
            _run_dns=mock.MagicMock(),
            _run_ipinfo=mock.MagicMock(),
            _run_urlinfo=mock.MagicMock(),
            _run_whois=mock.MagicMock(),
            _run_crt_sh=mock.MagicMock(),
        ) as patches:
            with self.assertRaises(TargetBlockedError):
                run_recon("10.0.0.1", scope_token="lab")
        for name, stub in patches.items():
            stub.assert_not_called()  # type: ignore[union-attr]

    def test_empty_target_raises(self) -> None:
        with self.assertRaises(ValueError):
            run_recon("", scope_token="lab")


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


class RenderMarkdownTests(unittest.TestCase):
    """``render_report_markdown`` produces a readable doc."""

    def test_contains_target_section(self) -> None:
        md = render_report_markdown(_sample_report())
        self.assertIn("## Target", md)
        self.assertIn("example.com", md)

    def test_contains_each_tool_section(self) -> None:
        md = render_report_markdown(_sample_report())
        for header in ("DNS", "ipinfo", "urlinfo", "WHOIS", "crt.sh"):
            self.assertIn(header, md)

    def test_contains_summary_table(self) -> None:
        md = render_report_markdown(_sample_report())
        self.assertIn("Summary", md)
        self.assertIn("| dns |", md)
        self.assertIn("| ipinfo |", md)
        self.assertIn("| urlinfo |", md)
        self.assertIn("| whois |", md)
        self.assertIn("| crt_sh |", md)

    def test_error_tool_renders_error_block(self) -> None:
        report = _sample_report()
        report = dataclasses.replace(
            report,
            ipinfo=ToolResult(
                tool="ipinfo", ok=False, value=None,
                error="ipinfo rate limit", duration_ms=12,
            ),
        )
        md = render_report_markdown(report)
        self.assertIn("ipinfo rate limit", md)
        self.assertIn("❌", md)

    def test_redirected_url_renders_marker(self) -> None:
        report = _sample_report()
        report = dataclasses.replace(
            report,
            urlinfo=ToolResult(
                tool="urlinfo", ok=True,
                value=urlinfo.URLInfoResult(
                    requested_url="https://example.com/",
                    final_url="https://www.example.com/",
                    http_status=200,
                    title="Example",
                    server="ECS",
                    content_type="text/html",
                    content_length=1256,
                ),
                error=None, duration_ms=11,
            ),
        )
        md = render_report_markdown(report)
        self.assertIn("Redirected", md)

    def test_crt_hosts_capped_at_50(self) -> None:
        many_hosts = [f"h{i}.example.com" for i in range(60)]
        report = _sample_report()
        report = dataclasses.replace(
            report,
            crt_sh=ToolResult(
                tool="crt_sh", ok=True,
                value=crt_sh.CrtShResult(
                    hosts=sorted(many_hosts),
                    cert_count=60,
                    issuers=["X"],
                    raw=[],
                ),
                error=None, duration_ms=7,
            ),
        )
        md = render_report_markdown(report)
        self.assertIn("… 10 more not shown", md)


class RenderJsonTests(unittest.TestCase):
    """``render_report_json`` produces stable JSON."""

    def test_json_is_valid(self) -> None:
        out = render_report_json(_sample_report())
        data = json.loads(out)
        self.assertIn("target", data)
        self.assertIn("display", data)
        self.assertIn("host", data)
        self.assertIn("scope_token", data)
        self.assertIn("total_ms", data)
        self.assertIn("tools", data)
        self.assertEqual(len(data["tools"]), 5)

    def test_json_tool_keys(self) -> None:
        out = render_report_json(_sample_report())
        data = json.loads(out)
        tool_names = [t["tool"] for t in data["tools"]]
        self.assertEqual(
            tool_names, ["dns", "ipinfo", "urlinfo", "whois", "crt_sh"]
        )

    def test_json_includes_typed_values(self) -> None:
        out = render_report_json(_sample_report())
        data = json.loads(out)
        by_tool = {t["tool"]: t for t in data["tools"]}
        self.assertEqual(by_tool["dns"]["value"]["ipv4"], ["93.184.216.34"])
        self.assertEqual(by_tool["ipinfo"]["value"]["ip"], "93.184.216.34")
        self.assertEqual(
            by_tool["urlinfo"]["value"]["http_status"], 200
        )
        self.assertEqual(by_tool["crt_sh"]["value"]["cert_count"], 2)

    def test_json_error_tool_value_is_null(self) -> None:
        report = _sample_report()
        report = dataclasses.replace(
            report,
            ipinfo=ToolResult(
                tool="ipinfo", ok=False, value=None,
                error="ipinfo rate limit", duration_ms=12,
            ),
        )
        out = render_report_json(report)
        data = json.loads(out)
        by_tool = {t["tool"]: t for t in data["tools"]}
        self.assertIsNone(by_tool["ipinfo"]["value"])
        self.assertEqual(by_tool["ipinfo"]["error"], "ipinfo rate limit")
        self.assertFalse(by_tool["ipinfo"]["ok"])


# ---------------------------------------------------------------------------
# Storage repo
# ---------------------------------------------------------------------------


class _TempDb(unittest.TestCase):
    """Mixin: set up a temporary DB path that each test can write to."""

    def setUp(self) -> None:
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "recon_test.sqlite"
        storage.init_db(self.db_path)
        self.chat_id = storage.create_chat(
            title="recon test", path=self.db_path
        )

    def tearDown(self) -> None:
        super().tearDown()
        with contextlib.suppress(Exception):
            self._tmp.cleanup()


class ReconStorageTests(_TempDb):
    """``app.storage`` recon repo functions."""

    def test_log_and_list(self) -> None:
        storage.log_recon_request(
            target="example.com",
            tool="dns",
            scope_token="lab",
            chat_id=self.chat_id,
            status="ok",
            duration_ms=10,
            result_excerpt="A: 1.2.3.4",
            path=self.db_path,
        )
        storage.log_recon_request(
            target="example.com",
            tool="ipinfo",
            scope_token="lab",
            chat_id=self.chat_id,
            status="error",
            duration_ms=12,
            result_excerpt="ipinfo outage",
            path=self.db_path,
        )
        rows = storage.list_recon_for_chat(self.chat_id, path=self.db_path)
        self.assertEqual(len(rows), 2)
        # Newest first.
        self.assertEqual(rows[0]["tool"], "ipinfo")
        self.assertEqual(rows[1]["tool"], "dns")
        for row in rows:
            self.assertEqual(row["target"], "example.com")
            self.assertEqual(row["chat_id"], self.chat_id)

    def test_count(self) -> None:
        self.assertEqual(
            storage.count_recon_requests(self.chat_id, path=self.db_path),
            0,
        )
        storage.log_recon_request(
            target="example.com",
            tool="dns",
            scope_token="lab",
            chat_id=self.chat_id,
            status="ok",
            duration_ms=10,
            result_excerpt="A: 1.2.3.4",
            path=self.db_path,
        )
        self.assertEqual(
            storage.count_recon_requests(self.chat_id, path=self.db_path),
            1,
        )

    def test_count_all_returns_aggregate(self) -> None:
        storage.log_recon_request(
            target="example.com",
            tool="dns",
            scope_token="lab",
            chat_id=self.chat_id,
            status="ok",
            duration_ms=10,
            result_excerpt="A: 1.2.3.4",
            path=self.db_path,
        )
        # No chat_id => aggregate over the whole table.
        self.assertEqual(
            storage.count_recon_requests(path=self.db_path),
            1,
        )

    def test_list_recon_for_chat_limit(self) -> None:
        for i in range(5):
            storage.log_recon_request(
                target=f"target-{i}.com",
                tool="dns",
                scope_token="lab",
                chat_id=self.chat_id,
                status="ok",
                duration_ms=10,
                result_excerpt="x",
                path=self.db_path,
            )
        rows = storage.list_recon_for_chat(
            self.chat_id, limit=2, path=self.db_path
        )
        self.assertEqual(len(rows), 2)

    def test_list_recon_for_chat_rejects_non_positive_limit(self) -> None:
        with self.assertRaises(ValueError):
            storage.list_recon_for_chat(
                self.chat_id, limit=0, path=self.db_path
            )

    def test_log_recon_request_returns_rowid(self) -> None:
        rowid = storage.log_recon_request(
            target="example.com",
            tool="dns",
            scope_token="lab",
            chat_id=self.chat_id,
            status="ok",
            duration_ms=10,
            result_excerpt="x",
            path=self.db_path,
        )
        self.assertIsInstance(rowid, int)
        self.assertGreater(rowid, 0)


if __name__ == "__main__":
    unittest.main()

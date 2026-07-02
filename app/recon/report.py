"""Recon report renderers.

Phase 15 — two output formats for a :class:`ReconReport`:

- :func:`render_report_markdown` — human-readable, for the Streamlit
  expander and the CLI. Uses headings, code blocks, and bullet lists.
- :func:`render_report_json` — machine-readable, for the audit log
  excerpt and any future API export. The JSON shape is stable: every
  per-tool result is a dict with the same five keys (``tool``,
  ``ok``, ``value``, ``error``, ``duration_ms``).

Renderers are pure functions of the report — they never read from
``recon_log`` or any other storage. The orchestrator builds the
report; the renderer prints it; the audit logger records it. This
separation is what lets the test suite pin the renderer output
without mocking the orchestrator.
"""

from __future__ import annotations

import json
from typing import Any

from . import crt_sh
from .crt_sh import CrtShResult
from .dns_lookup import DNSResult
from .ipinfo import IPInfoResult
from .orchestrator import ReconReport, ToolResult
from .urlinfo import URLInfoResult
from .whois import WHOISResult

#: Section headings in the markdown report. Kept as constants so the
#: test suite can pin them without copy-pasting the strings.
_H_TARGET = "## Target"
_H_DNS = "## DNS"
_H_IPINFO = "## IP info"
_H_URLINFO = "## URL info"
_H_WHOIS = "## WHOIS"
_H_CRT = "## Certificates (crt.sh)"
_H_SUMMARY = "## Summary"


def _format_dns(result: ToolResult) -> str:
    """Render the DNS tool result as a markdown block."""
    lines = [_H_DNS]
    if not result.ok or not isinstance(result.value, DNSResult):
        lines.append(f"**Error:** {result.error or 'unknown'}")
        lines.append("")
        lines.append(f"_Duration: {result.duration_ms} ms_")
        return "\n".join(lines)
    dns: DNSResult = result.value
    lines.append(f"- **IPv4:** {', '.join(dns.ipv4) or '_(none)_'}")
    lines.append(f"- **IPv6:** {', '.join(dns.ipv6) or '_(none)_'}")
    lines.append("")
    lines.append(f"_Duration: {result.duration_ms} ms_")
    return "\n".join(lines)


def _format_ipinfo(result: ToolResult) -> str:
    """Render the ipinfo tool result as a markdown block."""
    lines = [_H_IPINFO]
    if not result.ok or not isinstance(result.value, IPInfoResult):
        lines.append(f"**Error:** {result.error or 'unknown'}")
        lines.append("")
        lines.append(f"_Duration: {result.duration_ms} ms_")
        return "\n".join(lines)
    info: IPInfoResult = result.value
    lines.append(f"- **IP:** {info.ip}")
    lines.append(f"- **Hostname:** {info.hostname or '_(none)_'}")
    lines.append(f"- **City:** {info.city or '_(none)_'}")
    lines.append(f"- **Region:** {info.region or '_(none)_'}")
    lines.append(f"- **Country:** {info.country or '_(none)_'}")
    lines.append(f"- **Location:** {info.loc or '_(none)_'}")
    lines.append(f"- **Organization:** {info.org or '_(none)_'}")
    if info.asn:
        lines.append(f"- **ASN:** {info.asn}")
    lines.append(f"- **Postal:** {info.postal or '_(none)_'}")
    lines.append(f"- **Timezone:** {info.timezone or '_(none)_'}")
    lines.append("")
    lines.append(f"_Duration: {result.duration_ms} ms_")
    return "\n".join(lines)


def _format_urlinfo(result: ToolResult) -> str:
    """Render the urlinfo tool result as a markdown block."""
    lines = [_H_URLINFO]
    if not result.ok or not isinstance(result.value, URLInfoResult):
        lines.append(f"**Error:** {result.error or 'unknown'}")
        lines.append("")
        lines.append(f"_Duration: {result.duration_ms} ms_")
        return "\n".join(lines)
    info: URLInfoResult = result.value
    lines.append(f"- **Final URL:** {info.final_url}")
    lines.append(f"- **HTTP status:** {info.http_status}")
    lines.append(f"- **Title:** {info.title or '_(none)_'}")
    lines.append(f"- **Server:** {info.server or '_(none)_'}")
    lines.append(f"- **Content-Type:** {info.content_type or '_(none)_'}")
    lines.append(f"- **Content-Length:** {info.content_length}")
    if info.redirected:
        lines.append("- **Redirected:** yes")
    lines.append("")
    lines.append(f"_Duration: {result.duration_ms} ms_")
    return "\n".join(lines)


def _format_whois(result: ToolResult) -> str:
    """Render the WHOIS tool result as a markdown block.

    The full WHOIS body is included verbatim in a fenced code block
    because the registrar response contains fields (status, dates,
    contacts) that this dataclass does not enumerate and that an
    operator may want to read directly.
    """
    lines = [_H_WHOIS]
    if not result.ok or not isinstance(result.value, WHOISResult):
        lines.append(f"**Error:** {result.error or 'unknown'}")
        lines.append("")
        lines.append(f"_Duration: {result.duration_ms} ms_")
        return "\n".join(lines)
    who: WHOISResult = result.value
    lines.append(f"- **Registrar server:** "
                 f"{who.registrar_server or '_(not extracted)_'}")
    lines.append("")
    lines.append("### IANA response")
    lines.append("```")
    lines.append(who.iana_body.rstrip())
    lines.append("```")
    if who.registrar_body:
        lines.append("")
        lines.append("### Registrar response")
        lines.append("```")
        lines.append(who.registrar_body.rstrip())
        lines.append("```")
    lines.append("")
    lines.append(f"_Duration: {result.duration_ms} ms_")
    return "\n".join(lines)


def _format_crt_sh(result: ToolResult) -> str:
    """Render the crt.sh tool result as a markdown block.

    Branching logic:

    - **Hard error** (``ok=False`` or value not a
      :class:`CrtShResult`): render an error block with the
      orchestrator's reason. This is the "crt.sh returned 502"
      case where the HTTP call failed.
    - **Kill-switch** (``error == crt_sh.DISABLED_SENTINEL``):
      render the "crt.sh disabled" copy. The orchestrator tags
      every disabled-result with the sentinel, so this branch
      is unambiguous — the renderer never has to guess between
      "disabled" and "empty response". When the operator has
      populated :data:`app.config.RECON_FALLBACK_SUBDOMAINS`,
      the synthetic :class:`CrtShResult` carries a non-empty
      ``hosts`` tuple, and the renderer shows them as a real
      list so the section is useful even with crt.sh offline.
    - **Genuinely empty response** (``ok=True``, sentinel not
      set, ``cert_count=0``, no hosts, no issuers): render a
      short "crt.sh returned no certificates for this host"
      notice. This used to be collapsed into the same one-line
      "skipped" copy as the kill-switch, which made the two
      cases impossible to tell apart from the report alone.
    - **Normal result**: render the full count + issuers + hosts
      list, capped at 50 hosts.
    """
    lines = [_H_CRT]
    if not result.ok or not isinstance(result.value, CrtShResult):
        lines.append(f"**Error:** {result.error or 'unknown'}")
        lines.append("")
        lines.append(f"_Duration: {result.duration_ms} ms_")
        return "\n".join(lines)
    crt: CrtShResult = result.value
    # Kill-switch branch: the orchestrator tags every disabled
    # ToolResult with the sentinel, so we know the operator
    # flipped ``RECON_CRT_SH_ENABLED`` rather than crt.sh
    # returning an empty array. The copy is one bullet naming
    # both knobs — verbose multi-bullet explanations of "nothing
    # happened" are noise the operator does not need.
    if result.error == crt_sh.DISABLED_SENTINEL:
        lines.append(
            "- _crt.sh disabled via `.env` "
            "(`RECON_CRT_SH_ENABLED=false`); set it to `true` "
            "or populate `RECON_FALLBACK_SUBDOMAINS` to "
            "populate this section._"
        )
        if crt.hosts:
            # Operator populated RECON_FALLBACK_SUBDOMAINS, so we
            # have real content to show even though crt.sh was
            # skipped. Render it as a proper host list so the
            # section earns its keep.
            lines.append(
                f"- **Fallback hosts "
                f"(`RECON_FALLBACK_SUBDOMAINS`):** {len(crt.hosts)}"
            )
            shown = crt.hosts[:50]
            lines.append("")
            lines.append("### Hosts (first 50)")
            for host in shown:
                lines.append(f"- {host}")
            if len(crt.hosts) > 50:
                lines.append(
                    f"- _(… {len(crt.hosts) - 50} more not shown)_"
                )
        lines.append("")
        lines.append(f"_Duration: {result.duration_ms} ms_")
        return "\n".join(lines)
    if (
        crt.cert_count == 0
        and not crt.hosts
        and not crt.issuers
    ):
        # Genuinely-empty path: crt.sh was queried and returned an
        # empty array. Distinguish this from the kill-switch
        # branch above so an operator reading the report knows
        # whether to flip a switch or trust the upstream.
        lines.append(
            "- _crt.sh returned no certificates for this host._"
        )
        lines.append("")
        lines.append(f"_Duration: {result.duration_ms} ms_")
        return "\n".join(lines)
    lines.append(f"- **Certificates:** {crt.cert_count}")
    lines.append(f"- **Issuers:** {', '.join(crt.issuers) or '_(none)_'}")
    lines.append(f"- **Hosts seen:** {len(crt.hosts)}")
    if crt.hosts:
        # Cap the list at 50 to keep the report readable. The full
        # set is in the JSON output for callers who want every host.
        shown = crt.hosts[:50]
        lines.append("")
        lines.append("### Hosts (first 50)")
        for host in shown:
            lines.append(f"- {host}")
        if len(crt.hosts) > 50:
            lines.append(f"- _(… {len(crt.hosts) - 50} more not shown)_")
    lines.append("")
    lines.append(f"_Duration: {result.duration_ms} ms_")
    return "\n".join(lines)


def _format_summary(report: ReconReport) -> str:
    """Render the summary footer with totals and per-tool pass/fail."""
    lines = [_H_SUMMARY]
    lines.append(f"- **Scope:** {report.scope_token}")
    lines.append(f"- **Total duration:** {report.total_ms} ms")
    lines.append("")
    lines.append("| Tool | Status | Duration (ms) |")
    lines.append("| --- | --- | --- |")
    for r in report.tool_results():
        status = "✅ ok" if r.ok else "❌ error"
        lines.append(f"| {r.tool} | {status} | {r.duration_ms} |")
    return "\n".join(lines)


def render_report_markdown(report: ReconReport) -> str:
    """Render a :class:`ReconReport` as a markdown document.

    The output starts with a ``## Target`` section showing the
    display form and canonical host, then one section per tool,
    then a summary table at the bottom. Section order is fixed so
    the Streamlit expander renders identically every time.
    """
    parts: list[str] = [
        _H_TARGET,
        f"- **Display:** {report.display}",
        f"- **Host:** {report.host}",
        "",
        _format_dns(report.dns),
        _format_ipinfo(report.ipinfo),
        _format_urlinfo(report.urlinfo),
        _format_whois(report.whois),
        _format_crt_sh(report.crt_sh),
        _format_summary(report),
    ]
    return "\n\n".join(parts) + "\n"


def _tool_to_json(result: ToolResult) -> dict[str, Any]:
    """Convert a :class:`ToolResult` to a JSON-safe dict.

    ``value`` is a tool-specific dataclass; we delegate to its
    ``__dict__`` for the simple cases and to a dedicated mapping
    for the ones whose attributes are not all JSON-serializable
    (none today, but the indirection keeps us safe).
    """
    value: Any
    if result.value is None:
        value = None
    elif isinstance(result.value, DNSResult):
        value = {
            "ipv4": list(result.value.ipv4),
            "ipv6": list(result.value.ipv6),
            "error": result.value.error,
            "ok": result.value.ok,
        }
    elif isinstance(result.value, IPInfoResult):
        value = {
            "ip": result.value.ip,
            "hostname": result.value.hostname,
            "city": result.value.city,
            "region": result.value.region,
            "country": result.value.country,
            "loc": result.value.loc,
            "org": result.value.org,
            "asn": result.value.asn,
            "postal": result.value.postal,
            "timezone": result.value.timezone,
        }
    elif isinstance(result.value, URLInfoResult):
        value = {
            "requested_url": result.value.requested_url,
            "final_url": result.value.final_url,
            "http_status": result.value.http_status,
            "title": result.value.title,
            "server": result.value.server,
            "content_type": result.value.content_type,
            "content_length": result.value.content_length,
            "redirected": result.value.redirected,
        }
    elif isinstance(result.value, WHOISResult):
        value = {
            "registrar_server": result.value.registrar_server,
            "iana_body": result.value.iana_body,
            "registrar_body": result.value.registrar_body,
        }
    elif isinstance(result.value, CrtShResult):
        value = {
            "cert_count": result.value.cert_count,
            "hosts": list(result.value.hosts),
            "issuers": list(result.value.issuers),
        }
    else:
        # Fallback: stringify so JSON serialisation cannot fail.
        value = repr(result.value)
    return {
        "tool": result.tool,
        "ok": result.ok,
        "value": value,
        "error": result.error,
        "duration_ms": result.duration_ms,
    }


def render_report_json(report: ReconReport) -> str:
    """Render a :class:`ReconReport` as a JSON string.

    The output is a single object with ``target``, ``display``,
    ``host``, ``scope_token``, ``total_ms``, and ``tools`` (the
    list of per-tool dicts). The shape is stable across
    renderers — the Streamlit view can ``json.loads`` it for the
    audit-log excerpt.
    """
    payload = {
        "target": report.target,
        "display": report.display,
        "host": report.host,
        "scope_token": report.scope_token,
        "total_ms": report.total_ms,
        "tools": [_tool_to_json(r) for r in report.tool_results()],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
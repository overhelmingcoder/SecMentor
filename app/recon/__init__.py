"""Recon subsystem (Phase 15).

OSINT + light-touch reconnaissance for a single target (URL or domain).
The orchestrator fans the target out to five independent tools — DNS,
ipinfo, urlinfo, WHOIS, crt.sh — and renders a single structured report.

The subsystem is deliberately *read-only*. No port scans, no HTTP
fuzzing, no exploits. Every call is wrapped in a safety rail that
rejects RFC1918 / loopback / link-local / ULA / internal-TLD targets
and logs every invocation to ``recon_log``.

Public API
----------

- :func:`run_recon` — entry point. Returns a :class:`ReconReport`.
- :func:`render_report_markdown`, :func:`render_report_json` — output.
- :func:`assert_target_allowed`, :class:`TargetBlockedError` — safety.

Everything else is an implementation detail of the orchestrator and
its tool transports and is not re-exported here.
"""

from __future__ import annotations

from .normalize import (
    NormalizedTarget,
    normalize_target,
    refang,
)
from .orchestrator import (
    ReconReport,
    ToolResult,
    run_recon,
    stream_recon,
)
from .report import (
    render_report_json,
    render_report_markdown,
)
from .safety import (
    TargetBlockedError,
    assert_target_allowed,
)

__all__ = [
    # orchestrator
    "ReconReport",
    "ToolResult",
    "run_recon",
    "stream_recon",
    # output
    "render_report_markdown",
    "render_report_json",
    # normalize (re-exported because callers may want to preview the
    # normalized target before dispatching)
    "normalize_target",
    "refang",
    "NormalizedTarget",
    # safety (re-exported because the Streamlit wire-up checks it before
    # dispatch, separately from the orchestrator's internal check)
    "TargetBlockedError",
    "assert_target_allowed",
]

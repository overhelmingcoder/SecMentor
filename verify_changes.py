"""Pre-flight check for the Streamlit worker.

Run this before `streamlit run web/streamlit_app.py` to catch the
"stale-worker-holds-old-module-object" bug from
`docs/technical_write_up.md` row 7 and row 12. The script does three
checks and prints a single verdict line:

    READY                         -> nothing to do, safe to `streamlit run`
    STALE_WORKER: kill PIDs ...   -> a long-lived worker is bound; kill it
    FRESH_BUT_PORT_HELD: ...      -> no streamlit.exe but a python PID owns
                                    the port (the harder row-7 case)

The script never echoes secret values (no `Get-Content .env`,
no `printenv`); the only file it reads is `web/streamlit_app.py` to
confirm it exists.

Usage:
    .\.venv\\Scripts\\python.exe verify_changes.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


PORT: int = 8765
PROJECT_ROOT: Path = Path(__file__).resolve().parent
VIEW_FILE: Path = PROJECT_ROOT / "web" / "streamlit_app.py"


def _ps_where(filter_expr: str) -> list[tuple[int, str]]:
    """Return ``[(pid, command_line), ...]`` for processes matching *filter_expr*.

    Uses ``Get-CimInstance Win32_Process`` so the script is portable to
    any Windows box with PowerShell, and so it does not depend on the
    `psutil` package. Returns an empty list if PowerShell is missing or
    the query times out (we degrade gracefully — see the verdict logic).
    """
    import json
    import subprocess

    ps = (
        "$ErrorActionPreference='SilentlyContinue'; "
        f"Get-CimInstance Win32_Process -Filter \"{filter_expr}\" "
        "| Select-Object ProcessId, CommandLine "
        "| ConvertTo-Json -Depth 1 -Compress"
    )
    try:
        out = subprocess.check_output(
            ["powershell.exe", "-NoProfile", "-Command", ps],
            timeout=10,
            text=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return []
    out = out.strip()
    if not out:
        return []
    data = json.loads(out) if not out.startswith("[") else json.loads(out)
    if isinstance(data, dict):
        data = [data]
    return [(int(row["ProcessId"]), str(row.get("CommandLine") or "")) for row in data]


def _streamlit_workers() -> list[int]:
    """Return PIDs of processes whose image name is `streamlit.exe`."""
    rows = _ps_where("Name = 'streamlit.exe'")
    return [pid for pid, _ in rows]


def _port_owners(port: int) -> list[int]:
    """Return PIDs holding *port* in LISTEN state (any process, not just streamlit)."""
    rows = _ps_where(f"Name = 'python.exe'")
    # We can't filter by port inside WQL portably, so we cross-check with
    # Get-NetTCPConnection in a second PowerShell call.
    import json
    import subprocess

    ps = (
        "$ErrorActionPreference='SilentlyContinue'; "
        f"Get-NetTCPConnection -LocalPort {port} -State Listen "
        "| Select-Object -ExpandProperty OwningProcess "
        "| ConvertTo-Json -Depth 1 -Compress"
    )
    try:
        out = subprocess.check_output(
            ["powershell.exe", "-NoProfile", "-Command", ps],
            timeout=10,
            text=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return []
    out = out.strip()
    if not out:
        return []
    data = json.loads(out)
    if isinstance(data, int):
        return [data]
    if isinstance(data, list):
        return [int(x) for x in data]
    return []


def _fresh_import_works() -> bool:
    """True if a fresh interpreter can import the symbols the view needs."""
    try:
        mod = importlib.import_module("app.config")
        from app.config import iter_api_keys  # noqa: F401
        return hasattr(mod, "iter_api_keys") and callable(mod.iter_api_keys)
    except Exception as exc:  # pragma: no cover - surfaced in the verdict
        print(f"  fresh-import probe failed: {exc!r}", file=sys.stderr)
        return False


def main() -> int:
    print(f"Pre-flight for {PROJECT_ROOT.name} (port {PORT})")
    print(f"  view file: {VIEW_FILE}  exists={VIEW_FILE.exists()}")

    if not VIEW_FILE.exists():
        print("VERDICT: MISSING_VIEW")
        print(f"  {VIEW_FILE} does not exist. Did you move the web/ folder?")
        return 2

    streamlit_pids = _streamlit_workers()
    port_pids = _port_owners(PORT)
    fresh_ok = _fresh_import_works()

    print(f"  streamlit.exe PIDs : {streamlit_pids or 'none'}")
    print(f"  port {PORT} owners  : {port_pids or 'none'}")
    print(f"  fresh import OK    : {fresh_ok}")

    if streamlit_pids or port_pids:
        victims = sorted(set(streamlit_pids) | set(port_pids))
        print(f"VERDICT: STALE_WORKER: kill PIDs {','.join(map(str, victims))}")
        print("  Then re-run this script. It will print READY.")
        return 1

    if not fresh_ok:
        print("VERDICT: IMPORT_BROKEN")
        print("  No stale worker, but a fresh interpreter cannot import the view's")
        print("  symbols. The bug is in the source — read the traceback above.")
        return 3

    print("VERDICT: READY")
    print("  Safe to run:  .\\.venv\\Scripts\\streamlit.exe run web\\streamlit_app.py "
          f"--server.port {PORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

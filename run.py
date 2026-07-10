"""One-command runner: `python run.py` is the project's `npm run dev`.

What it does, in order:

1. Locates a Python 3.11+ interpreter on PATH.
2. Creates `.venv/` if it does not exist.
3. Installs `requirements.txt` into the venv (idempotent — re-runs are fast).
4. Creates `.env` from `.env.example` if `.env` is missing.
5. Runs `verify_changes.py` (pre-flight) inside the venv.
6. Launches `streamlit run web/streamlit_app.py` in the FOREGROUND, so
   Ctrl-C stops the worker cleanly and the terminal stays attached.

Usage:

    python run.py                # from the project root
    python run.py web            # `web` is accepted (and ignored) as an alias
    python run.py --port 9000    # override the default port (8765)
    python run.py --no-preflight # skip the verify_changes.py step
    python run.py --detach       # launch detached and exit (CI / scripts)

The runner only has one job — boot the Streamlit web UI — so any stray
positional argument is swallowed silently. Future subcommands (e.g. `cli`,
`test`) can be added here without breaking existing muscle memory.

The script never echoes secret values. It will refuse to start if `.env`
does not contain a real `OPENROUTER_API_KEY=sk-or-v1-...` value.

What the boot brings online (Phase 12 + PR-E, 2026-06-15, see
`docs/technical_write_up.md` rows 19-20 and `docs/phase_12_rag_and_history.md`):

- Persistent chat history sidebar (`app/storage.py` six-table SQLite schema,
  `+ New chat` / rename / two-step soft-delete, sidebar ordering
  `updated_at DESC`).
- Per-chat FAISS RAG (`app/rag_store.py:RagStore` — lazy `IndexFlatIP`,
  `_index_version` invalidation on the `chats` table, score-threshold 0.30,
  embedder-degraded mode via `PUKU_RAG_OFFLINE=1`, cross-chat isolation
  pinned by `tests/test_rag.py:CrossChatIsolationTests`).
- Global security corpus (`app/rag_global.py:GlobalCorpusStore` — second
  parallel FAISS index, 5 source kinds: OWASP, MITRE ATT&CK, CWE, GTFOBins,
  Sigma, SHA-256 dedup, scored-then-merge with the per-chat index in
  `web/chat_helpers.py:build_messages_with_rag`, one-shot CLI at
  `scripts/ingest_security_corpus.py`).
- Cumulative test suite: 365 tests (9 pre-existing failures + 1 error + 19
  environmental skips — all 68 Phase 12 tests are green, see
  `docs/technical_write_up.md` row 19-20 footers and Decision 12 for the
  durable design choices).
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import venv
from pathlib import Path


PROJECT_ROOT: Path = Path(__file__).resolve().parent
VENV_DIR: Path = PROJECT_ROOT / ".venv"
REQUIREMENTS: Path = PROJECT_ROOT / "requirements.txt"
ENV_FILE: Path = PROJECT_ROOT / ".env"
ENV_EXAMPLE: Path = PROJECT_ROOT / ".env.example"
PREFLIGHT: Path = PROJECT_ROOT / "verify_changes.py"
VIEW_FILE: Path = PROJECT_ROOT / "web" / "streamlit_app.py"
DEFAULT_PORT: int = 8765
MIN_PY: tuple[int, int] = (3, 11)


# --------------------------------------------------------------------------- #
# 1. Python interpreter discovery
# --------------------------------------------------------------------------- #

def _venv_python(venv_dir: Path) -> Path:
    """Return the path to the venv's Python executable."""
    if platform.system() == "Windows":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _ensure_venv() -> Path:
    """Create the venv if it does not exist, return the venv's python.exe."""
    py = _venv_python(VENV_DIR)
    if py.exists():
        return py
    print(f"[run.py] creating venv at {VENV_DIR} ...")
    builder = venv.EnvBuilder(
        system_site_packages=False,
        clear=False,
        symlinks=(platform.system() != "Windows"),
        upgrade=False,
        with_pip=True,
    )
    builder.create(VENV_DIR)
    if not py.exists():
        sys.exit(f"[run.py] venv created but {py} is missing")
    return py


def _check_host_python() -> None:
    """Refuse to run if the host Python is older than MIN_PY."""
    if sys.version_info < MIN_PY:
        want = ".".join(map(str, MIN_PY))
        have = ".".join(map(str, sys.version_info[:3]))
        sys.exit(f"[run.py] need Python {want}+, found {have}. Upgrade Python first.")


# --------------------------------------------------------------------------- #
# 2. Dependency install
# --------------------------------------------------------------------------- #

def _pip_install(py: Path) -> None:
    """Install requirements.txt into the venv. Idempotent."""
    if not REQUIREMENTS.exists():
        sys.exit(f"[run.py] missing {REQUIREMENTS}")
    print(f"[run.py] installing {REQUIREMENTS.name} into the venv (this is fast on re-runs) ...")
    subprocess.check_call(
        [str(py), "-m", "pip", "install", "--disable-pip-version-check", "-q", "-r", str(REQUIREMENTS)]
    )


# --------------------------------------------------------------------------- #
# 3. .env scaffolding
# --------------------------------------------------------------------------- #

def _ensure_env() -> bool:
    """Copy .env.example -> .env if missing. Return True if a copy was made."""
    if ENV_FILE.exists():
        return False
    if not ENV_EXAMPLE.exists():
        sys.exit(f"[run.py] missing both {ENV_FILE} and {ENV_EXAMPLE}")
    print(f"[run.py] creating {ENV_FILE.name} from {ENV_EXAMPLE.name} ...")
    ENV_FILE.write_text(ENV_EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    return True


def _check_env_key() -> None:
    """Refuse to start if no valid API key is found.
    Accepts either a real OpenRouter key (sk-or-v1-...) or any non-empty Gemini key.
    """
    if not ENV_FILE.exists():
        return  # _ensure_env will have already created it; the user must still fill it in
    text = ENV_FILE.read_text(encoding="utf-8")
    
    has_openrouter_key = "OPENROUTER_API_KEY=sk-or-v1-" in text and "PASTE" not in text
    has_gemini_key = "GEMINI_API_KEY=" in text and "YOUR_GOOGLE" not in text and len(text.split("GEMINI_API_KEY=")[1].splitlines()[0].strip()) > 10
    
    if not (has_openrouter_key or has_gemini_key):
        sys.exit(
            f"[run.py] {ENV_FILE} does not contain a real API key.\n"
            f"  Please provide either a GEMINI_API_KEY or an OPENROUTER_API_KEY (sk-or-v1-...),\n"
            f"  then re-run:  python run.py"
        )


# --------------------------------------------------------------------------- #
# 4. Pre-flight (optional)
# --------------------------------------------------------------------------- #

def _preflight(py: Path) -> None:
    """Run verify_changes.py inside the venv. Exit if it returns non-zero."""
    if not PREFLIGHT.exists():
        print(f"[run.py] {PREFLIGHT.name} not found, skipping pre-flight")
        return
    print(f"[run.py] running pre-flight ({PREFLIGHT.name}) ...")
    rc = subprocess.call([str(py), str(PREFLIGHT)])
    if rc != 0:
        sys.exit(f"[run.py] pre-flight failed (exit {rc}); see the verdict above")


# --------------------------------------------------------------------------- #
# 5. Server launch
# --------------------------------------------------------------------------- #

def _streamlit_exe(venv_dir: Path) -> Path:
    if platform.system() == "Windows":
        return venv_dir / "Scripts" / "streamlit.exe"
    return venv_dir / "bin" / "streamlit"


def _launch_foreground(venv_dir: Path, port: int) -> int:
    """Launch streamlit in the foreground. Ctrl-C stops the worker."""
    exe = _streamlit_exe(venv_dir)
    if not exe.exists():
        sys.exit(f"[run.py] {exe} not found — did the pip install step succeed?")
    cmd = [
        str(exe), "run", str(VIEW_FILE),
        "--server.port", str(port),
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
    ]
    print(f"[run.py] launching: {' '.join(cmd)}")
    print(f"[run.py] open http://localhost:{port} in your browser")
    print(f"[run.py] press Ctrl-C to stop")
    try:
        return subprocess.call(cmd, cwd=str(PROJECT_ROOT))
    except KeyboardInterrupt:
        return 0


def _launch_detached(venv_dir: Path, port: int) -> int:
    """Launch streamlit as a background process and return its PID."""
    exe = _streamlit_exe(venv_dir)
    out_log = PROJECT_ROOT / "streamlit.log.out"
    err_log = PROJECT_ROOT / "streamlit.log.err"
    args = [
        "run", str(VIEW_FILE),
        "--server.port", str(port),
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
    ]
    if platform.system() == "Windows":
        # On Windows we cannot simply Popen with start_new_session=True, because
        # when run.py exits, the console break is propagated to the child and
        # streamlit gets killed. We use DETACHED_PROCESS so the worker has no
        # controlling console at all — it then survives this script's exit.
        # The args list is passed as a real Python list (not joined), so the
        # child receives them as proper argv (the same shape the foreground
        # branch uses). This avoids every PowerShell arg-encoding gotcha.
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        out_fh = open(out_log, "ab")
        err_fh = open(err_log, "ab")
        proc = subprocess.Popen(
            [str(exe), *args],
            cwd=str(PROJECT_ROOT),
            stdout=out_fh,
            stderr=err_fh,
            stdin=subprocess.DEVNULL,
            creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
            close_fds=True,
        )
        out_fh.close()
        err_fh.close()
        pid = proc.pid
    else:
        proc = subprocess.Popen(
            [str(exe), *args],
            cwd=str(PROJECT_ROOT),
            stdout=open(out_log, "ab"),
            stderr=open(err_log, "ab"),
            start_new_session=True,
        )
        pid = proc.pid
    print(f"[run.py] detached worker PID = {pid}")
    print(f"[run.py] open http://localhost:{port} in your browser")
    print(f"[run.py] to stop:  {'Stop-Process -Id ' + str(pid) + ' -Force' if platform.system() == 'Windows' else 'kill ' + str(pid)}")
    return 0


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(
        description="One-command runner for the AI Security Chatbot (Stage 1)."
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="web",
        choices=["web"],
        help="Optional subcommand. Only 'web' is supported (and is the default); provided so `python run.py web` works as an alias.",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Streamlit port (default {DEFAULT_PORT})")
    parser.add_argument("--no-preflight", action="store_true", help="Skip the verify_changes.py step")
    parser.add_argument("--detach", action="store_true", help="Launch the server in the background and return its PID")
    args = parser.parse_args()
    # `args.command` is validated by argparse's `choices=["web"]` and reserved
    # for future subcommands (e.g. `cli`, `test`); touch it so static
    # analysers don't flag it as unused.
    _ = args.command

    _check_host_python()
    py = _ensure_venv()
    _pip_install(py)
    env_created = _ensure_env()
    if env_created:
        sys.exit(
            f"[run.py] created {ENV_FILE.name} from {ENV_EXAMPLE.name}.\n"
            f"  Open it in any editor, paste your real OPENROUTER_API_KEY=sk-or-v1-... value,\n"
            f"  save, then re-run:  python run.py"
        )
    _check_env_key()
    if not args.no_preflight:
        _preflight(py)

    if args.detach:
        return _launch_detached(VENV_DIR, args.port)
    return _launch_foreground(VENV_DIR, args.port)


if __name__ == "__main__":
    raise SystemExit(main())

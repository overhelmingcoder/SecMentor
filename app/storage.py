"""SQLite persistence layer for Phase 12 (chat history + RAG).

This module is the **single source of truth** for the on-disk schema that
links chats, messages, artifacts, and chunk embeddings. The schema is
pinned in :data:`_SCHEMA_SQL` and must not change without a migration
(see ``docs/phase_12_rag_and_history.md`` §3 — the contract with
Phase 14).

Design notes (mirroring the design notes in the doc, but executable):

* **No ``users`` table.** Single-tenant v1. Phase 14 will add a
  ``users`` table and backfill every existing row to a default
  ``owner_id``.
* **No ``sessions`` table.** A "session" in this project *is* the row
  the user is currently looking at in ``chats``. There is no separate
  concept of session until auth lands.
* **Soft delete.** ``chats.deleted_at`` is the marker; a hard delete
  would cascade through ``messages`` and ``chunks`` mid-session and
  surprise the user with a blank retriever.
* **Embeddings are BLOBs.** ``np.float32.tobytes()`` round-trips via
  ``np.frombuffer(blob, dtype=np.float32)``. Storing them inside the
  same SQLite file means one file to back up, one file to delete for a
  hard reset. The wrapper in ``app/rag_store.py`` (Phase 12 PR-B) is
  the seam for switching to a sidecar FAISS file later.
* **Repository functions return a *new* list.** Every list-building
  function copies the SQLite rows into a fresh Python ``list`` so the
  caller can mutate the result without corrupting the underlying
  ``sqlite3.Row`` view. This is pinned by a test.
* **No ``updated_at`` trigger.** The view layer is the only writer; it
  sets ``updated_at = now()`` on every append. A trigger would mask
  bugs in the view (forgot to bump the timestamp → sidebar shows stale
  ordering).
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional, Sequence


# --- Schema -----------------------------------------------------------------

# The single source of truth. Pinned here AND in §3 of
# `docs/phase_12_rag_and_history.md`. All `CREATE … IF NOT EXISTS` so
# `init_db()` is idempotent and safe to re-run on every startup.
_SCHEMA_SQL: str = """
CREATE TABLE IF NOT EXISTS chats (
    id            TEXT PRIMARY KEY,                 -- uuid4 hex
    title         TEXT NOT NULL,                    -- first 60 chars of first user turn
    created_at    TEXT NOT NULL,                    -- ISO 8601 UTC
    updated_at    TEXT NOT NULL,
    teaching_mode TEXT NOT NULL DEFAULT 'mentor',  -- denormalised from session_state
    deleted_at    TEXT                              -- soft delete; NULL = live
);

CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id    TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role       TEXT NOT NULL,                       -- 'user' | 'assistant' | 'system'
    content    TEXT NOT NULL,                       -- JSON: str or list[content parts]
    created_at TEXT NOT NULL,
    ord        INTEGER NOT NULL                     -- turn order within a chat
);
CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, ord);

CREATE TABLE IF NOT EXISTS artifacts (
    id         TEXT PRIMARY KEY,                    -- uuid4 hex
    chat_id    TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    filename   TEXT NOT NULL,
    mime       TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_artifacts_chat ON artifacts(chat_id);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    artifact_id TEXT NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
    ord         INTEGER NOT NULL,
    text        TEXT NOT NULL,
    embedding   BLOB NOT NULL                       -- np.float32.tobytes()
);
CREATE INDEX IF NOT EXISTS idx_chunks_artifact ON chunks(artifact_id);

-- Phase 12 PR-E: global security corpus (OWASP / MITRE / CWE / GTFOBins /
-- Sigma). This table holds the *curated* knowledge base that every chat
-- can retrieve from. Unlike ``chunks`` (which is per-chat, per-artifact),
-- ``global_chunks`` is keyed on a ``source_id`` string (e.g. "owasp",
-- "mitre") and is *not* attached to any user-visible chat. Splitting the
-- two keeps user uploads (with PII risk) cleanly separated from public
-- open-license material.
--
-- License is denormalised onto each row so the retriever can render a
-- provenance footer per chunk without a join. It is also the audit hook:
-- if a chunk is ever disputed, ``source_url`` is the URL we cite and
-- ``license`` is the licence we rendered under. The PR-E doc spells out
-- the per-source license values; do not change them without a docs PR.
CREATE TABLE IF NOT EXISTS global_chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id   TEXT NOT NULL,                      -- 'owasp' | 'mitre' | 'cwe' | 'gtfobins' | 'sigma'
    source_url  TEXT NOT NULL,                      -- canonical URL of the source doc
    license     TEXT NOT NULL,                      -- SPDX-like: 'CC-BY-SA-4.0', 'CC-BY-4.0', 'MIT'
    ord         INTEGER NOT NULL,                   -- position within the source doc
    text        TEXT NOT NULL,
    embedding   BLOB NOT NULL,                      -- np.float32.tobytes()
    created_at  TEXT NOT NULL,
    UNIQUE(source_id, source_url, ord)
);
CREATE INDEX IF NOT EXISTS idx_global_chunks_source ON global_chunks(source_id);

-- Phase 15 — reconnaissance audit log. One row per recon tool
-- invocation, regardless of success. ``chat_id`` is nullable so a
-- free-form recon from the sidebar (no active chat) still gets
-- audited; the ON DELETE SET NULL keeps recon history when a chat
-- row is hard-deleted. ``target`` and ``tool`` are the same strings
-- the orchestrator uses, so an operator can filter by tool ('dns',
-- 'ipinfo', 'urlinfo', 'whois', 'crt_sh', 'nmap') or replay every
-- call against a specific target. ``scope_token`` is the user-
-- affirmed scope (e.g. 'engagement', 'ctf', 'labs', 'personal-lab')
-- so the audit log can prove the call was authorized.
CREATE TABLE IF NOT EXISTS recon_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id         TEXT REFERENCES chats(id) ON DELETE SET NULL,
    target          TEXT NOT NULL,
    tool            TEXT NOT NULL,
    scope_token     TEXT,
    status          TEXT NOT NULL DEFAULT 'ok',   -- 'ok' | 'blocked' | 'error'
    timestamp       TEXT NOT NULL,                -- ISO 8601 UTC with Z
    duration_ms     INTEGER,                      -- wall-clock for the call
    result_excerpt  TEXT                          -- first 500 chars of the result
);
CREATE INDEX IF NOT EXISTS idx_recon_log_chat   ON recon_log(chat_id);
CREATE INDEX IF NOT EXISTS idx_recon_log_target ON recon_log(target);
CREATE INDEX IF NOT EXISTS idx_recon_log_tool   ON recon_log(tool);
"""


# --- Configuration ----------------------------------------------------------

#: Default on-disk location for the SQLite database. Can be overridden by
#: setting the ``SECMENTOR_DB_PATH`` environment variable. The directory
#: is created on first use by :func:`init_db`.
_DEFAULT_DB_PATH: Path = Path.home() / ".cache" / "secmentor" / "secmentor.db"


def db_path() -> Path:
    """Return the resolved on-disk path to the SQLite database.

    Reads ``SECMENTOR_DB_PATH`` from the environment on every call so
    tests that monkeypatch the env (e.g. ``monkeypatch.setenv``) are
    picked up without re-importing the module.

    Returns:
        The absolute path that :func:`init_db` and the repository
        functions will use.
    """
    override = os.getenv("SECMENTOR_DB_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return _DEFAULT_DB_PATH


# --- Connection helper ------------------------------------------------------


@contextmanager
def _connect(path: Optional[Path] = None) -> Iterator[sqlite3.Connection]:
    """Open a SQLite connection with the project-standard settings.

    The connection uses :class:`sqlite3.Row` as its row factory so the
    repository functions can address columns by name (more readable
    than tuple indexing and immune to ``SELECT *`` column-order
    changes). Foreign keys are enabled per-connection (SQLite's
    ``PRAGMA foreign_keys`` is per-connection, not per-database) so
    the ``ON DELETE CASCADE`` clauses actually fire.

    When called with an explicit non-``":memory:"`` path, the
    schema is lazy-initialised on first connect so test fixtures
    that build a per-test ``db_path`` do not have to remember to
    call :func:`init_db` themselves. The default-path branch
    (production) skips this: the Streamlit view's
    :func:`web.streamlit_app._init_state` already calls
    :func:`init_db` once at boot.

    Args:
        path: Explicit on-disk path. Defaults to :func:`db_path`. Pass
            ``Path(":memory:")`` (or the string ``":memory:"``) for
            the test suite so every connection gets a fresh
            database.

    Yields:
        A configured :class:`sqlite3.Connection`.
    """
    if path is None:
        actual: Any = db_path()
    elif str(path) == ":memory:":
        # ``Path(":memory:")`` is technically a Path that points at
        # the literal directory name ":memory:". We translate it
        # back to the SQLite in-memory token so tests can pass
        # either a string or a Path.
        actual = ":memory:"
    else:
        actual = path
        # Lazy schema init for explicit-path callers. This
        # function (``_connect``) is called *before*
        # ``init_db`` is defined at module-load time because
        # ``init_db`` itself uses ``_connect``. We guard with
        # a module-level flag so the cost is paid once per
        # process even when many connections are opened
        # against the same path.
        _ensure_schema(actual)
    conn = sqlite3.connect(actual)
    conn.row_factory = sqlite3.Row
    # Per-connection FK enforcement. SQLite defaults to OFF for legacy
    # reasons; the schema's CASCADE clauses are silently ignored without
    # this. Pinned by StorageSchemaTests.
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


#: Per-process cache of paths whose schema has already been
#: created via :func:`_ensure_schema`. Keyed on the resolved
#: path string so the same physical DB is only initialised
#: once even when many connections are opened against it.
_SCHEMA_READY: set = set()


def _ensure_schema(path: Path) -> None:
    """Initialise the schema for ``path`` once per process.

    Used by :func:`_connect` for explicit-path callers (tests,
    the CLI ingest script). The default-path production branch
    is handled by the Streamlit view's ``_init_state`` and
    skips this — the production DB is bootstrapped exactly
    once at boot.

    ``CREATE … IF NOT EXISTS`` makes this a no-op on the
    second call, but the per-process cache short-circuits the
    file-system check entirely on subsequent calls.

    Windows holds the SQLite file handle until the connection
    is *explicitly* closed; ``with sqlite3.connect(...)`` only
    commits, it does not release the OS-level lock. Tests that
    use a per-test ``TemporaryDirectory`` would otherwise
    fail teardown with a ``PermissionError`` on the file, so
    we ``.close()`` explicitly and ``gc.collect()`` to nudge
    any lingering references.
    """
    key = str(path)
    if key in _SCHEMA_READY:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target)
    try:
        conn.executescript(_SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()
    # On Windows, the file lock is released only after the
    # connection is GC'd. ``import gc`` at module level would be
    # cheaper, but a local import keeps the helper self-contained
    # and the cost is paid only once per path.
    import gc

    gc.collect()
    _SCHEMA_READY.add(key)


# --- Time helpers -----------------------------------------------------------


def _utcnow_iso() -> str:
    """Return the current UTC time as an ISO 8601 string with a ``Z`` suffix.

    The ``Z`` suffix is friendlier to JSON consumers and avoids the
    ``+00:00`` vs ``Z`` parsing pitfall in some JavaScript clients. The
    ``str(datetime.isoformat())`` shape with a trailing ``Z`` is the
    canonical "UTC" representation.
    """
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _new_chat_id() -> str:
    """Return a fresh chat id (uuid4 hex, 32 chars).

    The doc pins the column as ``uuid4 hex``; the canonical form is
    ``uuid.uuid4().hex`` which is 32 lowercase hex chars with no
    dashes. Using ``str(uuid4)`` (with dashes) would also be 36 chars
    and still be a valid primary key — the choice is style only —
    but we pin the 32-char form so tests can assert length.
    """
    return uuid.uuid4().hex


# --- Schema initialisation --------------------------------------------------


def init_db(path: Optional[Path] = None) -> Path:
    """Create the schema if it does not exist; no-op if it does.

    Re-running this function on an existing database is a no-op
    (every ``CREATE`` uses ``IF NOT EXISTS``). On a fresh install it
    also creates the parent directory of the database file.

    Args:
        path: Override the on-disk location. Useful for tests; in
            production leave this as ``None`` and let :func:`db_path`
            decide.

    Returns:
        The absolute path to the database that was initialised. The
        caller can use this for log lines and for passing to
        subsequent repository calls.
    """
    target = path if path is not None else db_path()
    if str(target) != ":memory:":
        # ``Path.parent.mkdir(parents=True, exist_ok=True)`` is the
        # cross-version mkdir -p. For an in-memory database the
        # ``str`` comparison above short-circuits this block entirely.
        target.parent.mkdir(parents=True, exist_ok=True)
    # ``_connect`` will lazy-init the schema via ``_ensure_schema``
    # for explicit paths, so the explicit ``executescript`` here is
    # only needed for the default-path branch where the lazy
    # guard in ``_connect`` is bypassed.
    with _connect(target) as conn:
        if path is None:
            conn.executescript(_SCHEMA_SQL)
    return target


# --- Repository: chats ------------------------------------------------------


def create_chat(
    *,
    title: str,
    teaching_mode: str = "mentor",
    chat_id: Optional[str] = None,
    path: Optional[Path] = None,
) -> str:
    """Insert a new chat row and return its id.

    Args:
        title: The chat title (typically the first 60 chars of the
            first user turn).
        teaching_mode: Denormalised teaching persona tag. Defaults to
            ``"mentor"`` to match the schema default and the web UI
            default. Pinned so the sidebar can render the right
            label without a join.
        chat_id: Explicit id (e.g. in tests). Defaults to a fresh
            uuid4 hex. Useful for property-based tests that need
            reproducible ids.
        path: Override the on-disk location. ``None`` uses
            :func:`db_path`.

    Returns:
        The id of the newly created chat.
    """
    cid = chat_id or _new_chat_id()
    now = _utcnow_iso()
    with _connect(path) as conn:
        conn.execute(
            """
            INSERT INTO chats (id, title, created_at, updated_at, teaching_mode, deleted_at)
            VALUES (?, ?, ?, ?, ?, NULL)
            """,
            (cid, title, now, now, teaching_mode),
        )
    return cid


def list_chats(
    *,
    limit: int = 20,
    include_deleted: bool = False,
    path: Optional[Path] = None,
) -> list[dict]:
    """Return chats ordered by ``updated_at`` DESC (newest first).

    Args:
        limit: Maximum number of rows to return. Defaults to 20 to
            match the sidebar's "recent chats" cap (see
            `docs/phase_12_rag_and_history.md` PR-C).
        include_deleted: When ``True``, soft-deleted rows are
            included. Default ``False`` — the sidebar must not show
            them.
        path: Override the on-disk location. ``None`` uses
            :func:`db_path`.

    Returns:
        A **new** ``list`` of ``dict`` (one per chat). The list is a
        fresh copy; mutating it does not affect the database. The
        dicts are built from :class:`sqlite3.Row` snapshots and are
        safe to mutate.
    """
    if limit <= 0:
        raise ValueError(f"limit must be a positive int; got {limit!r}")
    sql = """
        SELECT id, title, created_at, updated_at, teaching_mode, deleted_at
        FROM chats
        {where}
        ORDER BY updated_at DESC
        LIMIT ?
    """.format(where="" if include_deleted else "WHERE deleted_at IS NULL")
    with _connect(path) as conn:
        rows = conn.execute(sql, (limit,)).fetchall()
    # Materialise into plain dicts so the caller can mutate freely
    # without touching the (now-closed) Row objects. The "return a
    # new list, not a view" contract is pinned by ChatRepositoryTests.
    return [dict(r) for r in rows]


def get_chat(chat_id: str, path: Optional[Path] = None) -> Optional[dict]:
    """Return a single chat row, or ``None`` if it does not exist.

    Includes soft-deleted rows. Use :func:`list_chats` for the
    sidebar's "live chats" view.

    Args:
        chat_id: The chat's uuid4 hex id.
        path: Override the on-disk location. ``None`` uses
            :func:`db_path`.

    Returns:
        A ``dict`` with the chat's columns, or ``None`` if no row
        matches.
    """
    with _connect(path) as conn:
        row = conn.execute(
            """
            SELECT id, title, created_at, updated_at, teaching_mode, deleted_at
            FROM chats WHERE id = ?
            """,
            (chat_id,),
        ).fetchone()
    return dict(row) if row is not None else None


def soft_delete_chat(chat_id: str, path: Optional[Path] = None) -> bool:
    """Mark a chat as soft-deleted.

    The row is excluded from :func:`list_chats` after the call but
    the underlying data is preserved (so a future "undo" button is
    one UPDATE away). Returns ``True`` if a row was updated, ``False``
    if the chat does not exist (or was already soft-deleted — the
    WHERE clause excludes the latter so the return value is honest
    about the *transition*).

    Args:
        chat_id: The chat's uuid4 hex id.
        path: Override the on-disk location. ``None`` uses
            :func:`db_path`.

    Returns:
        ``True`` if a live row was transitioned to deleted, ``False``
        otherwise.
    """
    now = _utcnow_iso()
    with _connect(path) as conn:
        cur = conn.execute(
            "UPDATE chats SET deleted_at = ?, updated_at = ? "
            "WHERE id = ? AND deleted_at IS NULL",
            (now, now, chat_id),
        )
    return cur.rowcount > 0


def hard_delete_chat(chat_id: str, path: Optional[Path] = None) -> bool:
    """Permanently delete a chat and all its messages / artifacts / chunks.

    This is the destructive counterpart to :func:`soft_delete_chat`.
    CASCADE on the foreign keys removes the dependent rows
    automatically — the storage layer does not have to do it by
    hand. Use with care: once this returns, the data is gone and a
    VACUUM would erase the freed pages.

    Args:
        chat_id: The chat's uuid4 hex id.
        path: Override the on-disk location. ``None`` uses
            :func:`db_path`.

    Returns:
        ``True`` if a row was deleted, ``False`` if the chat did not
        exist.
    """
    with _connect(path) as conn:
        cur = conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    return cur.rowcount > 0


def touch_chat(chat_id: str, *, path: Optional[Path] = None) -> None:
    """Bump ``updated_at`` to now() for the given chat.

    The view calls this after every append so the sidebar ordering
    reflects the most recent activity. We do not rely on a SQL
    trigger (see module docstring).

    Args:
        chat_id: The chat's uuid4 hex id.
        path: Override the on-disk location. ``None`` uses
            :func:`db_path`.

    Raises:
        sqlite3.IntegrityError: If ``chat_id`` does not exist (the
            FK from ``messages`` would have caught it earlier; this
            is a defensive last line).
    """
    now = _utcnow_iso()
    with _connect(path) as conn:
        conn.execute(
            "UPDATE chats SET updated_at = ? WHERE id = ?",
            (now, chat_id),
        )


def update_chat_title(
    chat_id: str,
    title: str,
    *,
    path: Optional[Path] = None,
) -> None:
    """Rename a chat row.

    The view calls this once on the *first* user turn of a chat so the
    sidebar shows the actual question instead of the placeholder
    ``"New chat"`` string that :func:`create_chat` seeds. A subsequent
    rename (the user clicks the title, types a new one) goes through the
    same function — we never silently overwrite a title the user has
    hand-edited, so callers must check the current title first if they
    want to avoid stomping user changes.

    The title is stored as-is (no truncation, no strip here). The
    caller is responsible for trimming whitespace and shortening to the
    display cap; the storage layer just persists bytes.

    Args:
        chat_id: The chat's uuid4 hex id.
        title: The new title. Empty string is allowed and clears the
            label, but the view is expected to never call this with an
            empty string (the sidebar falls back to ``"Untitled chat"``
            on empty rows).
        path: Override the on-disk location. ``None`` uses
            :func:`db_path`.

    Raises:
        sqlite3.IntegrityError: If ``chat_id`` does not exist.
    """
    with _connect(path) as conn:
        conn.execute(
            "UPDATE chats SET title = ? WHERE id = ?",
            (title, chat_id),
        )


# --- Repository: messages ---------------------------------------------------


def append_message(
    chat_id: str,
    *,
    role: str,
    content: Any,
    path: Optional[Path] = None,
) -> int:
    """Append a message to a chat and bump the chat's ``updated_at``.

    The ``ord`` column is the turn-order within the chat. We compute
    it as ``MAX(ord) + 1`` (or 0 for the first message) so the view
    can replay the conversation in order without storing an
    index that gets out of sync with deletes / reorders.

    ``content`` may be either a plain string (text-only turn) or a
    list of content parts (multimodal turn with ``image_url``). The
    storage layer encodes the list as JSON so the rest of the code
    never has to know the difference — see §3 design note 5 of the
    Phase 12 doc.

    Args:
        chat_id: The chat to append to. Must already exist (the FK
            will raise ``IntegrityError`` otherwise).
        role: One of ``"user"``, ``"assistant"``, ``"system"``.
        content: The turn content — ``str`` or ``list[dict]``.
        path: Override the on-disk location. ``None`` uses
            :func:`db_path`.

    Returns:
        The autoincrement ``id`` of the new message row.
    """
    if role not in ("user", "assistant", "system"):
        raise ValueError(
            f"role must be one of 'user', 'assistant', 'system'; got {role!r}"
        )
    encoded = content if isinstance(content, str) else json.dumps(content)
    now = _utcnow_iso()
    with _connect(path) as conn:
        next_ord_row = conn.execute(
            "SELECT COALESCE(MAX(ord), -1) + 1 AS next_ord "
            "FROM messages WHERE chat_id = ?",
            (chat_id,),
        ).fetchone()
        next_ord = int(next_ord_row["next_ord"])
        cur = conn.execute(
            """
            INSERT INTO messages (chat_id, role, content, created_at, ord)
            VALUES (?, ?, ?, ?, ?)
            """,
            (chat_id, role, encoded, now, next_ord),
        )
        # Bump the chat's updated_at in the same transaction so the
        # sidebar ordering reflects the new turn without a second
        # write.
        conn.execute(
            "UPDATE chats SET updated_at = ? WHERE id = ?",
            (now, chat_id),
        )
    return int(cur.lastrowid)


def load_messages(
    chat_id: str,
    *,
    path: Optional[Path] = None,
) -> list[dict]:
    """Return all messages for a chat, ordered by ``ord`` ASC.

    The ``content`` field is decoded back to its original shape:
    a ``str`` for text-only turns, a ``list[dict]`` for multimodal
    turns. The detection is "looks like JSON" via :func:`json.loads`
    with a ``JSONDecodeError`` fallback that preserves the string
    shape. This keeps the wire format opaque to the caller.

    Args:
        chat_id: The chat to load.
        path: Override the on-disk location. ``None`` uses
            :func:`db_path`.

    Returns:
        A **new** ``list`` of ``dict`` (one per message). Mutating
        the list does not affect the database. The dicts are
        materialised from the rows so they are safe to mutate.
    """
    with _connect(path) as conn:
        rows = conn.execute(
            """
            SELECT id, chat_id, role, content, created_at, ord
            FROM messages
            WHERE chat_id = ?
            ORDER BY ord ASC
            """,
            (chat_id,),
        ).fetchall()
    out: list[dict] = []
    for r in rows:
        d = dict(r)
        # Decode the JSON envelope. A text-only turn stored as a
        # plain string round-trips to itself (json.loads on a string
        # is a TypeError, which we catch and treat as "not JSON").
        # We use the explicit str-shape check first to avoid the
        # cost of a JSON parse for the common text-only case.
        raw = d["content"]
        if isinstance(raw, str) and raw.startswith(("[", "{")):
            try:
                d["content"] = json.loads(raw)
            except json.JSONDecodeError:
                # Malformed JSON: keep the string shape so the caller
                # sees what we have rather than losing data.
                pass
        out.append(d)
    return out


# --- Repository: artifacts + chunks -----------------------------------------
# These two repositories are stubbed in PR-A so the storage module is
# self-contained. The full behaviour (chunk upload + embedding BLOB
# round-trip) is added in PR-B alongside the RAG pipeline. The stubs
# exist so the CASCADE behaviour between `chats` -> `artifacts` and
# `artifacts` -> `chunks` can be verified with a single fixture.


def add_artifact(
    chat_id: str,
    *,
    filename: str,
    mime: str,
    size_bytes: int,
    artifact_id: Optional[str] = None,
    path: Optional[Path] = None,
) -> str:
    """Record a new artifact attached to a chat.

    An "artifact" is a user-uploaded file (image, PDF, JSON, …) whose
    bytes are NOT stored here — only metadata. The actual chunks and
    their embeddings live in the ``chunks`` table. The split keeps
    the metadata index small (so the sidebar can list uploads
    quickly) and the embedding BLOBs out of the way of ordinary
    metadata queries.

    Args:
        chat_id: The owning chat. Must exist.
        filename: The original filename as the user attached it.
        mime: The MIME type (e.g. ``"application/pdf"``).
        size_bytes: The file size in bytes.
        artifact_id: Explicit id. Defaults to a fresh uuid4 hex.
        path: Override the on-disk location. ``None`` uses
            :func:`db_path`.

    Returns:
        The id of the newly created artifact.
    """
    aid = artifact_id or _new_chat_id()
    now = _utcnow_iso()
    with _connect(path) as conn:
        conn.execute(
            """
            INSERT INTO artifacts (id, chat_id, filename, mime, size_bytes, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (aid, chat_id, filename, mime, size_bytes, now),
        )
    return aid


def list_artifacts(
    chat_id: str, *, path: Optional[Path] = None
) -> list[dict]:
    """Return all artifacts for a chat, ordered by ``created_at`` ASC.

    Args:
        chat_id: The owning chat.
        path: Override the on-disk location. ``None`` uses
            :func:`db_path`.

    Returns:
        A **new** ``list`` of ``dict``. Safe to mutate.
    """
    with _connect(path) as conn:
        rows = conn.execute(
            """
            SELECT id, chat_id, filename, mime, size_bytes, created_at
            FROM artifacts
            WHERE chat_id = ?
            ORDER BY created_at ASC
            """,
            (chat_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_artifact(
    artifact_id: str, *, path: Optional[Path] = None
) -> bool:
    """Delete an artifact and (via CASCADE) all its chunks.

    Args:
        artifact_id: The artifact to delete.
        path: Override the on-disk location. ``None`` uses
            :func:`db_path`.

    Returns:
        ``True`` if a row was deleted, ``False`` otherwise.
    """
    with _connect(path) as conn:
        cur = conn.execute(
            "DELETE FROM artifacts WHERE id = ?", (artifact_id,)
        )
    return cur.rowcount > 0


def add_chunks(
    artifact_id: str,
    chunks: Iterable[tuple[str, bytes]],
    *,
    path: Optional[Path] = None,
) -> int:
    """Persist ``(text, embedding_bytes)`` rows for an artifact.

    Each chunk's ``embedding`` is stored as a BLOB; the storage layer
    does not interpret it. The expected producer is
    ``Embedder.encode(...)`` followed by ``arr.astype(np.float32)
    .tobytes()``. The full round-trip contract is pinned by
    ``EmbeddingBlobRoundTripTests`` in PR-B; PR-A only pins the
    write path so the CASCADE behaviour is testable.

    Args:
        artifact_id: The owning artifact. Must exist.
        chunks: An iterable of ``(text, embedding_bytes)`` pairs.
            The order of the iterable is the chunk's ``ord``.
        path: Override the on-disk location. ``None`` uses
            :func:`db_path`.

    Returns:
        The number of rows inserted.
    """
    rows = [
        (artifact_id, ord_, text, emb)
        for ord_, (text, emb) in enumerate(chunks)
    ]
    if not rows:
        return 0
    with _connect(path) as conn:
        cur = conn.executemany(
            """
            INSERT INTO chunks (artifact_id, ord, text, embedding)
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )
    return cur.rowcount


def add_chunks_returning_ids(
    artifact_id: str,
    chunks: Sequence[tuple[str, bytes]],
    *,
    path: Optional[Path] = None,
) -> List[int]:
    """Persist ``(text, embedding_bytes)`` rows and return their ``id``s.

    This is the variant the RAG store uses: it needs the chunk
    ids back so it can map FAISS index positions to source rows
    when reporting search hits. The ids are returned in the same
    order as the input sequence; ``chunk[i]`` -> ``ids[i]``.

    Args:
        artifact_id: The owning artifact. Must exist.
        chunks: An ordered sequence of ``(text, embedding_bytes)``
            pairs. The list materialisation is required so the
            ``id``s line up.
        path: Override the on-disk location. ``None`` uses
            :func:`db_path`.

    Returns:
        A new ``list[int]`` of chunk ``id``s, parallel to
        ``chunks``. Empty input returns ``[]``.
    """
    if not chunks:
        return []
    rows = [
        (artifact_id, ord_, text, emb)
        for ord_, (text, emb) in enumerate(chunks)
    ]
    with _connect(path) as conn:
        cur = conn.executemany(
            """
            INSERT INTO chunks (artifact_id, ord, text, embedding)
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )
        # ``lastrowid`` is the id of the *last* inserted row. With
        # AUTOINCREMENT and contiguous inserts, the ids form the
        # arithmetic sequence ``[lastrowid - n + 1, ..., lastrowid]``.
        # This is the SQLite-documented contract.
        n = cur.rowcount
        last = int(cur.lastrowid or 0)
        if last == 0:
            # Defensive: if lastrowid is unexpectedly 0, fall back
            # to re-selecting by (artifact_id, ord). This path is
            # only hit if the driver returns a degenerate cursor.
            placeholders = ",".join("?" * len(rows))
            ord_values = [r[1] for r in rows]
            sel = conn.execute(
                f"""
                SELECT id FROM chunks
                WHERE artifact_id = ?
                  AND ord IN ({placeholders})
                ORDER BY ord ASC
                """,
                (artifact_id, *ord_values),
            ).fetchall()
            return [int(r["id"]) for r in sel]
    if n <= 0:
        return []
    return list(range(last - n + 1, last + 1))


def list_chunks_for_chat(
    chat_id: str,
    *,
    path: Optional[Path] = None,
) -> List[dict]:
    """Return all chunks for a chat, joined through ``artifacts``.

    The RAG store rebuilds its per-chat FAISS index from this
    list. The join is necessary because ``chunks`` is keyed on
    ``artifact_id``, not on ``chat_id`` directly — a chat can
    hold several artifacts, each with its own chunk sequence.

    The result is ordered by ``artifacts.created_at`` then
    ``chunks.ord`` so the per-chat order is stable across
    rebuilds. The store relies on the order being deterministic
    to keep FAISS index positions aligned with chunk ids.

    Each row dict has the keys ``id``, ``artifact_id``,
    ``ord``, ``text``, ``embedding`` (the BLOB, ready to be
    rehydrated with ``np.frombuffer``).

    Args:
        chat_id: The chat to read chunks for. Chunks from
            soft-deleted chats are *not* returned (the join
            filters on ``chats.deleted_at IS NULL``).
        path: Override the on-disk location.

    Returns:
        A new ``list[dict]``. Empty list if the chat has no
        chunks, or does not exist, or is soft-deleted.
    """
    with _connect(path) as conn:
        rows = conn.execute(
            """
            SELECT c.id, c.artifact_id, c.ord, c.text, c.embedding
            FROM chunks AS c
            JOIN artifacts AS a ON a.id = c.artifact_id
            JOIN chats    AS ch ON ch.id = a.chat_id
            WHERE ch.id = ?
              AND ch.deleted_at IS NULL
            ORDER BY a.created_at ASC, c.ord ASC
            """,
            (chat_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_chat_id_for_artifact(
    artifact_id: str,
    *,
    path: Optional[Path] = None,
) -> Optional[str]:
    """Return the owning ``chat_id`` of an artifact, or ``None``.

    The RAG store needs this to bump the per-chat ``_index_version``
    counter on every ``add()`` call, so the next ``search(chat_id)``
    rebuilds the FAISS index. Returning ``None`` (rather than raising)
    means the caller can simply skip the version bump for a deleted
    artifact; nothing downstream breaks.

    Args:
        artifact_id: The artifact to look up.
        path: Override the on-disk location.

    Returns:
        The chat id, or ``None`` if the artifact does not exist.
    """
    with _connect(path) as conn:
        row = conn.execute(
            "SELECT chat_id FROM artifacts WHERE id = ?",
            (artifact_id,),
        ).fetchone()
    if row is None:
        return None
    return str(row["chat_id"])


# --- Repository: global corpus chunks (Phase 12 PR-E) ------------------------
#
# The global chunks are the *curated knowledge base* that every chat
# retrieves against. Each row is a chunk of public, openly-licensed
# security reference material (OWASP cheatsheets, MITRE ATT&CK, CWE,
# GTFOBins, Sigma rules) with provenance metadata embedded.
#
# Why a separate table from ``chunks``:
#   * User uploads can contain PII / company-confidential material; the
#     global corpus is public. Splitting prevents a sloppy join from
#     leaking a user's PDF into someone else's retrieval.
#   * The schema for global chunks does NOT need ``artifact_id`` (there
#     is no chat) and adds ``source_id`` / ``source_url`` / ``license``
#     for provenance footers. Adding columns to ``chunks`` would have
#     broken the existing 303 tests; a new table is additive.
#   * Refresh cadence is different. The global corpus is rebuilt by a
#     CLI (scripts/ingest_security_corpus.py) and replaced wholesale;
#     per-chat chunks are appended by user actions.
#
# License: rows are stored under their source's license. The CLI is
# responsible for NOT ingesting material whose license is not on the
# allow-list (CC-BY-4.0, CC-BY-SA-4.0, MIT, public-domain). The
# ``license`` column makes the legal provenance of every chunk explicit
# and queryable.


def add_global_chunk(
    source_id: str,
    source_url: str,
    license: str,
    text: str,
    embedding: bytes,
    *,
    ord: int = 0,
    path: Optional[Path] = None,
) -> int:
    """Insert one global chunk and return its ``id``.

    Args:
        source_id: Lowercase short id (e.g. ``"owasp"``, ``"mitre"``).
        source_url: Canonical URL of the source document the chunk
            came from. Used as the provenance citation in the prompt.
        license: SPDX-like license string (``"CC-BY-SA-4.0"`` etc.).
        text: The chunk text.
        embedding: ``np.float32.tobytes()`` blob.
        ord: Position of the chunk within the source document.
        path: Override the on-disk location.

    Returns:
        The new chunk ``id``.
    """
    now = _utcnow_iso()
    with _connect(path) as conn:
        cur = conn.execute(
            """
            INSERT INTO global_chunks
                (source_id, source_url, license, ord, text, embedding, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (source_id, source_url, license, ord, text, embedding, now),
        )
        return int(cur.lastrowid)


def add_global_chunks_batch(
    source_id: str,
    rows: Sequence[tuple[str, str, str, bytes]],
    *,
    path: Optional[Path] = None,
) -> List[int]:
    """Bulk-insert global chunks and return their ``id``s.

    Args:
        source_id: Lowercase short id applied to every row in the batch.
        rows: Sequence of ``(source_url, license, text, embedding_bytes)``
            tuples. ``ord`` is auto-assigned as the row's position in
            the sequence.
        path: Override the on-disk location.

    Returns:
        New chunk ``id``s in input order. Empty input returns ``[]``.

    Notes:
        Python's :mod:`sqlite3` module reports ``Cursor.rowcount == 0``
        after ``executemany`` on many Python versions. We therefore
        pre-compute the ``id`` range from the pre-insert ``MAX(id)``,
        which is correct because ``id`` is an ``INTEGER PRIMARY KEY``
        (auto-incrementing alias for ``ROWID``). This gives accurate
        ids in input order without depending on ``rowcount``.
    """
    if not rows:
        return []
    now = _utcnow_iso()
    payload = [
        (source_id, url, lic, i, text, emb, now)
        for i, (url, lic, text, emb) in enumerate(rows)
    ]
    with _connect(path) as conn:
        max_before = conn.execute(
            "SELECT COALESCE(MAX(id), 0) FROM global_chunks"
        ).fetchone()[0]
        conn.executemany(
            """
            INSERT INTO global_chunks
                (source_id, source_url, license, ord, text, embedding, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
        max_after = conn.execute(
            "SELECT COALESCE(MAX(id), 0) FROM global_chunks"
        ).fetchone()[0]
    n = max_after - max_before
    if n <= 0:
        return []
    # Sanity check: n must equal len(rows). If it doesn't, the
    # caller has hit a UNIQUE collision (e.g. re-ingesting into a
    # non-empty source with the same source_url+ord). We fall back
    # to counting in the DB by source_id+ord.
    if n != len(rows):
        with _connect(path) as conn:
            db_rows = conn.execute(
                "SELECT id FROM global_chunks WHERE source_id = ? "
                "ORDER BY source_url, ord",
                (source_id,),
            ).fetchall()
        return [int(r[0]) for r in db_rows]
    return list(range(max_before + 1, max_after + 1))


def list_global_chunks(
    source_id: Optional[str] = None,
    *,
    path: Optional[Path] = None,
) -> List[dict]:
    """Return all global chunks, optionally filtered by ``source_id``.

    The result is ordered by ``(source_id, source_url, ord)`` so the
    :class:`GlobalIndex` can rebuild per-source FAISS indices in a
    deterministic order — index position ``i`` always corresponds to
    the same row across rebuilds.

    Args:
        source_id: If set, return only chunks for that source.
        path: Override the on-disk location.

    Returns:
        A new ``list`` of ``dict`` rows with keys ``id``, ``source_id``,
        ``source_url``, ``license``, ``ord``, ``text``, ``embedding``.
    """
    if source_id is not None:
        sql = (
            "SELECT id, source_id, source_url, license, ord, text, embedding "
            "FROM global_chunks WHERE source_id = ? "
            "ORDER BY source_id, source_url, ord"
        )
        params: tuple = (source_id,)
    else:
        sql = (
            "SELECT id, source_id, source_url, license, ord, text, embedding "
            "FROM global_chunks "
            "ORDER BY source_id, source_url, ord"
        )
        params = ()
    with _connect(path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def count_global_chunks(
    source_id: Optional[str] = None,
    *,
    path: Optional[Path] = None,
) -> int:
    """Return the number of indexed global chunks, optionally per-source.

    Args:
        source_id: If set, count only that source.
        path: Override the on-disk location.

    Returns:
        The chunk count. ``0`` for a fresh DB or unknown source.
    """
    if source_id is not None:
        sql = "SELECT COUNT(*) AS n FROM global_chunks WHERE source_id = ?"
        params: tuple = (source_id,)
    else:
        sql = "SELECT COUNT(*) AS n FROM global_chunks"
        params = ()
    with _connect(path) as conn:
        row = conn.execute(sql, params).fetchone()
    return int(row["n"])


def list_corpus_sources(
    *, path: Optional[Path] = None
) -> List[dict]:
    """Return the distinct sources currently indexed in the global corpus.

    Each returned dict has ``source_id`` and ``chunk_count`` and
    ``distinct_docs`` (number of unique ``source_url`` values). This is
    the data the sidebar "Knowledge base" panel renders.

    Args:
        path: Override the on-disk location.

    Returns:
        A new ``list`` of ``dict`` rows ordered by ``source_id`` ASC.
    """
    sql = (
        "SELECT source_id, "
        "       COUNT(*) AS chunk_count, "
        "       COUNT(DISTINCT source_url) AS distinct_docs "
        "FROM global_chunks "
        "GROUP BY source_id "
        "ORDER BY source_id ASC"
    )
    with _connect(path) as conn:
        rows = conn.execute(sql).fetchall()
    return [dict(r) for r in rows]


def clear_global_corpus(
    source_id: Optional[str] = None,
    *,
    path: Optional[Path] = None,
) -> int:
    """Delete global chunks, optionally scoped to one ``source_id``.

    Used by ``scripts/ingest_security_corpus.py --clear`` and by the
    sidebar "Rebuild" button. Returns the number of rows deleted so the
    operator sees what happened in the log.

    Args:
        source_id: If set, delete only that source. If ``None``,
            delete **all** global chunks (full reset).
        path: Override the on-disk location.

    Returns:
        The number of rows deleted.
    """
    if source_id is not None:
        sql = "DELETE FROM global_chunks WHERE source_id = ?"
        params: tuple = (source_id,)
    else:
        sql = "DELETE FROM global_chunks"
        params = ()
    with _connect(path) as conn:
        cur = conn.execute(sql, params)
    return int(cur.rowcount)


# --- Recon audit log ---------------------------------------------------------
# Phase 15 — every recon tool invocation writes one row here so an operator
# can answer "who ran which tool against which target, and when". The schema
# is intentionally narrow: a recon call is a small, structured event, not a
# document. ``result_excerpt`` is the first 500 chars of the rendered output
# so an investigator can scan the audit without re-running the tool, but
# the full payload is NOT persisted (privacy + storage cost). ``duration_ms``
# is measured wall-clock at the call site, not by SQLite, so it includes
# network latency and the orchestrator's own overhead.


_RECON_TOOL_STATUSES: tuple[str, ...] = ("ok", "blocked", "error")


def log_recon_request(
    *,
    target: str,
    tool: str,
    scope_token: str | None = None,
    chat_id: str | None = None,
    status: str = "ok",
    duration_ms: int | None = None,
    result_excerpt: str | None = None,
    path: Path | None = None,
) -> int:
    """Append one row to ``recon_log`` and return the new id.

    The function is the audit-log primitive for Phase 15. Every successful
    OR failed recon call should land here so the operator has a single
    place to audit who ran what against which target. ``status`` is the
    only field the caller controls: ``"ok"`` for a successful tool call,
    ``"blocked"`` when the safety rail rejected the target, and
    ``"error"`` for a transport / parse / network failure.

    ``target`` and ``tool`` are required and non-empty; a missing target
    is a programming error, not a recoverable runtime condition. The
    function does NOT validate that ``target`` is safe — the safety rail
    is the orchestrator's job, and a blocked call still gets logged with
    ``status="blocked"`` so the attempt itself is auditable.

    The return value is the autoincrement primary key, mirroring
    :func:`add_global_chunk` and :func:`add_chunks_returning_ids`. Tests
    use it to assert the row was actually written.
    """
    if not target or not target.strip():
        raise ValueError("log_recon_request requires a non-empty target")
    if not tool or not tool.strip():
        raise ValueError("log_recon_request requires a non-empty tool")
    if status not in _RECON_TOOL_STATUSES:
        raise ValueError(
            f"log_recon_request: status must be one of {_RECON_TOOL_STATUSES!r}, "
            f"got {status!r}"
        )
    excerpt: str | None = None
    if result_excerpt:
        # Truncate to 500 chars so a multi-KB WHOIS blob cannot bloat the
        # audit log. 500 is enough for an operator to recognise the call.
        excerpt = result_excerpt[:500]
    sql = (
        "INSERT INTO recon_log "
        "(chat_id, target, tool, scope_token, status, timestamp, "
        " duration_ms, result_excerpt) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
    )
    params = (
        chat_id,
        target.strip(),
        tool.strip(),
        scope_token,
        status,
        _utcnow_iso(),
        duration_ms,
        excerpt,
    )
    with _connect(path) as conn:
        cur = conn.execute(sql, params)
    return int(cur.lastrowid or 0)


def list_recon_for_chat(
    chat_id: str,
    *,
    limit: int = 50,
    path: Path | None = None,
) -> list[dict]:
    """Return the most recent ``recon_log`` rows for ``chat_id``, newest first.

    Used by the sidebar / per-chat audit panel in the Streamlit view.
    A non-positive ``limit`` raises ``ValueError`` so a typo in the
    caller cannot silently return zero rows. The function is a thin
    SELECT — no pagination, no aggregation, no time-window filter —
    because the audit panel only ever wants the tail.
    """
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    sql = (
        "SELECT id, chat_id, target, tool, scope_token, status, "
        "       timestamp, duration_ms, result_excerpt "
        "FROM recon_log "
        "WHERE chat_id = ? "
        "ORDER BY id DESC "
        "LIMIT ?"
    )
    with _connect(path) as conn:
        rows = conn.execute(sql, (chat_id, limit)).fetchall()
    return [dict(r) for r in rows]


def list_recent_recon(
    *,
    target: str | None = None,
    limit: int = 20,
    path: Path | None = None,
) -> list[dict]:
    """Return the most recent ``recon_log`` rows, optionally filtered by target.

    Used by the operator-facing dashboard (out of scope for Phase 15,
    but the function is here so the test suite can pin the contract
    now). ``target`` is matched with ``LIKE '%<target>%'`` so a partial
    string ("github") finds rows whose target was "github.com" or
    "api.github.com" — the goal is *recall* in an audit search, not
    exact-string lookup.
    """
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    if target:
        sql = (
            "SELECT id, chat_id, target, tool, scope_token, status, "
            "       timestamp, duration_ms, result_excerpt "
            "FROM recon_log "
            "WHERE target LIKE ? "
            "ORDER BY id DESC "
            "LIMIT ?"
        )
        params: tuple = (f"%{target}%", limit)
    else:
        sql = (
            "SELECT id, chat_id, target, tool, scope_token, status, "
            "       timestamp, duration_ms, result_excerpt "
            "FROM recon_log "
            "ORDER BY id DESC "
            "LIMIT ?"
        )
        params = (limit,)
    with _connect(path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def count_recon_requests(
    chat_id: str | None = None,
    *,
    path: Path | None = None,
) -> int:
    """Return the count of ``recon_log`` rows, optionally scoped to ``chat_id``.

    Used by the chat-history sidebar to show "12 recon calls this
    chat" badges without pulling the full row set. Returns 0 (not
    ``None``) when no rows match, so callers can use the value in an
    arithmetic expression without a ``None`` check.
    """
    with _connect(path) as conn:
        if chat_id:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM recon_log WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM recon_log"
            ).fetchone()
    return int(row["n"] or 0)


# --- Public re-exports ------------------------------------------------------

__all__ = [
    "db_path",
    "init_db",
    # chat repo
    "create_chat",
    "list_chats",
    "get_chat",
    "soft_delete_chat",
    "hard_delete_chat",
    "touch_chat",
    # message repo
    "append_message",
    "load_messages",
    # artifact / chunk repo
    "add_artifact",
    "list_artifacts",
    "delete_artifact",
    "add_chunks",
    "add_chunks_returning_ids",
    "list_chunks_for_chat",
    "get_chat_id_for_artifact",
    # global corpus repo
    "add_global_chunk",
    "add_global_chunks_batch",
    "list_global_chunks",
    "count_global_chunks",
    "list_corpus_sources",
    "clear_global_corpus",
    # recon audit repo
    "log_recon_request",
    "list_recon_for_chat",
    "list_recent_recon",
    "count_recon_requests",
]

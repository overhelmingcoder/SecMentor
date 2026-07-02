"""Tests for the Phase 12 persistence layer (``app/storage.py``).

These tests cover PR-A of ``docs/phase_12_rag_and_history.md`` — the
storage layer. PR-A is the **first** PR in the phase; it ships:

* the canonical SQLite schema (pinned in :data:`_SCHEMA_SQL`),
* the :func:`init_db` idempotency guarantee,
* the ON DELETE CASCADE behaviour for ``chats → messages`` and
  ``chats → artifacts → chunks``,
* the chat repository (create / list / get / soft-delete /
  hard-delete),
* the message repository (append / load, with the
  text-vs-multimodal content contract),
* the artifact + chunk stub repositories (enough to pin the
  cascade and the BLOB write path).

The RAG-side tests (RagStore, embedder integration, FAISS
search) live in ``tests/test_rag.py`` and ship with PR-B.

All tests run against an in-memory SQLite (``":memory:"``) so the
test suite has no on-disk side effects and can run in parallel
without coordination. The ``_mem_path`` helper below wraps the
"``":memory:"`` is a fresh database per connection" gotcha — we
keep a single connection open via the ``_connect`` context
manager, so the schema and the rows share the same in-memory
database for the lifetime of the test.

Run with:  python -m unittest tests.test_storage -v
"""

import json
import os
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke


# --- Path bootstrap ---------------------------------------------------------
# Mirror the project-root-cd pattern used in tests/test_smoke.py and
# tests/test_files.py so this file works no matter where the user
# invokes `python -m unittest` from.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _fresh_db_path() -> Path:
    """Return a path to a brand-new, empty SQLite file.

    Each call returns a unique path so two tests in the same
    process get two isolated databases. ``sqlite3.connect``
    creates an empty database file on first connection, which
    is exactly what we want — no leftover rows from previous
    tests, and the file is removed by ``_CleanupTempDir``
    below (or by the OS if the test crashes).
    """
    # ``mkstemp`` is atomic and returns a (fd, path) pair. We
    # close the fd immediately; ``sqlite3.connect`` will open
    # its own handle. ``suffix=".db"`` so the file is obvious
    # in a debugger.
    fd, path = tempfile.mkstemp(suffix=".db", prefix="secmentor_test_")
    os.close(fd)
    return Path(path)


class _CleanupTempPath:
    """Context-manager-friendly helper for tests that own a temp DB.

    Each test in this module creates a fresh SQLite file via
    :func:`_fresh_db_path`. The ``unlink`` in ``tearDown`` makes
    sure the file is removed even if the assertion failed. This
    keeps the test runner's working directory clean and avoids
    the "lots of secmentor_test_*.db files in $TEMP" surprise.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def cleanup(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


# --- Constants --------------------------------------------------------------
# A handful of well-known values used across tests. Keeping them here
# (not at the class level) means the tests read top-to-bottom without
# scrolling, and the test author can tweak a single value without
# hunting through the file.
_SAMPLE_TITLE_A: str = "What is prompt injection?"
_SAMPLE_TITLE_B: str = "How do I do a SQLi demo in a lab?"
_VALID_ROLES: tuple[str, ...] = ("user", "assistant", "system")


# --- Tests ------------------------------------------------------------------


class StorageSchemaTests(unittest.TestCase):
    """Pin the schema shape, ``init_db`` idempotency, and FK cascades.

    Acceptance criteria covered here (from PR-A):

    * ``init_db()`` is idempotent (re-running it is a no-op).
    * ``ON DELETE CASCADE`` actually cascades from chats → messages
      and chats → artifacts → chunks.
    * The schema has the expected tables and indices.
    """

    def setUp(self):
        # A fresh per-test SQLite file. SQLite's ``":memory:"``
        # token gives a fresh database *per connection*, which
        # would discard the schema between ``init_db`` and the
        # next repository call. A per-test file is the standard
        # pytest pattern for SQLite tests and keeps the schema
        # alive for the whole test.
        from app import storage
        self.storage = storage
        self.path = _fresh_db_path()
        self._cleanup = _CleanupTempPath(self.path)
        self.storage.init_db(self.path)

    def tearDown(self) -> None:
        self._cleanup.cleanup()

    def test_init_db_is_idempotent(self):
        """Running ``init_db`` twice on the same path must succeed
        and produce the same schema. Pinned by PR-A acceptance
        criterion (a)."""
        # setUp already called init_db once; this call is the
        # "second" one and must be a no-op.
        self.storage.init_db(self.path)

    def test_init_db_creates_all_expected_tables(self):
        """Every table pinned in §3 of the Phase 12 doc must exist
        after ``init_db``."""
        with self.storage._connect(self.path) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            ).fetchall()
        table_names = {r["name"] for r in rows}
        self.assertEqual(
            table_names,
            {"chats", "messages", "artifacts", "chunks", "global_chunks"},
            f"expected the five pinned tables; got {sorted(table_names)}",
        )

    def test_init_db_creates_all_expected_indices(self):
        """The pinned indices must exist after ``init_db``.

        The ``idx_*`` indices are the ones called out in §3 — they
        keep the per-chat message and artifact lookups from doing
        a full table scan. If a future refactor drops one of them
        the test fails immediately rather than in production
        when a chat grows to thousands of messages.
        """
        with self.storage._connect(self.path) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'index' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            ).fetchall()
        index_names = {r["name"] for r in rows}
        self.assertEqual(
            index_names,
            {
                "idx_messages_chat",
                "idx_artifacts_chat",
                "idx_chunks_artifact",
                "idx_global_chunks_source",
            },
            f"missing or extra indices: {sorted(index_names)}",
        )

    def test_chat_cascade_deletes_messages(self):
        """Deleting a chat must cascade to its messages. This is
        the most important FK cascade in the schema — if it
        silently no-ops, a single hard-delete would leave
        orphaned rows and the next ``list_chats`` would show
        empty entries.

        Pinned by PR-A acceptance criterion (b).
        """
        cid = self.storage.create_chat(
            title="cascade test", chat_id="cid-cascade", path=self.path
        )
        self.storage.append_message(
            cid, role="user", content="hi", path=self.path
        )
        self.storage.append_message(
            cid, role="assistant", content="hello", path=self.path
        )
        # Sanity: two rows in `messages` for this chat.
        with self.storage._connect(self.path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) AS n FROM messages WHERE chat_id = ?",
                (cid,),
            ).fetchone()["n"]
        self.assertEqual(count, 2, "precondition: messages were inserted")
        # Hard-delete the chat and assert the messages are gone.
        self.assertTrue(self.storage.hard_delete_chat(cid, path=self.path))
        with self.storage._connect(self.path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) AS n FROM messages WHERE chat_id = ?",
                (cid,),
            ).fetchone()["n"]
        self.assertEqual(
            count, 0,
            "hard_delete_chat must cascade to messages (FK ON DELETE CASCADE)",
        )

    def test_chat_cascade_deletes_artifacts_and_chunks(self):
        """The ``chats → artifacts → chunks`` cascade must work
        end-to-end. This is the second leg of PR-A acceptance
        criterion (b) and pins the schema's three-level FK
        relationship. A regression here would leave orphan
        chunks that the retriever would happily return
        against a different chat's id."""
        cid = self.storage.create_chat(
            title="art-cascade", chat_id="cid-art", path=self.path
        )
        aid = self.storage.add_artifact(
            cid,
            filename="spec.pdf",
            mime="application/pdf",
            size_bytes=1234,
            artifact_id="aid-1",
            path=self.path,
        )
        # Two chunks under the artifact.
        self.storage.add_chunks(
            aid,
            [("alpha", b"\x00" * 8), ("beta", b"\x00" * 8)],
            path=self.path,
        )
        # Sanity: 1 artifact, 2 chunks.
        with self.storage._connect(self.path) as conn:
            n_art = conn.execute(
                "SELECT COUNT(*) AS n FROM artifacts WHERE chat_id = ?",
                (cid,),
            ).fetchone()["n"]
            n_chunk = conn.execute(
                "SELECT COUNT(*) AS n FROM chunks "
                "WHERE artifact_id = ?",
                (aid,),
            ).fetchone()["n"]
        self.assertEqual((n_art, n_chunk), (1, 2), "precondition fixture")
        # Hard-delete the chat. CASCADE must wipe both.
        self.storage.hard_delete_chat(cid, path=self.path)
        with self.storage._connect(self.path) as conn:
            n_art = conn.execute(
                "SELECT COUNT(*) AS n FROM artifacts WHERE chat_id = ?",
                (cid,),
            ).fetchone()["n"]
            n_chunk = conn.execute(
                "SELECT COUNT(*) AS n FROM chunks "
                "WHERE artifact_id = ?",
                (aid,),
            ).fetchone()["n"]
        self.assertEqual(
            (n_art, n_chunk), (0, 0),
            "chats → artifacts → chunks CASCADE failed; orphan rows remain",
        )

    def test_artifact_cascade_deletes_chunks(self):
        """Deleting a single artifact must cascade to its chunks
        but leave the chat and other artifacts alone."""
        cid = self.storage.create_chat(
            title="art-only cascade", chat_id="cid-art2", path=self.path
        )
        a1 = self.storage.add_artifact(
            cid, filename="a.pdf", mime="application/pdf",
            size_bytes=10, artifact_id="aid-a", path=self.path,
        )
        a2 = self.storage.add_artifact(
            cid, filename="b.pdf", mime="application/pdf",
            size_bytes=20, artifact_id="aid-b", path=self.path,
        )
        self.storage.add_chunks(a1, [("x", b"\x00" * 4)], path=self.path)
        self.storage.add_chunks(a2, [("y", b"\x00" * 4)], path=self.path)
        # Delete a1. a2's chunk must survive.
        self.assertTrue(self.storage.delete_artifact(a1, path=self.path))
        with self.storage._connect(self.path) as conn:
            self.assertEqual(
                conn.execute(
                    "SELECT COUNT(*) AS n FROM chunks "
                    "WHERE artifact_id = ?", (a2,)
                ).fetchone()["n"], 1,
                "deleting a1 must not affect a2's chunks",
            )

    def test_db_path_honours_env_override(self):
        """``SECMENTOR_DB_PATH`` must win over the default. We
        re-implement the override check inline so the test
        does not depend on monkeypatching a module constant —
        the function reads the env on every call (see
        :func:`app.storage.db_path`).

        We use a per-test temp file so the test never touches a
        real on-disk path on the CI machine. The expected value
        is the *resolved* (absolute, ``~``-expanded) form —
        the storage layer normalises the input so a config
        like ``"~/db.sqlite"`` always lands at the same path.
        """
        from app import storage
        sentinel = _fresh_db_path().resolve()
        old = os.environ.get("SECMENTOR_DB_PATH")
        try:
            os.environ["SECMENTOR_DB_PATH"] = str(sentinel)
            self.assertEqual(
                storage.db_path(), sentinel,
                "db_path() must read SECMENTOR_DB_PATH on every call",
            )
        finally:
            if old is None:
                os.environ.pop("SECMENTOR_DB_PATH", None)
            else:
                os.environ["SECMENTOR_DB_PATH"] = old


class ChatRepositoryTests(unittest.TestCase):
    """Pin the chat + message repository surface.

    Acceptance criteria covered here (from PR-A):

    * ``list_chats(limit=20)`` returns chats in ``updated_at DESC``
      order — pin: d (newest first).
    * ``soft_delete_chat()`` excludes the row from
      ``list_chats()`` but does not erase the data — pin: e
      (future "undo" is one UPDATE away).
    * All repository functions return a *new* list, not a view
      onto the SQLite row — pin: c.
    * Repository functions handle the multimodal ``content`` shape
      correctly (JSON envelope in, plain list out).
    """

    def setUp(self):
        from app import storage
        self.storage = storage
        self.path = _fresh_db_path()
        self.storage.init_db(self.path)

    # --- list_chats / ordering -------------------------------------------

    def test_list_chats_returns_newest_first(self):
        """Two chats created in a known order must come back with
        the later one first. We touch the first chat to force
        ``updated_at`` to advance so the ordering test is
        deterministic (a sub-second ``created_at`` would be
        ambiguous on fast hardware)."""
        a = self.storage.create_chat(
            title="older", chat_id="cid-a", path=self.path
        )
        b = self.storage.create_chat(
            title="newer", chat_id="cid-b", path=self.path
        )
        # Touch `a` so its updated_at jumps to "now" AFTER `b`'s
        # creation. The list should still rank `a` first because
        # touched-now is the most recent activity.
        self.storage.touch_chat(a, path=self.path)
        rows = self.storage.list_chats(path=self.path)
        self.assertEqual(
            [r["id"] for r in rows], [a, b],
            "list_chats must order by updated_at DESC",
        )

    def test_list_chats_respects_limit(self):
        """``list_chats(limit=N)`` must return at most N rows even
        when more chats exist.

        The schema stores ``updated_at`` as ISO-second text, so
        five inserts inside the same second produce equal
        timestamps and the ``ORDER BY updated_at DESC`` becomes
        a no-op (SQLite is free to return rows in any order for
        ties). We sleep between inserts to give each chat a
        distinct timestamp — this mirrors the pattern in
        ``test_append_message_bumps_chat_updated_at`` and pins
        the intent: limit means "two most recently *updated*".
        """
        import time
        ids = []
        for i in range(5):
            ids.append(self.storage.create_chat(
                title=f"chat-{i}", chat_id=f"cid-{i}", path=self.path
            ))
            time.sleep(1.05)  # advance the ISO-second clock
        rows = self.storage.list_chats(limit=2, path=self.path)
        self.assertEqual(len(rows), 2)
        # The first two returned are the two most recently updated.
        self.assertEqual({r["id"] for r in rows}, {ids[-1], ids[-2]})

    def test_list_chats_rejects_non_positive_limit(self):
        """A zero or negative limit is a programming error; raise
        loudly rather than silently returning all rows."""
        with self.assertRaises(ValueError):
            self.storage.list_chats(limit=0, path=self.path)
        with self.assertRaises(ValueError):
            self.storage.list_chats(limit=-3, path=self.path)

    # --- "return a new list" contract ------------------------------------

    def test_list_chats_returns_independent_list(self):
        """Mutating the list returned by ``list_chats`` must not
        affect a subsequent call. This is acceptance criterion
        (c) of PR-A: callers must be free to ``.sort()`` or
        ``.append()`` the result without corrupting the DB.
        """
        self.storage.create_chat(
            title="t", chat_id="cid-x", path=self.path
        )
        first = self.storage.list_chats(path=self.path)
        first.clear()
        first.append({"sentinel": True})
        # A fresh call returns the real list, untouched.
        second = self.storage.list_chats(path=self.path)
        self.assertEqual(len(second), 1)
        self.assertNotIn("sentinel", second[0])

    def test_load_messages_returns_independent_list(self):
        """Same contract as :meth:`list_chats`: the returned
        list is a fresh Python ``list``, not a view onto the
        SQLite row iterator."""
        cid = self.storage.create_chat(
            title="t", chat_id="cid-y", path=self.path
        )
        self.storage.append_message(
            cid, role="user", content="hi", path=self.path
        )
        first = self.storage.load_messages(cid, path=self.path)
        first.clear()
        second = self.storage.load_messages(cid, path=self.path)
        self.assertEqual(len(second), 1)

    # --- soft delete + hard delete ---------------------------------------

    def test_soft_delete_hides_chat_from_list(self):
        """A soft-deleted chat must not appear in ``list_chats``
        (default), but its data must still be intact on disk
        — acceptance criterion (e)."""
        cid = self.storage.create_chat(
            title="to be deleted", chat_id="cid-d", path=self.path
        )
        self.storage.append_message(
            cid, role="user", content="remember me", path=self.path
        )
        self.assertTrue(self.storage.soft_delete_chat(cid, path=self.path))
        # Default list: gone.
        self.assertEqual(self.storage.list_chats(path=self.path), [])
        # Including deleted: still there.
        with_deleted = self.storage.list_chats(
            include_deleted=True, path=self.path
        )
        self.assertEqual(len(with_deleted), 1)
        self.assertIsNotNone(with_deleted[0]["deleted_at"])
        # The message is still on disk (so a future "undo" can
        # restore the chat by clearing `deleted_at`).
        msgs = self.storage.load_messages(cid, path=self.path)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["content"], "remember me")

    def test_soft_delete_returns_false_for_already_deleted(self):
        """A second ``soft_delete_chat`` on the same id must
        return ``False`` so callers can tell "this is the
        transition" from "this was already gone"."""
        cid = self.storage.create_chat(
            title="once", chat_id="cid-twice", path=self.path
        )
        self.assertTrue(self.storage.soft_delete_chat(cid, path=self.path))
        self.assertFalse(self.storage.soft_delete_chat(cid, path=self.path))

    def test_hard_delete_removes_chat_and_its_messages(self):
        """A hard delete must remove the chat and cascade to its
        messages — the destructive counterpart to soft delete.
        """
        cid = self.storage.create_chat(
            title="goodbye", chat_id="cid-bye", path=self.path
        )
        self.storage.append_message(
            cid, role="user", content="x", path=self.path
        )
        self.assertTrue(self.storage.hard_delete_chat(cid, path=self.path))
        self.assertIsNone(self.storage.get_chat(cid, path=self.path))
        # And the messages are gone too.
        self.assertEqual(
            self.storage.load_messages(cid, path=self.path), []
        )

    # --- chat row + teaching_mode ---------------------------------------

    def test_create_chat_persists_teaching_mode(self):
        """``teaching_mode`` is denormalised onto the chat row so
        the sidebar can render the persona label without a
        join. A round-trip must preserve it exactly."""
        cid = self.storage.create_chat(
            title="mentor chat",
            teaching_mode="mentor",
            chat_id="cid-mentor",
            path=self.path,
        )
        self.assertEqual(
            self.storage.get_chat(cid, path=self.path)["teaching_mode"],
            "mentor",
        )
        cid2 = self.storage.create_chat(
            title="defensive chat",
            teaching_mode="defensive",
            chat_id="cid-defensive",
            path=self.path,
        )
        self.assertEqual(
            self.storage.get_chat(cid2, path=self.path)["teaching_mode"],
            "defensive",
        )

    def test_get_chat_returns_none_for_unknown_id(self):
        """A missing id is not an error — return ``None`` so the
        view can branch (``st.warning("chat not found")``)
        without try/except around the read."""
        self.assertIsNone(
            self.storage.get_chat("not-a-real-id", path=self.path)
        )

    # --- message append / load (text + multimodal) ----------------------

    def test_append_message_increments_ord(self):
        """Each append must compute ``ord`` as ``MAX(ord) + 1`` so
        the conversation can be replayed in order. Pin the
        ordering is stable across many appends."""
        cid = self.storage.create_chat(
            title="ord", chat_id="cid-ord", path=self.path
        )
        for content in ("u1", "a1", "u2", "a2", "u3"):
            self.storage.append_message(
                cid, role="user" if content.startswith("u") else "assistant",
                content=content, path=self.path,
            )
        msgs = self.storage.load_messages(cid, path=self.path)
        self.assertEqual(
            [m["content"] for m in msgs], ["u1", "a1", "u2", "a2", "u3"],
        )
        # ``ord`` is contiguous starting at 0.
        self.assertEqual([m["ord"] for m in msgs], [0, 1, 2, 3, 4])

    def test_append_message_rejects_unknown_role(self):
        """``role`` is constrained to ``user | assistant | system``.
        A typo (``"bot"``) must fail loudly at the storage
        boundary, not silently insert a row the LLM would
        never recognise."""
        cid = self.storage.create_chat(
            title="bad role", chat_id="cid-bad-role", path=self.path
        )
        with self.assertRaises(ValueError):
            self.storage.append_message(
                cid, role="bot", content="hi", path=self.path
            )

    def test_append_message_bumps_chat_updated_at(self):
        """Appending a message must bump the chat's ``updated_at``
        so the sidebar reorders correctly. This is the
        no-trigger policy: the storage layer is the writer,
        and the repository is the place that bumps the
        timestamp."""
        cid = self.storage.create_chat(
            title="ts", chat_id="cid-ts", path=self.path
        )
        before = self.storage.get_chat(cid, path=self.path)["updated_at"]
        # Force a clock advance so the second timestamp is
        # strictly greater (ISO second resolution is the
        # schema's clock granularity).
        # The actual bump happens inside ``append_message``;
        # we just check the on-disk value moved.
        import time
        time.sleep(1.05)
        self.storage.append_message(
            cid, role="user", content="tick", path=self.path
        )
        after = self.storage.get_chat(cid, path=self.path)["updated_at"]
        self.assertGreater(
            after, before,
            f"updated_at must advance on append; before={before!r} after={after!r}",
        )

    def test_load_messages_decodes_text_turns_unchanged(self):
        """A plain string turn is stored and loaded as a plain
        string — no JSON envelope in the way. This is the
        common case (text-only chat) and must not pay a
        JSON parse cost."""
        cid = self.storage.create_chat(
            title="text", chat_id="cid-text", path=self.path
        )
        self.storage.append_message(
            cid, role="user", content="hello", path=self.path
        )
        msgs = self.storage.load_messages(cid, path=self.path)
        self.assertEqual(len(msgs), 1)
        self.assertIsInstance(msgs[0]["content"], str)
        self.assertEqual(msgs[0]["content"], "hello")

    def test_load_messages_decodes_multimodal_turns(self):
        """A list-of-parts turn is stored as JSON and rehydrated
        to a list of parts. This is the contract pinned by
        §3 design note 5 of the Phase 12 doc — the storage
        layer is the only place that knows about the JSON
        envelope."""
        cid = self.storage.create_chat(
            title="mm", chat_id="cid-mm", path=self.path
        )
        parts = [
            {"type": "text", "text": "what is in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
        ]
        self.storage.append_message(
            cid, role="user", content=parts, path=self.path
        )
        msgs = self.storage.load_messages(cid, path=self.path)
        self.assertEqual(len(msgs), 1)
        loaded = msgs[0]["content"]
        self.assertIsInstance(loaded, list)
        self.assertEqual(loaded, parts)

    def test_append_message_round_trips_unicode(self):
        """A real chat will have emoji and non-ASCII text. The
        storage round-trip must preserve the bytes (UTF-8)
        exactly. Pins the schema's TEXT type defaulting to
        UTF-8, which is what the engine depends on."""
        cid = self.storage.create_chat(
            title="unicode", chat_id="cid-uni", path=self.path
        )
        text = "🔐 Explain § 4.2 in 日本語."
        self.storage.append_message(
            cid, role="user", content=text, path=self.path
        )
        msgs = self.storage.load_messages(cid, path=self.path)
        self.assertEqual(msgs[0]["content"], text)


class ArtifactRepositoryTests(unittest.TestCase):
    """Pin the artifact repository surface.

    Not in the original PR-A list, but the FK cascade test in
    StorageSchemaTests already exercises ``add_artifact`` and
    ``add_chunks``; this class pins their public contract so a
    bad refactor in PR-B (RAG pipeline) cannot silently change
    the schema's caller surface.
    """

    def setUp(self):
        from app import storage
        self.storage = storage
        self.path = _fresh_db_path()
        self.storage.init_db(self.path)
        self.cid = self.storage.create_chat(
            title="art repo", chat_id="cid-ar", path=self.path
        )

    def test_add_artifact_returns_id_and_lists(self):
        """An added artifact must round-trip via ``list_artifacts``
        with the same filename, mime, and size."""
        aid = self.storage.add_artifact(
            self.cid,
            filename="paper.pdf",
            mime="application/pdf",
            size_bytes=4096,
            path=self.path,
        )
        self.assertIsInstance(aid, str)
        self.assertEqual(len(aid), 32, "uuid4 hex is 32 chars")
        rows = self.storage.list_artifacts(self.cid, path=self.path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], aid)
        self.assertEqual(rows[0]["filename"], "paper.pdf")
        self.assertEqual(rows[0]["mime"], "application/pdf")
        self.assertEqual(rows[0]["size_bytes"], 4096)

    def test_add_chunks_persists_ord_and_bytes(self):
        """``add_chunks`` must persist each row with the iterable
        index as ``ord`` and the exact bytes supplied. PR-B
        will pin the numpy round-trip; PR-A pins the write
        path so the cascade test is meaningful."""
        aid = self.storage.add_artifact(
            self.cid, filename="x.bin", mime="application/octet-stream",
            size_bytes=0, artifact_id="aid-x", path=self.path,
        )
        chunks = [
            ("alpha", b"\x01\x02\x03"),
            ("beta", b"\xff\xfe\xfd"),
            ("gamma", b""),
        ]
        n = self.storage.add_chunks(aid, chunks, path=self.path)
        self.assertEqual(n, 3)
        with self.storage._connect(self.path) as conn:
            rows = conn.execute(
                "SELECT ord, text, embedding FROM chunks "
                "WHERE artifact_id = ? ORDER BY ord ASC",
                (aid,),
            ).fetchall()
        self.assertEqual(
            [r["ord"] for r in rows], [0, 1, 2],
        )
        self.assertEqual(
            [r["text"] for r in rows], ["alpha", "beta", "gamma"],
        )
        self.assertEqual(
            [bytes(r["embedding"]) for r in rows],
            [b"\x01\x02\x03", b"\xff\xfe\xfd", b""],
        )

    def test_add_chunks_empty_iterable_is_noop(self):
        """``add_chunks([])`` must return 0 and insert no rows.
        This is the path taken by a no-text scanned PDF (see
        ``improvements.md`` C.13) — the chunker produces no
        chunks, the storage layer accepts that gracefully,
        and the artifact remains in the table without any
        chunk rows."""
        aid = self.storage.add_artifact(
            self.cid, filename="empty.pdf", mime="application/pdf",
            size_bytes=10, artifact_id="aid-empty", path=self.path,
        )
        n = self.storage.add_chunks(aid, [], path=self.path)
        self.assertEqual(n, 0)
        with self.storage._connect(self.path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) AS n FROM chunks WHERE artifact_id = ?",
                (aid,),
            ).fetchone()["n"]
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()

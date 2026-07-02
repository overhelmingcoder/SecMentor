"""Global corpus retrieval: per-source FAISS indices over the curated knowledge base.

This module is the **retrieval** half of the global corpus feature
introduced in Phase 12 PR-E. It sits next to :mod:`app.rag_store`
(per-chat retrieval) and is the second index the Streamlit view
queries before building a prompt.

Architecture
------------
* **Per-source FAISS indices.** One ``IndexFlatIP`` per
  ``source_id`` (``owasp``, ``mitre``, …). A single global index
  would also work, but per-source indices give us:
    - Cheap per-source filtering at query time (``source_filter=``).
    - Cheap refresh: re-indexing a single source rebuilds one
      small index, not the whole corpus.
    - Easy "enable / disable" semantics: a source with no index
      returns ``[]`` immediately.

* **Lazy build.** Indices are built on first ``search()`` for a
  given ``source_id`` from the rows in :class:`app.rag_store`…
  actually, from the new ``global_chunks`` table introduced in
  PR-E. We never load the whole table into memory at import time;
  ``add_source`` writes to SQLite + invalidates the cache, and
  ``search`` triggers a per-source rebuild on cache miss.

* **Cosine similarity via inner product.** Vectors are L2-normalised
  at ``add_source`` time, and the index uses inner product. This
  is the same recipe :class:`app.rag_store.RagStore` uses for
  per-chat retrieval, so the two indices behave identically at
  query time and can be merged by raw score.

* **Degraded mode is silent.** If the embedder is ``None`` or
  reports ``is_available() == False`` — or if FAISS itself fails
  to import — :meth:`GlobalIndex.search` returns ``[]`` and never
  raises. The Streamlit view treats an empty result as "no
  relevant context" and falls back to the per-chat index plus the
  sentinel block. This is the same contract :class:`RagStore`
  honours, so the view's error handling is uniform.

Public surface
--------------
* :class:`GlobalIndex` — the class the view holds in
  ``st.cache_resource``. Constructed with an embedder; the
  embedder is the only required dependency.
* :class:`GlobalHit` — the row returned by ``search()``.
  ``(source_id, source_url, license, text, score)`` — five fields,
  identical shape to what ``format_rag_excerpts`` already renders
  for per-chat chunks, so the prompt builder does not need a
  second format.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from app.rag_chunker import chunk_text
from app.rag_corpus import CorpusDoc
from app.rag_embedder import EMBEDDING_DIM, Embedder, MissingEmbedder

log = logging.getLogger(__name__)


# Lazy FAISS import. ``faiss-cpu`` is a hard dependency of PR-B and
# is always importable in the runtime image; the lazy import keeps
# the import cost off the path of code that does not need it (e.g.
# the test suite, the CLI's ``--status`` code path).
try:  # pragma: no cover - import guarded for environments without faiss
    import faiss  # type: ignore

    _FAISS_OK = True
except ImportError:  # pragma: no cover
    faiss = None  # type: ignore
    _FAISS_OK = False


@dataclass(frozen=True)
class GlobalHit:
    """One retrieved chunk from the global corpus.

    The shape is deliberately a superset of what the per-chat
    :class:`app.rag_store.RagHit` returns, so the view can
    concatenate the two lists before passing them to the prompt
    builder without a special case.
    """

    source_id: str
    source_url: str
    license: str
    text: str
    score: float


#: Minimum score for a hit to be considered relevant. Mirrors the
#: per-chat threshold in :class:`app.rag_store.RagStore`. The
#: lower bound (0.10 vs. the per-chat 0.30) accounts for very
#: short queries — a one- or two-word query against a
#: paragraph-sized chunk typically scores 0.15 even when the
#: topic matches, because the L2-normalised bag-of-tokens
#: vector only overlaps on a few buckets.
SCORE_THRESHOLD: float = 0.10

#: Default top-k for global retrieval. Smaller than the per-chat
#: default because the global index is meant to *complement* the
#: per-chat context, not replace it.
DEFAULT_K: int = 4


class GlobalIndex:
    """One in-memory FAISS index per ``source_id``.

    Construction is cheap; the actual FAISS indices are built on
    first ``search()`` for a given source. Use
    :meth:`add_source` to push a batch of :class:`CorpusDoc`
    rows into the underlying SQLite table; the index for that
    source is invalidated and rebuilt lazily on the next
    ``search()`` call.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        *,
        db_path: Optional[str] = None,
    ) -> None:
        # Default to ``MissingEmbedder`` (cheap, reports
        # ``is_available() == False``) so constructing a
        # ``GlobalIndex`` for read-only ``status()`` or ``clear()``
        # does not trigger a model download. The Streamlit view
        # injects the real embedder via ``st.cache_resource``.
        self._embedder: Embedder = embedder or MissingEmbedder()
        self._db_path: Optional[str] = db_path
        self._lock = threading.Lock()
        # _indices caches {source_id: (index, ids, texts)}. ids is
        # the list of row ids in the same order as the index rows;
        # we use it to look up the text/URL/license at query time.
        self._indices: dict[str, Tuple[object, List[int], List[Tuple[str, str, str, str]]]] = {}
        # Per-source ``_dirty`` flag, set by ``add_source`` /
        # ``clear`` to force a rebuild on the next ``search()``.
        self._dirty: set[str] = set()
        # Sources this ``GlobalIndex`` instance has ever seen.
        # ``status()`` always includes these so a freshly-cleared
        # source still shows up in the sidebar with ``chunks == 0``
        # rather than disappearing from the panel entirely.
        self._seen_sources: set[str] = set()
        self._init_dirty()

    # ---- Setup helpers ---------------------------------------------------

    def _init_dirty(self) -> None:
        """Mark every source already in the DB as dirty.

        We don't know which sources the DB contains until we open
        it, but :meth:`search` only builds an index for sources
        the caller actually queries, so we mark everything dirty
        and let the first ``search()`` re-check. The
        :meth:`_load_source` method drops the dirty bit once it
        successfully builds an index.
        """
        # All currently-known sources go dirty. New sources added
        # later are appended to the set by ``add_source``.
        self._dirty.add("*")

    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection. Uses the project's storage helper
        if no explicit ``db_path`` was provided, so tests can pass a
        temp file via the ``SECMENTOR_DB_PATH`` env var.

        When a custom ``db_path`` is supplied (tests, the CLI ingest
        script), we lazy-initialise the schema on first connect so
        the test setup does not have to call
        :func:`app.storage.init_db` explicitly. ``init_db`` is
        idempotent (every ``CREATE`` uses ``IF NOT EXISTS``) so this
        is a one-shot cost on the very first call. The default-path
        branch assumes the project bootstrap has already initialised
        the schema (Streamlit does this in ``_init_state``).
        """
        if self._db_path is not None:
            # Lazy import so this module is importable without
            # the rest of the project at import time. In
            # production, ``app.storage`` is always present.
            from app.storage import init_db

            init_db(self._db_path)
            return sqlite3.connect(self._db_path)
        from app.storage import _connect  # type: ignore

        return _connect()  # type: ignore[return-value]

    # ---- Capability queries ---------------------------------------------

    def is_available(self) -> bool:
        """Return True iff search() can return real hits.

        A ``GlobalIndex`` is available when:
          * the embedder is present and reports ``is_available()``
          * the FAISS import succeeded

        If either is false, ``search()`` returns ``[]``.
        """
        if self._embedder is None:
            return False
        if not _FAISS_OK:
            return False
        # Some embedders (e.g. ``MissingEmbedder``) explicitly
        # report themselves as unavailable.
        if hasattr(self._embedder, "is_available") and not self._embedder.is_available():
            return False
        return True

    # ---- Writes ----------------------------------------------------------

    def add_source(
        self,
        source_id: str,
        docs: Sequence[CorpusDoc],
        *,
        license: str = "",
    ) -> int:
        """Chunk + embed + persist ``docs`` for ``source_id``.

        Args:
            source_id: The :class:`CorpusSource.source_id`. Must
                already be registered (the CLI enforces this).
            docs: The parsed documents from the parser.
            license: Default license; the per-doc ``license`` wins
                if set on the doc itself. Empty string skips the
                check (used by tests with synthetic data).

        Returns:
            The number of chunks written to the DB.

        Notes:
            * If the embedder is unavailable, this is a no-op and
              returns 0. The CLI logs a warning in that case.
            * The FAISS index for ``source_id`` is marked dirty
              so the next ``search()`` rebuilds it.
        """
        if not self.is_available():
            log.warning(
                "GlobalIndex.add_source: embedder unavailable; "
                "skipping %s (%d docs)",
                source_id, len(docs),
            )
            return 0
        if not docs:
            return 0

        # Lazy import to keep this module importable without
        # the storage module at import time.
        from app.storage import add_global_chunks_batch

        all_rows: List[tuple[str, str, str, bytes]] = []
        all_meta: List[tuple[str, str, str, str]] = []  # (source_url, license, title, text)

        for doc in docs:
            chunks = chunk_text(doc.text)
            if not chunks:
                continue
            # Encode one doc at a time; chunk_text is small and
            # the embedder batches internally.
            vecs = self._embedder.encode(chunks)
            if vecs is None or len(vecs) != len(chunks):
                # Degraded: the embedder returned None / wrong
                # shape. Skip the doc rather than store garbage.
                log.warning(
                    "GlobalIndex.add_source: embedder returned %s for %s; skipping",
                    type(vecs).__name__ if vecs is not None else "None",
                    doc.source_url,
                )
                continue
            # Normalise once. Inner product on unit vectors == cosine.
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = (vecs / norms).astype("float32")
            doc_license = doc.license or license
            for i, (chunk, vec) in enumerate(zip(chunks, vecs)):
                all_rows.append((doc.source_url, doc_license, chunk, vec.tobytes()))
                all_meta.append((doc.source_url, doc_license, doc.title, chunk))

        if not all_rows:
            # Even with no rows we want ``status()`` to know this
            # source exists — it makes the sidebar panel render
            # the same way before and after an empty re-ingest.
            self._seen_sources.add(source_id)
            return 0

        # Drop any existing rows for this source so the new
        # ingestion fully replaces the old one. This is
        # semantically what an operator expects: a re-run of
        # ``ingest_security_corpus --source owasp`` means "OWASP
        # is now exactly the cheatsheets I just downloaded", not
        # "OWASP is whatever was there + the new bits".
        self.clear(source_id)

        ids = add_global_chunks_batch(source_id, all_rows, path=self._db_path)
        # Remember (id, source_url, license, title, text) so we
        # can build the index without re-fetching from SQLite.
        # We don't actually need the id in memory because the
        # index is rebuilt from SQLite on cache miss, but we
        # store the row id to keep the diagnostics honest.
        with self._lock:
            self._indices.pop(source_id, None)
            self._dirty.add(source_id)
        self._seen_sources.add(source_id)
        return len(ids)

    def clear(self, source_id: Optional[str] = None) -> int:
        """Delete rows for ``source_id`` (or all rows).

        Returns:
            The number of rows deleted. ``0`` if the source was
            already empty.
        """
        from app.storage import clear_global_corpus

        n = clear_global_corpus(source_id, path=self._db_path)
        with self._lock:
            if source_id is None:
                self._indices.clear()
                self._dirty.add("*")
                # Wipe the seen-sources set: after a full clear
                # the sidebar panel should show an empty corpus,
                # not phantom rows for sources we once indexed.
                self._seen_sources.clear()
            else:
                self._indices.pop(source_id, None)
                self._dirty.add(source_id)
                # Single-source clear keeps the source in the
                # panel so the operator can see it exists and is
                # currently empty (chunks == 0).
                self._seen_sources.add(source_id)
        return n

    # ---- Reads -----------------------------------------------------------

    def status(self) -> dict:
        """Return a per-source summary of what's indexed.

        Returns:
            ``{source_id: {"chunks": N, "distinct_docs": M}}`` plus
            an ``"_all"`` summary. The shape mirrors
            :func:`app.storage.list_corpus_sources` so the view
            can render the sidebar panel from one source of truth.

            Sources that this ``GlobalIndex`` has indexed at any
            point in its lifetime (``self._seen_sources``) appear
            even if their chunk count is currently zero — that
            way ``clear("owasp")`` leaves the source in the panel
            rather than vanishing it.
        """
        from app.storage import list_corpus_sources, count_global_chunks

        rows = list_corpus_sources(path=self._db_path)
        total = count_global_chunks(path=self._db_path)
        out: dict = {"_all": {"chunks": total, "distinct_docs": total}}
        seen: set[str] = set()
        for r in rows:
            out[r["source_id"]] = {
                "chunks": r["chunk_count"],
                "distinct_docs": r["distinct_docs"],
            }
            seen.add(r["source_id"])
        # Backfill sources the index has touched before but
        # which currently have no rows. Without this, a
        # ``clear("owasp")`` would delete the ``owasp`` entry
        # from the panel and the operator would have no way to
        # see "yes, OWASP exists, it's just empty".
        for sid in self._seen_sources - seen:
            out[sid] = {"chunks": 0, "distinct_docs": 0}
        return out

    def search(
        self,
        query_embedding: Union[str, np.ndarray],
        k: int = DEFAULT_K,
        *,
        source_filter: Optional[Sequence[str]] = None,
    ) -> List[GlobalHit]:
        """Return the top-k global hits for ``query_embedding``.

        Args:
            query_embedding: Either a raw query string (which we
                encode via the injected embedder) or a 1-D
                ``np.ndarray`` of shape ``(EMBEDDING_DIM,)``.
                Strings are the convenient form for tests and the
                sidebar "knowledge base search" box; arrays are
                what the prompt builder already has on hand. We
                normalise to unit length inside this method so
                callers do not have to remember to.
            k: Maximum number of hits to return. Clamped to >= 1.
            source_filter: Optional list of ``source_id`` values
                to query. ``None`` queries every indexed source.
                Empty list returns ``[]`` immediately (useful for
                tests).

        Returns:
            Up to ``k`` :class:`GlobalHit` instances, sorted by
            score descending, each with ``score >= SCORE_THRESHOLD``.

        Notes:
            * If :meth:`is_available` is False, returns ``[]``.
            * If the FAISS index for a source fails to build
              (e.g. zero rows), that source contributes nothing.
            * Merging across sources is by raw score, so the
              top-2k hits from a corpus of two sources with
              k=4 yields at most 8 rows, then trimmed to 2k
              (= 8 here, but the view can pass any k).
        """
        if not self.is_available():
            return []
        if k <= 0:
            return []
        if source_filter is not None and len(source_filter) == 0:
            return []

        # Accept either a raw string (encode via the embedder)
        # or a pre-computed 1-D array. This is the same contract
        # ``app.rag_store.RagStore.search`` exposes — the only
        # difference is the global index also handles the string
        # path so the sidebar "search the KB" box can call
        # ``global_index.search(query_text)`` directly.
        if isinstance(query_embedding, str):
            try:
                encoded = self._embedder.encode([query_embedding])
            except Exception as exc:  # pragma: no cover - defensive
                log.warning(
                    "GlobalIndex.search: embedder.encode failed: %s", exc,
                )
                return []
            if encoded is None or len(encoded) == 0:
                return []
            q = np.asarray(encoded[0], dtype="float32").reshape(-1)
        else:
            q = np.asarray(query_embedding, dtype="float32").reshape(-1)
        if q.shape[0] != EMBEDDING_DIM:
            log.warning(
                "GlobalIndex.search: query dim %d != %d",
                q.shape[0], EMBEDDING_DIM,
            )
            return []
        norm = float(np.linalg.norm(q))
        if norm == 0.0:
            return []
        q = q / norm
        q = q.reshape(1, -1)

        # Pick the sources to query.
        all_sources = self._known_source_ids()
        if source_filter is None:
            sources = list(all_sources)
        else:
            sources = [s for s in source_filter if s in all_sources]
            if not sources:
                return []

        hits: List[GlobalHit] = []
        for source_id in sources:
            index, rows = self._load_source(source_id)
            if index is None or not rows:
                continue
            try:
                scores, idxs = index.search(q, min(k, len(rows)))
            except Exception as exc:  # pragma: no cover - defensive
                log.warning(
                    "GlobalIndex.search: %s search failed: %s",
                    source_id, exc,
                )
                continue
            for score, idx in zip(scores[0], idxs[0]):
                if idx < 0 or idx >= len(rows):
                    continue
                if score < SCORE_THRESHOLD:
                    continue
                row = rows[idx]
                hits.append(
                    GlobalHit(
                        source_id=source_id,
                        source_url=row[0],
                        license=row[1],
                        text=row[2],
                        score=float(score),
                    )
                )

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[: max(k, 1)]

    # ---- Internals -------------------------------------------------------

    def _known_source_ids(self) -> List[str]:
        """Return the list of source_ids present in the DB.

        We do not consult the :class:`CorpusRegistry` here — the
        view's wire-up does that. The index's only job is "do we
        have rows for this source_id?".
        """
        from app.storage import list_corpus_sources

        try:
            rows = list_corpus_sources(path=self._db_path)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("GlobalIndex._known_source_ids: %s", exc)
            return []
        return [r["source_id"] for r in rows]

    def _load_source(
        self, source_id: str
    ) -> Tuple[Optional[object], List[Tuple[str, str, str, str]]]:
        """Return the (index, rows) for ``source_id``, building if needed.

        ``rows`` is a list of ``(source_url, license, text, title)``
        in the same order as the index vectors. The title is
        carried along so the prompt can render a citation.
        """
        with self._lock:
            if source_id not in self._dirty and source_id in self._indices:
                idx, rows = self._indices[source_id]
                return idx, rows
        # Build outside the lock so concurrent first-time
        # searches on different sources can run in parallel.
        from app.storage import list_global_chunks

        try:
            db_rows = list_global_chunks(source_id, path=self._db_path)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("GlobalIndex._load_source: %s list failed: %s", source_id, exc)
            return None, []
        if not db_rows:
            with self._lock:
                self._indices.pop(source_id, None)
                self._dirty.discard(source_id)
            return None, []
        vectors = np.frombuffer(
            b"".join(r["embedding"] for r in db_rows), dtype="float32"
        ).reshape(len(db_rows), EMBEDDING_DIM)
        # Defensive re-normalise; rows stored pre-normalised but
        # if a future migration writes un-normalised vectors we
        # want the search to still work.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        index.add(vectors)
        rows = [
            (str(r["source_url"]), str(r["license"]), str(r["text"]), str(r["source_id"]))
            for r in db_rows
        ]
        with self._lock:
            self._indices[source_id] = (index, rows)
            self._dirty.discard(source_id)
        return index, rows


__all__ = [
    "DEFAULT_K",
    "GlobalHit",
    "GlobalIndex",
    "SCORE_THRESHOLD",
]

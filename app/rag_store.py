"""Per-chat FAISS retrieval over chunks persisted in SQLite.

The store is the *retrieval* half of the RAG pipeline. The persistence
half lives in :mod:`app.storage`; the chunking half in
:mod:`app.rag_chunker`; the embedding half in :mod:`app.rag_embedder`.
This module glues those three together and is intentionally thin.

Design decisions (all from ``docs/phase_12_rag_and_history.md`` §4 PR-B):

* **Per-chat index, built lazily.** On the first ``search(chat_id)``
  call, all chunks for that chat are pulled from SQLite, L2-normalized,
  and packed into an ``IndexFlatIP``. The index is cached in-process
  keyed by ``chat_id``; subsequent searches reuse it. ``add()`` does
  *not* touch the index — it only persists rows and bumps a version
  counter, so search stays read-only on the hot path.
* **Invalidation via ``_index_version`` counter.** Every ``add()`` bumps
  a per-chat counter (via a small SQL hop from ``artifact_id`` to its
  owning ``chat_id``). When ``search()`` sees the counter moved past
  the cached version, the cached index is rebuilt. This is O(n) per
  chat per session; for n > 1000 use ``IndexIDMap2.add_with_ids``.
* **Cosine = inner product.** Vectors are L2-normalized at ``add()``
  time; ``IndexFlatIP`` is searched with the normalized query. This is
  the sentence-transformers recommended recipe.
* **Score threshold.** ``search()`` returns hits with score
  ``> self.score_threshold`` (default ``DEFAULT_SCORE_THRESHOLD`` =
  0.30, pinned in :mod:`app.rag_embedder`). The threshold is
  empirical; tune it on a fixture corpus when retraining.
* **Lazy FAISS import.** ``import faiss`` happens at module load behind
  a try/except, so the module imports cleanly on systems without
  FAISS. ``search()`` degrades to ``[]`` (the embedder-degraded branch
  already does this too). ``add()`` does not need FAISS at all — it
  persists via :mod:`app.storage`.
* **Embedder-degraded mode.** When ``embedder.is_available()`` is
  ``False`` (model not installed, ``PUKU_RAG_OFFLINE=1``, or a
  :class:`MissingEmbedder` passed in), ``add()`` returns ``[]`` and
  ``search()`` returns ``[]``. No exception escapes.

The class is stateless across instances (re-instantiating is safe), but
the in-memory FAISS cache is per-instance. A long-running app keeps a
single ``RagStore`` alive for the lifetime of the session.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.rag_embedder import (
    DEFAULT_SCORE_THRESHOLD,
    EMBEDDING_DIM,
    EmbedderProtocol,
    l2_normalize,
)

# FAISS is imported lazily. Tests + apps that never call ``search()``
# (i.e. only persist chunks via ``add()``) do not need FAISS installed.
# A failed import is recorded in ``_FAISS_IMPORT_ERROR`` for diagnostics
# — surfaced in the warning emitted by ``search()`` when called.
_FAISS_IMPORT_ERROR: Optional[BaseException] = None
try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE: bool = True
except Exception as _exc:  # pragma: no cover - exercised only without faiss
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False
    _FAISS_IMPORT_ERROR = _exc


log = logging.getLogger(__name__)


# --- Constants ----------------------------------------------------------------

#: Default number of hits returned by :meth:`RagStore.search`.
DEFAULT_K: int = 4


# --- Cached-index record ------------------------------------------------------


@dataclass
class _CachedIndex:
    """The in-memory cache for one chat's FAISS index.

    The ``version`` field is the value of ``_index_version[chat_id]`` at
    the time the index was built. If a subsequent ``add()`` bumps the
    version, the cache is treated as stale and rebuilt on the next
    ``search()``.

    ``chunk_ids`` is parallel to FAISS index positions:
    ``chunk_ids[i]`` is the database ``chunks.id`` stored at FAISS
    position ``i``. ``text_by_id`` is the inverted map for O(1) text
    lookup during search.
    """

    index: object  # faiss.IndexFlatIP; typed as object to avoid the import
    chunk_ids: List[int]
    text_by_id: Dict[int, str]
    version: int


# --- Store --------------------------------------------------------------------


class RagStore:
    """Per-chat retrieval over chunks persisted in SQLite.

    Args:
        embedder: An object satisfying :class:`EmbedderProtocol`. The
            store consults ``embedder.is_available()`` on every call;
            ``add()`` is a no-op and ``search()`` returns ``[]`` when
            the embedder is degraded. Tests pass a deterministic fake.
        db_path: Override for the SQLite location. ``None`` uses
            :func:`app.storage.db_path`.
        score_threshold: Hits below this cosine score are filtered out.
            Defaults to :data:`app.rag_embedder.DEFAULT_SCORE_THRESHOLD`
            (0.30). Pass a different value to make the retriever more
            or less aggressive.
    """

    def __init__(
        self,
        embedder: EmbedderProtocol,
        *,
        db_path: Optional[Path] = None,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    ) -> None:
        self._embedder = embedder
        self._db_path = db_path
        self._score_threshold = float(score_threshold)
        self._cache: Dict[str, _CachedIndex] = {}
        self._index_version: Dict[str, int] = {}
        # Serializes lazy index builds per chat. Cheap, only matters
        # if a single app instance has concurrent calls (Streamlit
        # is single-threaded per session, so the guard is defensive).
        self._lock = threading.Lock()

    # ---- Read-only helpers ------------------------------------------------

    @property
    def embedder(self) -> EmbedderProtocol:
        return self._embedder

    @property
    def score_threshold(self) -> float:
        return self._score_threshold

    @property
    def faiss_available(self) -> bool:
        """True iff the FAISS extension is importable in this process."""
        return _FAISS_AVAILABLE

    def is_available(self) -> bool:
        """True iff retrieval is possible right now.

        Both the embedder and FAISS must be usable. ``add()`` does not
        need FAISS, so callers that only persist chunks can ignore
        this and just try ``add()``.
        """
        return bool(self._embedder.is_available()) and _FAISS_AVAILABLE

    # ---- Write -----------------------------------------------------------

    def add(
        self,
        artifact_id: str,
        chunks: Sequence[str],
        embeddings: np.ndarray,
    ) -> List[int]:
        """Persist ``chunks`` for ``artifact_id`` and return their ``id``s.

        The embeddings are L2-normalized before being written to SQLite
        so the in-memory FAISS index (built later on ``search()``) sees
        unit-length vectors. The normalization is idempotent — if the
        caller already normalized, this is a no-op.

        Degraded mode: if ``embedder.is_available()`` is False, this
        returns ``[]`` and does not write anything. The caller can
        detect this by checking the return value.

        Args:
            artifact_id: The owning artifact. Must already exist in
                ``artifacts`` (call :func:`app.storage.add_artifact`
                first).
            chunks: The chunk texts, in order. Must be the same length
                as ``embeddings``. May be empty.
            embeddings: A ``(n, EMBEDDING_DIM)`` ``float32`` array
                parallel to ``chunks``. May have n=0.

        Returns:
            The new chunk ``id``s in the same order as ``chunks``.
            Empty list in degraded mode or when ``chunks`` is empty.

        Raises:
            ValueError: If ``len(chunks) != len(embeddings)`` or the
                embedding width is not :data:`EMBEDDING_DIM`. The
                shape check is strict because a silent mismatch would
                produce a corrupt FAISS index on the next search.
        """
        # Lazy import to avoid a hard dep at module load.
        from app import storage

        if not self._embedder.is_available():
            log.debug(
                "RagStore.add: skipped (embedder unavailable: %s)",
                getattr(self._embedder, "last_error", None),
            )
            return []

        n = len(chunks)
        if embeddings.shape != (n, EMBEDDING_DIM):
            raise ValueError(
                f"embeddings shape {embeddings.shape} does not match "
                f"chunks ({n}); expected ({n}, {EMBEDDING_DIM})"
            )
        if n == 0:
            return []
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32, copy=False)

        # Normalize. Cheap; idempotent on already-unit vectors.
        normalized = l2_normalize(embeddings)

        rows = [
            (text, np.ascontiguousarray(emb, dtype=np.float32).tobytes())
            for text, emb in zip(chunks, normalized)
        ]
        ids = storage.add_chunks_returning_ids(
            artifact_id, rows, path=self._db_path
        )

        # Bump the version for the owning chat so the cache is rebuilt
        # on the next search. If the artifact was deleted between
        # ``add_chunks_returning_ids`` and now (unlikely but possible
        # in a multi-writer scenario), get_chat_id_for_artifact
        # returns None and we silently skip — no version to bump
        # for a chat that no longer owns this artifact.
        chat_id = storage.get_chat_id_for_artifact(
            artifact_id, path=self._db_path
        )
        if chat_id is not None:
            with self._lock:
                self._index_version[chat_id] = (
                    self._index_version.get(chat_id, 0) + 1
                )

        return ids

    # ---- Read ------------------------------------------------------------

    def search(
        self,
        chat_id: str,
        query_emb: np.ndarray,
        *,
        k: int = DEFAULT_K,
    ) -> List[Tuple[int, str, float]]:
        """Return the top-``k`` chunks for ``chat_id`` matching ``query_emb``.

        Returns ``list[(chunk_id, text, score)]`` in descending score
        order, filtered to ``score > self.score_threshold``. Degraded
        mode (embedder or FAISS unavailable, empty chat) returns ``[]``
        — never raises.

        Args:
            chat_id: The chat to search. Chunks are scoped to this chat
                only (no cross-chat leakage — this is the property
                pinned by :class:`CrossChatIsolationTests`).
            query_emb: A ``(EMBEDDING_DIM,)`` or ``(1, EMBEDDING_DIM)``
                ``float32`` array. Will be L2-normalized.
            k: Max number of hits to return. Clamped to the number of
                indexed chunks.

        Returns:
            A new ``list`` of ``(chunk_id, text, score)`` triples.
        """
        if not self._embedder.is_available():
            return []
        if not _FAISS_AVAILABLE:
            log.warning(
                "RagStore.search: FAISS is not installed; retrieval "
                "is disabled. Install faiss-cpu to enable. (%s)",
                _FAISS_IMPORT_ERROR,
            )
            return []

        # Normalize the query (idempotent on already-unit vectors).
        q = np.asarray(query_emb, dtype=np.float32).reshape(1, EMBEDDING_DIM)
        q = l2_normalize(q)

        cached = self._ensure_index(chat_id)
        if cached is None or cached.index.ntotal == 0:
            return []

        k_eff = min(int(k), int(cached.index.ntotal))
        scores, positions = cached.index.search(q, k_eff)
        # FAISS returns shape (1, k_eff) for a single-query search.
        scores_row = scores[0]
        positions_row = positions[0]

        out: List[Tuple[int, str, float]] = []
        for pos, score in zip(positions_row, scores_row):
            # FAISS returns -1 for "no result" when k > ntotal, even
            # though we clamp k_eff above; defensive against future
            # FAISS API changes.
            if int(pos) < 0:
                continue
            if int(pos) >= len(cached.chunk_ids):
                continue
            if float(score) <= self._score_threshold:
                continue
            chunk_id = cached.chunk_ids[int(pos)]
            text = cached.text_by_id.get(chunk_id)
            if text is None:
                # Defensive: cached.chunk_ids and text_by_id are
                # built together, so this should be unreachable.
                continue
            out.append((int(chunk_id), text, float(score)))
        return out

    # ---- Internals -------------------------------------------------------

    def _ensure_index(self, chat_id: str) -> Optional[_CachedIndex]:
        """Return a fresh-or-cached FAISS index for ``chat_id``."""
        from app import storage

        with self._lock:
            cached = self._cache.get(chat_id)
            current_version = self._index_version.get(chat_id, 0)
            if cached is not None and cached.version == current_version:
                return cached

            rows = storage.list_chunks_for_chat(chat_id, path=self._db_path)
            if not rows:
                # Cache an empty index so we do not re-query on every
                # search. The version is stamped so the next ``add()``
                # invalidates correctly.
                empty_index = faiss.IndexFlatIP(EMBEDDING_DIM)
                empty = _CachedIndex(
                    index=empty_index,
                    chunk_ids=[],
                    text_by_id={},
                    version=current_version,
                )
                self._cache[chat_id] = empty
                return empty

            vectors = np.empty((len(rows), EMBEDDING_DIM), dtype=np.float32)
            chunk_ids: List[int] = []
            text_by_id: Dict[int, str] = {}
            for i, row in enumerate(rows):
                vectors[i] = np.frombuffer(
                    row["embedding"], dtype=np.float32
                )
                cid = int(row["id"])
                chunk_ids.append(cid)
                text_by_id[cid] = str(row["text"])

            # Re-normalize on read, in case a row was written by an
            # older code path that did not normalize. l2_normalize
            # preserves all-zero rows as zero; non-zero rows become
            # unit-length.
            vectors = l2_normalize(vectors)

            index = faiss.IndexFlatIP(EMBEDDING_DIM)
            index.add(vectors)
            cached = _CachedIndex(
                index=index,
                chunk_ids=chunk_ids,
                text_by_id=text_by_id,
                version=current_version,
            )
            self._cache[chat_id] = cached
            return cached

    # ---- Bookkeeping -----------------------------------------------------

    def invalidate(self, chat_id: str) -> None:
        """Drop the cached index for ``chat_id``.

        Test-only utility; production callers should let the version
        counter do the work. Useful when a test deletes chunks directly
        via the storage layer and wants the next ``search()`` to pick
        that up.
        """
        with self._lock:
            self._cache.pop(chat_id, None)
            self._index_version[chat_id] = (
                self._index_version.get(chat_id, 0) + 1
            )


__all__ = [
    "DEFAULT_K",
    "RagStore",
    "_FAISS_AVAILABLE",
]

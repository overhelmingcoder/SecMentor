"""Lazy-loaded sentence-transformers wrapper for Phase 12 RAG.

This module owns exactly one responsibility: turn a list of strings
into a ``(n, 384)`` float32 numpy array of L2-normalized embeddings.
It does not chunk, store, retrieve, or render — those are the
chunker, the store, and the view layer's jobs respectively.

Design notes
------------

* **Lazy load.** The model is ~80 MB. The first ``.encode()`` call
  downloads it (one-time, cached by Hugging Face) and is slow
  (5–10 s on a warm cache, longer on a cold one). The lazy load
  is what makes ``streamlit run web/streamlit_app.py`` *start*
  without a 5-second hitch: the model is not loaded until the
  first RAG request. The view layer wraps the instance in
  ``st.cache_resource`` so it survives Streamlit reruns.

* **L2-normalized output.** The embedder's ``encode()`` is called
  with ``normalize_embeddings=True`` so the returned vectors are
  unit length. This is the standard recipe for using cosine
  similarity via FAISS's ``IndexFlatIP`` (inner product on unit
  vectors == cosine similarity). It also means a query can be
  embedded the same way and ``index.search`` returns cosine scores
  in ``[-1.0, 1.0]``.

* **Single model, pinned.** The model name
  (``sentence-transformers/all-MiniLM-L6-v2``) and the output
  dimension (384) are both pinned. Changing the model is a
  breaking change for the FAISS index (it has to be rebuilt from
  scratch) and for the doc's stated "384-dim" contract. If you
  swap the model, bump a version constant and force a full
  re-index — see PR-D's wire-up.

* **Empty input is a shape-(0, 384) array.** The embedder does
  not raise on ``encode([])``; the doc pinned this as a
  footgun-prevention rule. The store relies on it: a chunker that
  returns ``[]`` for a scanned PDF should not crash the embedder
  or the store.

* **Degraded mode.** The :class:`MissingEmbedder` stub is the
  graceful-degradation path: when the model cannot be loaded
  (offline install, broken torch wheel, etc.) the app still runs,
  the sidebar still works, and RAG searches return ``[]`` with
  a one-line warning. The :class:`Embedder` factory returns this
  stub instead of raising so the view layer never has to wrap
  the encode call in a try/except.

This module is *not* imported by the chunker. The chunker is
stdlib-only and unit-testable without ``sentence-transformers``
installed — see the module docstring of :mod:`app.rag_chunker`.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional, Sequence

import numpy as np


# --- Constants (pinned) -----------------------------------------------------

#: The model name. Pinned by the Phase 12 doc (§3, §4) and by the
#: FAISS index dimension below. Changing this is a breaking change.
MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

#: Output embedding dimension. ``all-MiniLM-L6-v2`` produces 384-dim
#: vectors. The FAISS index in :mod:`app.rag_store` is built with
#: this exact dimension; a mismatch would raise at search time.
EMBEDDING_DIM: int = 384

#: The minimum cosine score for a chunk to be considered a "hit".
#: Below this threshold, the chunk is treated as noise and the
#: store returns ``[]`` (or fewer than ``k`` hits) instead of
#: injecting low-relevance paragraphs into the prompt. The
#: threshold is empirical — see PR-B's design notes in the doc.
DEFAULT_SCORE_THRESHOLD: float = 0.30

#: Environment variable to force the embedder into degraded mode
#: without actually trying to load the model. Used by the tests
#: to pin the "model unavailable" branch deterministically.
RAG_OFFLINE_ENV: str = "PUKU_RAG_OFFLINE"


# --- Public protocol --------------------------------------------------------


class EmbedderProtocol:
    """The narrow interface the store and view layer depend on.

    Defined as a class (not a ``typing.Protocol``) because the
    Streamlit-degraded stub :class:`MissingEmbedder` does not
    inherit from anything else and the test suite does an
    ``isinstance`` check on it. Keeping it as a plain class
    means the stub can be a subclass *and* the type hint is
    stable across Python versions.
    """

    @property
    def dim(self) -> int:  # pragma: no cover - trivial
        return EMBEDDING_DIM

    @property
    def model_name(self) -> str:  # pragma: no cover - trivial
        return MODEL_NAME

    def is_available(self) -> bool:
        """Return True iff this embedder can actually produce vectors."""
        raise NotImplementedError

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a batch of strings into a ``(n, dim)`` float32 array.

        Vectors are L2-normalized; the caller can use inner
        product as a proxy for cosine similarity.
        """
        raise NotImplementedError


# --- Implementation ---------------------------------------------------------


class Embedder(EmbedderProtocol):
    """Lazy-loaded wrapper around the pinned sentence-transformers model.

    The first call to :meth:`encode` downloads the model and is
    slow. Subsequent calls are fast. If the download or import
    fails for any reason, :meth:`is_available` returns ``False``
    and :meth:`encode` returns an empty array — the view layer
    surfaces a one-line ``st.warning`` and falls back to the
    no-RAG prompt.
    """

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        # The model is not loaded here. We only remember the
        # name; the heavy import + download happens on first
        # ``encode()`` call. This is what keeps ``streamlit run``
        # snappy on cold start.
        self._model_name = model_name
        self._model: Optional[object] = None
        self._load_error: Optional[str] = None

    # ---- Public API ------------------------------------------------------

    @property
    def dim(self) -> int:
        return EMBEDDING_DIM

    @property
    def model_name(self) -> str:
        return self._model_name

    def is_available(self) -> bool:
        """Return True iff the model loaded successfully (or will on next call).

        Note: this triggers a load attempt if one has not
        happened yet, so the *first* call to ``is_available()``
        is slow. Use :data:`RAG_OFFLINE_ENV` to short-circuit.
        """
        if os.environ.get(RAG_OFFLINE_ENV) == "1":
            self._load_error = "degraded via PUKU_RAG_OFFLINE=1"
            return False
        if self._model is not None:
            return True
        self._try_load()
        return self._model is not None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode ``texts`` into a ``(len(texts), dim)`` float32 array.

        Empty input returns a shape ``(0, dim)`` array (no model
        call). If the model is unavailable, returns the same
        shape and stores the error in :attr:`last_error` for
        the view layer to display.
        """
        # Pin: empty input -> shape (0, dim) without touching
        # the model. This is the chunker-returns-[] footgun
        # guard.
        texts = list(texts)
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        if os.environ.get(RAG_OFFLINE_ENV) == "1":
            self._load_error = "degraded via PUKU_RAG_OFFLINE=1"
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        if self._model is None:
            self._try_load()
        if self._model is None:  # still None after the attempt
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        # ``normalize_embeddings=True`` is the cosine-via-inner-product
        # recipe from the sentence-transformers README. The
        # returned vectors are unit length.
        vectors = self._model.encode(  # type: ignore[attr-defined]
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        # Pin the dtype. sentence-transformers already returns
        # float32 on CPU, but the explicit cast documents the
        # contract and protects against a future model swap
        # that changes the dtype.
        return np.asarray(vectors, dtype=np.float32)

    @property
    def last_error(self) -> Optional[str]:
        """The most recent load error, for the view layer to display.

        ``None`` if the model loaded successfully (or has not
        been attempted yet).
        """
        return self._load_error

    # ---- Internals -------------------------------------------------------

    def _try_load(self) -> None:
        """Attempt the lazy import + load. On any failure, store the error."""
        try:
            # Imported lazily so ``import app.rag_embedder`` does
            # not require sentence-transformers to be installed.
            # The chunker test suite depends on this.
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - import guard
            self._load_error = (
                f"sentence-transformers import failed: {exc!r}"
            )
            self._model = None
            return

        try:
            self._model = SentenceTransformer(self._model_name)
        except Exception as exc:  # pragma: no cover - network guard
            self._load_error = (
                f"SentenceTransformer({self._model_name!r}) failed: {exc!r}"
            )
            self._model = None


# --- Degraded stub ----------------------------------------------------------


class MissingEmbedder(EmbedderProtocol):
    """The graceful-degradation embedder.

    Returned by :func:`get_embedder` when the model cannot be
    loaded. The store and view layer can call :meth:`encode` and
    :meth:`is_available` without try/except: every call is a
    no-op that returns the empty array / ``False``.

    The stub is *not* a singleton — the factory returns a fresh
    instance per call. This keeps the test suite's
    ``MissingEmbedder()`` construction simple (no shared
    mutable state between tests).
    """

    def __init__(self, reason: str = "embedding model unavailable") -> None:
        self._reason = reason

    @property
    def dim(self) -> int:
        return EMBEDDING_DIM

    @property
    def model_name(self) -> str:
        return f"<missing: {self._reason}>"

    def is_available(self) -> bool:
        return False

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        # Always returns the empty array. Shape (0, dim) so the
        # store's ``arr.shape[0]`` check works.
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    @property
    def last_error(self) -> str:  # type: ignore[override]
        # Note: we *deliberately* do not match ``Optional[str]``
        # here — a missing embedder always has a reason. The
        # view layer uses ``is_available()`` to gate the
        # warning, not a truthy check on ``last_error``.
        return self._reason


# --- Factory ----------------------------------------------------------------


def get_embedder() -> EmbedderProtocol:
    """Return the embedder for the current process.

    The default factory returns an :class:`Embedder` instance
    (which loads the model lazily on first encode). If
    ``PUKU_RAG_OFFLINE=1`` is set, or if the model's import
    fails on a *eager* probe, the factory returns a
    :class:`MissingEmbedder` instead so the rest of the app
    can run in degraded mode.

    The eager probe in this function is what makes the
    "first call to ``is_available()``" free in the common
    case: by the time the view layer checks, we already know
    whether the import works.
    """
    if os.environ.get(RAG_OFFLINE_ENV) == "1":
        return MissingEmbedder(reason="PUKU_RAG_OFFLINE=1")

    # Eagerly probe the import. The model itself is still
    # loaded lazily on first encode — we just want to know
    # whether ``sentence_transformers`` is installed.
    try:
        import sentence_transformers  # noqa: F401
    except Exception as exc:
        return MissingEmbedder(
            reason=f"sentence-transformers not importable: {exc!r}"
        )

    return Embedder()


# --- Sanity helpers (for tests) --------------------------------------------


def l2_normalize(arr: np.ndarray) -> np.ndarray:
    """Return a copy of ``arr`` with each row L2-normalized.

    Used by the store at add time, and by the tests to verify
    the embedder's output is already normalized. The function
    is tolerant of zero rows (returns the zero row unchanged,
    rather than emitting NaN).
    """
    if arr.size == 0:
        return arr.astype(np.float32, copy=True)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    # Avoid division by zero on the zero-vector row.
    safe = np.where(norms == 0.0, 1.0, norms)
    out = arr / safe
    # Restore the zero rows exactly.
    out = np.where(norms == 0.0, 0.0, out)
    return out.astype(np.float32, copy=False)


def cosine_score(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-d float32 vectors.

    Defined here (not in the store) so the tests can pin the
    math without depending on FAISS. Both inputs are assumed
    to be L2-normalized; if they are not, the result is the
    raw dot product over the product of norms (which is the
    real definition of cosine).
    """
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def is_unit_vector(v: np.ndarray, atol: float = 1e-5) -> bool:
    """Return True iff ``v`` has L2 norm 1.0 within ``atol``."""
    v = np.asarray(v, dtype=np.float32).ravel()
    if v.size == 0:
        return True
    return math.isclose(float(np.linalg.norm(v)), 1.0, abs_tol=atol)

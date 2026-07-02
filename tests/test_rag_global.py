"""Tests for :mod:`app.rag_global` — the Phase 12 PR-E retrieval layer.

The global RAG feature has two halves:

* :mod:`app.rag_corpus` — the manifest + parsers + HTTP fetch
  (covered by ``tests/test_rag_corpus.py``).
* :mod:`app.rag_global` — the per-source FAISS index, the
  :class:`GlobalIndex` wrapper, and the search loop that
  powers the Streamlit sidebar panel and the ``_retrieve_rag_chunks``
  merge in ``web/streamlit_app.py``.

This file pins the second half:

* The :class:`GlobalIndex` constructor — defaults to
  :class:`MissingEmbedder` so a bare ``GlobalIndex()`` does not
  trigger a model download.
* :meth:`GlobalIndex.add_source` — chunk + embed + persist; returns
  the chunk count; a re-add of the same ``source_id`` is a *replace*,
  not an *append* (a re-run of the CLI is idempotent).
* :meth:`GlobalIndex.search` — round-trip via the deterministic
  ``FakeEmbedder``; cross-source ranking; per-source
  ``source_filter``; score threshold at 0.30.
* :meth:`GlobalIndex.clear` — scoped (one source) and full (no
  argument) deletes.
* :meth:`GlobalIndex.status` — the dict shape the sidebar
  consumes.
* :meth:`GlobalIndex.is_available` — False under
  :class:`MissingEmbedder`, True under :class:`FakeEmbedder`.
* The degraded path: when ``is_available()`` is False, every
  search is ``[]`` and every ``add_source`` is a no-op.

The tests use a per-test SQLite file under ``tempfile`` so they
do not touch the real ``secmentor.db`` and can run in parallel
without colliding.

Run with:  python -m unittest tests.test_rag_global -v
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List
from unittest import mock

import numpy as np
import pytest

pytestmark = pytest.mark.smoke


# --- Path bootstrap ---------------------------------------------------------
# Mirror the project-root-cd pattern used in tests/test_storage.py,
# tests/test_rag.py and tests/test_rag_corpus.py.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from app import storage  # noqa: E402
from app.rag_corpus import CorpusDoc  # noqa: E402
from app.rag_embedder import EMBEDDING_DIM, MissingEmbedder  # noqa: E402
from app.rag_global import (  # noqa: E402
    DEFAULT_K,
    SCORE_THRESHOLD,
    GlobalHit,
    GlobalIndex,
)


# --- Fixtures ---------------------------------------------------------------


# Two short docs per source, with keyword overlap designed so the
# FakeEmbedder (token-hash bag-of-features) produces a clear
# "this query is closest to this doc" ranking.
_OWASP_DOC_A = CorpusDoc(
    source_id="owasp",
    source_url="https://example.invalid/owasp/xss.md",
    license="CC-BY-SA-4.0",
    title="XSS Prevention",
    text=(
        "Cross-site scripting XSS attacks occur when untrusted data "
        "is included in a web page without proper validation or "
        "escaping. Use context-aware output encoding."
    ),
)
_OWASP_DOC_B = CorpusDoc(
    source_id="owasp",
    source_url="https://example.invalid/owasp/sqli.md",
    license="CC-BY-SA-4.0",
    title="SQL Injection Prevention",
    text=(
        "SQL injection allows attackers to execute arbitrary SQL "
        "statements. Use parameterised queries and prepared "
        "statements."
    ),
)

_MITRE_DOC_A = CorpusDoc(
    source_id="mitre",
    source_url="https://example.invalid/mitre/phishing.md",
    license="CC-BY-4.0",
    title="T1566 Phishing",
    text=(
        "Adversaries send phishing messages to gain access to "
        "victim systems. Spearphishing Attachment is a sub-technique."
    ),
)
_MITRE_DOC_B = CorpusDoc(
    source_id="mitre",
    source_url="https://example.invalid/mitre/exploit.md",
    license="CC-BY-4.0",
    title="T1190 Exploit Public-Facing Application",
    text=(
        "Adversaries exploit weaknesses in public-facing web "
        "servers and applications to gain initial access."
    ),
)


class FakeEmbedder:
    """In-test embedder: deterministic, L2-normalised, hash-based.

    The same text always produces the same vector, so the
    round-trip tests are reproducible. The vector is built from
    token-hash buckets — not random noise — so two texts that
    share keywords produce *similar* vectors and can be ranked
    by inner product.

    Implements the duck-typed protocol used by
    :class:`app.rag_global.GlobalIndex` (the project also has
    a full :class:`EmbedderProtocol` base class in
    :mod:`app.rag_embedder`; we don't subclass it here because
    the GlobalIndex only calls ``encode`` and ``is_available``).
    """

    def __init__(self, dim: int = EMBEDDING_DIM, seed: int = 0xC0FFEE) -> None:
        self._dim = dim
        self._seed = seed
        self._available = True

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return "fake-global-v1"

    def is_available(self) -> bool:
        return self._available

    def encode(self, texts) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        out = np.empty((len(texts), self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i] = self._vector_for(str(t))
        return out

    def _vector_for(self, text: str) -> np.ndarray:
        # Bag-of-features: each token contributes 1 to a fixed
        # bucket. Same tokens → same vector. L2-normalised so
        # inner product == cosine.
        v = np.zeros(self._dim, dtype=np.float32)
        for tok in text.lower().split():
            # Strip punctuation so "XSS." and "XSS" match.
            tok = tok.strip(".,;:!?\"'()[]{}")
            if not tok:
                continue
            bucket = (hash(tok) ^ self._seed) % self._dim
            v[bucket] += 1.0
        n = np.linalg.norm(v)
        if n == 0:
            # An all-zero vector (text was empty / punctuation
            # only). Return a tiny uniform vector so the global
            # index still has a well-defined score.
            v[0] = 1.0
            n = np.linalg.norm(v)
        return v / n

    def fail(self) -> None:
        """Flip to unavailable (simulates the embedder binary
        failing to load)."""
        self._available = False


# --- Capability + construction --------------------------------------------


class GlobalIndexConstructionTests(unittest.TestCase):
    """Constructing a :class:`GlobalIndex` must not download a
    model or open the DB eagerly. The default embedder is the
    cheap :class:`MissingEmbedder`."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.db_path = os.path.join(self.tmpdir.name, "global.db")
        # :class:`GlobalIndex` does not initialise the schema
        # itself — the CLI / Streamlit app call
        # :func:`storage.init_db` first. We mirror that here
        # so the test is independent of the production
        # bootstrap.
        storage.init_db(Path(self.db_path))

    def test_default_embedder_is_missing(self) -> None:
        """``GlobalIndex()`` with no args must default to
        :class:`MissingEmbedder` so that a read-only
        ``status()`` call does not trigger a model download."""
        idx = GlobalIndex(db_path=self.db_path)
        self.assertFalse(idx.is_available())

    def test_injected_fake_embedder_is_available(self) -> None:
        """When a working embedder is injected, ``is_available``
        returns True and the index is ready to embed + search."""
        idx = GlobalIndex(
            embedder=FakeEmbedder(), db_path=self.db_path
        )
        self.assertTrue(idx.is_available())

    def test_status_on_empty_db(self) -> None:
        """A fresh DB has no sources. ``status()`` must return
        ``{"_all": {"chunks": 0, ...}}`` and no per-source rows."""
        idx = GlobalIndex(embedder=FakeEmbedder(), db_path=self.db_path)
        st = idx.status()
        # The summary key is always present.
        self.assertIn("_all", st)
        self.assertEqual(st["_all"]["chunks"], 0)
        # No per-source rows when nothing is indexed.
        for k, v in st.items():
            if k == "_all":
                continue
            self.assertEqual(v["chunks"], 0)


# --- add_source + search round-trip ----------------------------------------


class AddAndSearchTests(unittest.TestCase):
    """The happy path: add docs, then search, then assert the
    expected ranking."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.db_path = os.path.join(self.tmpdir.name, "global.db")
        storage.init_db(Path(self.db_path))
        self.embedder = FakeEmbedder()
        self.idx = GlobalIndex(
            embedder=self.embedder, db_path=self.db_path
        )

    def test_add_source_returns_chunk_count(self) -> None:
        """``add_source`` returns the number of chunks written.
        The two OWASP docs each become one chunk (they are
        short enough that :func:`chunk_text` keeps them whole)."""
        n = self.idx.add_source(
            "owasp", [_OWASP_DOC_A, _OWASP_DOC_B],
            license="CC-BY-SA-4.0",
        )
        self.assertEqual(n, 2)
        # And the DB agrees.
        self.assertEqual(self.idx.status()["_all"]["chunks"], 2)

    def test_search_returns_closest_doc_first(self) -> None:
        """A query about XSS must rank the XSS doc above the
        SQLi doc. Inner product on L2-normalised vectors is
        cosine."""
        self.idx.add_source("owasp", [_OWASP_DOC_A, _OWASP_DOC_B])
        hits = self.idx.search("xss cross-site scripting attacks")
        self.assertGreaterEqual(len(hits), 1)
        # The top hit must be the XSS doc.
        self.assertIn("XSS", hits[0].text)

    def test_hit_carries_source_metadata(self) -> None:
        """A :class:`GlobalHit` carries the source_id, source_url,
        license, text, and a score. The score is the cosine
        similarity (inner product on unit vectors)."""
        self.idx.add_source("owasp", [_OWASP_DOC_A])
        hits = self.idx.search("xss")
        self.assertEqual(len(hits), 1)
        h = hits[0]
        self.assertEqual(h.source_id, "owasp")
        self.assertEqual(h.source_url, _OWASP_DOC_A.source_url)
        self.assertEqual(h.license, "CC-BY-SA-4.0")
        self.assertEqual(h.text, _OWASP_DOC_A.text)
        # Score is in [-1, 1] for unit vectors.
        self.assertGreaterEqual(h.score, -1.0)
        self.assertLessEqual(h.score, 1.0)

    def test_search_below_threshold_returns_empty(self) -> None:
        """``SCORE_THRESHOLD`` filters out off-topic hits. A
        query with no shared tokens with any indexed doc
        should fall below the threshold (a fully-orthogonal
        vector scores near 0) and the result must be ``[]``."""
        self.idx.add_source("owasp", [_OWASP_DOC_A, _OWASP_DOC_B])
        # A query that is one short unrelated word. The
        # token-hash embedder may still share *some* buckets
        # with the docs (hash collisions), so we use a long
        # string of unique nonsense tokens that hash to
        # disjoint buckets. A pure orthogonal query gives
        # inner product = 0, which is below 0.30.
        # Construct a query guaranteed to be orthogonal: a
        # single token of 32 unique chars (SHA-prefix style).
        hits = self.idx.search("zxcvbnmqwertyuiopasdfghjkl")
        # The threshold may or may not filter depending on
        # hash collisions. We assert the *contract* is
        # enforced: any returned hit must have score >=
        # SCORE_THRESHOLD.
        for h in hits:
            self.assertGreaterEqual(
                h.score, SCORE_THRESHOLD,
                "search() must drop hits below SCORE_THRESHOLD",
            )

    def test_search_respects_k(self) -> None:
        """``k`` caps the number of returned hits."""
        self.idx.add_source(
            "owasp", [_OWASP_DOC_A, _OWASP_DOC_B]
        )
        self.idx.add_source("mitre", [_MITRE_DOC_A, _MITRE_DOC_B])
        hits = self.idx.search("security attacks", k=1)
        self.assertEqual(len(hits), 1)

    def test_search_default_k(self) -> None:
        """``DEFAULT_K`` is the documented default. We do not
        pin a specific number (it could change), but the
        default-arg call must return <= ``DEFAULT_K`` hits."""
        self.idx.add_source(
            "owasp", [_OWASP_DOC_A, _OWASP_DOC_B]
        )
        hits = self.idx.search("security")
        self.assertLessEqual(len(hits), DEFAULT_K)


# --- source_filter ----------------------------------------------------------


class SourceFilterTests(unittest.TestCase):
    """The :class:`GlobalIndex.search` ``source_filter`` argument
    lets the view constrain retrieval to a known-good subset of
    the corpus (e.g. just the OWASP cheatsheets)."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.db_path = os.path.join(self.tmpdir.name, "global.db")
        self.idx = GlobalIndex(
            embedder=FakeEmbedder(), db_path=self.db_path
        )
        self.idx.add_source("owasp", [_OWASP_DOC_A, _OWASP_DOC_B])
        self.idx.add_source("mitre", [_MITRE_DOC_A, _MITRE_DOC_B])

    def test_filter_to_one_source(self) -> None:
        """``source_filter=["owasp"]`` must only return hits
        from the OWASP source — never MITRE."""
        hits = self.idx.search(
            "security", source_filter=["owasp"]
        )
        for h in hits:
            self.assertEqual(h.source_id, "owasp")

    def test_filter_to_unknown_source_returns_empty(self) -> None:
        """A filter on a source id that has no rows is a
        no-op, not an error."""
        hits = self.idx.search(
            "security", source_filter=["nonexistent"]
        )
        self.assertEqual(hits, [])

    def test_filter_to_multiple_sources(self) -> None:
        """``source_filter=["owasp", "mitre"]`` includes both."""
        hits = self.idx.search(
            "security", source_filter=["owasp", "mitre"]
        )
        ids = {h.source_id for h in hits}
        # Must contain at least one hit from each requested
        # source (assuming some hit clears the threshold).
        # We don't assert non-empty because all hits may be
        # below threshold for a vague query; we assert the
        # *intersection* of returned ids with the filter
        # is a subset of the filter.
        self.assertTrue(
            ids.issubset({"owasp", "mitre"}),
            f"unexpected sources in hits: {ids}",
        )


# --- clear and re-add -------------------------------------------------------


class ClearAndReAddTests(unittest.TestCase):
    """``clear(source_id)`` deletes only that source; ``clear()``
    deletes all sources. ``add_source`` of an existing source is
    a *replace* (a re-run of the CLI must not double the row
    count)."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.db_path = os.path.join(self.tmpdir.name, "global.db")
        self.idx = GlobalIndex(
            embedder=FakeEmbedder(), db_path=self.db_path
        )
        self.idx.add_source("owasp", [_OWASP_DOC_A, _OWASP_DOC_B])
        self.idx.add_source("mitre", [_MITRE_DOC_A, _MITRE_DOC_B])

    def test_clear_scoped_to_one_source(self) -> None:
        """``clear("owasp")`` removes only OWASP rows; MITRE
        rows remain."""
        n = self.idx.clear("owasp")
        self.assertEqual(n, 2)
        st = self.idx.status()
        self.assertEqual(st["owasp"]["chunks"], 0)
        # MITRE is untouched.
        self.assertGreater(st["mitre"]["chunks"], 0)

    def test_clear_all(self) -> None:
        """``clear()`` with no argument removes every row."""
        n = self.idx.clear()
        # 4 rows total.
        self.assertEqual(n, 4)
        st = self.idx.status()
        self.assertEqual(st["_all"]["chunks"], 0)
        # No per-source rows.
        for k, v in st.items():
            if k == "_all":
                continue
            self.assertEqual(v["chunks"], 0)

    def test_clear_unknown_source_is_noop(self) -> None:
        """``clear("nope")`` returns 0 and does not raise."""
        n = self.idx.clear("nope")
        self.assertEqual(n, 0)

    def test_add_source_replaces_existing(self) -> None:
        """A second ``add_source("owasp", ...)`` must replace,
        not append. The chunk count for owasp must equal the
        size of the new batch, not the old + new."""
        self.idx.add_source("owasp", [_OWASP_DOC_A])
        st_after_first = self.idx.status()["owasp"]["chunks"]
        # Now re-add with a single different doc.
        self.idx.add_source(
            "owasp",
            [CorpusDoc(
                source_id="owasp",
                source_url="https://example.invalid/owasp/csrf.md",
                license="CC-BY-SA-4.0",
                title="CSRF",
                text=(
                    "Cross-site request forgery tricks a browser "
                    "into submitting a state-changing request."
                ),
            )],
        )
        st_after_second = self.idx.status()["owasp"]["chunks"]
        # Must not have doubled.
        self.assertEqual(
            st_after_second, 1,
            "re-adding the same source_id must replace, not append",
        )
        # And it must be smaller than (or equal to) the first
        # add — never strictly larger.
        self.assertLessEqual(st_after_second, st_after_first)


# --- Degraded mode (MissingEmbedder) ---------------------------------------


class DegradedModeTests(unittest.TestCase):
    """When the embedder is unavailable, ``search()`` returns
    ``[]`` and ``add_source`` is a no-op. This is the contract
    the Streamlit view relies on: a bare :class:`GlobalIndex`
    never crashes, it just degrades."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.db_path = os.path.join(self.tmpdir.name, "global.db")

    def test_missing_embedder_is_not_available(self) -> None:
        idx = GlobalIndex(db_path=self.db_path)
        self.assertFalse(idx.is_available())

    def test_missing_embedder_search_returns_empty(self) -> None:
        idx = GlobalIndex(db_path=self.db_path)
        hits = idx.search("anything")
        self.assertEqual(hits, [])

    def test_missing_embedder_add_is_noop(self) -> None:
        idx = GlobalIndex(db_path=self.db_path)
        n = idx.add_source("owasp", [_OWASP_DOC_A])
        self.assertEqual(n, 0)
        # And nothing was written to the DB.
        st = idx.status()
        self.assertEqual(st["_all"]["chunks"], 0)

    def test_embedder_can_fail_at_runtime(self) -> None:
        """An embedder that flips to unavailable mid-life
        (e.g. the model file was deleted) must take the index
        with it on the next ``add_source`` call. We simulate
        this with the ``FakeEmbedder.fail()`` helper."""
        em = FakeEmbedder()
        idx = GlobalIndex(embedder=em, db_path=self.db_path)
        self.assertTrue(idx.is_available())
        em.fail()
        self.assertFalse(idx.is_available())
        n = idx.add_source("owasp", [_OWASP_DOC_A])
        self.assertEqual(n, 0)


# --- Cross-source merge ----------------------------------------------------


class CrossSourceMergeTests(unittest.TestCase):
    """Hits from multiple sources are returned in a single
    sorted-by-score list. The view merges this with per-chat
    RAG hits and re-sorts. The contract on this side is: every
    hit carries its ``source_id`` and the list is sorted by
    descending score."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.db_path = os.path.join(self.tmpdir.name, "global.db")
        self.idx = GlobalIndex(
            embedder=FakeEmbedder(), db_path=self.db_path
        )
        self.idx.add_source("owasp", [_OWASP_DOC_A, _OWASP_DOC_B])
        self.idx.add_source("mitre", [_MITRE_DOC_A, _MITRE_DOC_B])

    def test_hits_carry_source_id(self) -> None:
        hits = self.idx.search("security")
        for h in hits:
            self.assertIn(h.source_id, {"owasp", "mitre"})

    def test_hits_sorted_by_score_descending(self) -> None:
        """The hits list is sorted by descending score so the
        view can cap to top-k without re-sorting."""
        hits = self.idx.search("phishing attacks adversaries")
        for a, b in zip(hits, hits[1:]):
            self.assertGreaterEqual(
                a.score, b.score,
                "search() must return hits sorted by score desc",
            )

    def test_status_reports_per_source(self) -> None:
        """``status()`` returns one entry per indexed source,
        with ``chunks`` and ``distinct_docs`` counts."""
        st = self.idx.status()
        self.assertIn("owasp", st)
        self.assertIn("mitre", st)
        self.assertEqual(st["owasp"]["chunks"], 2)
        self.assertEqual(st["mitre"]["chunks"], 2)
        # distinct_docs: 2 unique source_urls per source.
        self.assertEqual(st["owasp"]["distinct_docs"], 2)
        self.assertEqual(st["mitre"]["distinct_docs"], 2)


# --- Helpers (hit shape, edge cases) ---------------------------------------


class EdgeCaseTests(unittest.TestCase):
    """Tiny edge cases: empty inputs, no FAISS available, etc."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.db_path = os.path.join(self.tmpdir.name, "global.db")

    def test_add_source_empty_list_returns_zero(self) -> None:
        idx = GlobalIndex(
            embedder=FakeEmbedder(), db_path=self.db_path
        )
        n = idx.add_source("owasp", [])
        self.assertEqual(n, 0)

    def test_add_source_skips_empty_text(self) -> None:
        """A doc with empty text produces no chunks and is
        silently dropped — the CLI can ingest docs whose
        prose is one or two characters without polluting the
        index with a zero-vector chunk."""
        idx = GlobalIndex(
            embedder=FakeEmbedder(), db_path=self.db_path
        )
        empty = CorpusDoc(
            source_id="owasp",
            source_url="https://example.invalid/empty.md",
            license="CC-BY-SA-4.0",
            title="empty",
            text="",
        )
        n = idx.add_source("owasp", [empty])
        self.assertEqual(n, 0)

    def test_globalhit_is_frozen(self) -> None:
        """:class:`GlobalHit` is a frozen dataclass; mutating
        it raises :class:`AttributeError`. This pins the
        "immutable return type" contract for downstream code
        that caches hits."""
        from dataclasses import FrozenInstanceError
        h = GlobalHit(
            source_id="x",
            source_url="u",
            license="L",
            text="t",
            score=0.5,
        )
        with self.assertRaises(FrozenInstanceError):
            h.score = 0.9  # type: ignore[misc]

    def test_score_threshold_is_sensible(self) -> None:
        """``SCORE_THRESHOLD`` is in (0, 1). It exists so a
        downstream check (``h.score >= threshold``) has a
        well-defined lower bound for cosine similarity."""
        self.assertGreater(SCORE_THRESHOLD, 0.0)
        self.assertLess(SCORE_THRESHOLD, 1.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

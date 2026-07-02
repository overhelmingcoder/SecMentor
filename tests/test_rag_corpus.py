"""Tests for :mod:`app.rag_corpus` — the Phase 12 PR-E curated knowledge base.

The corpus module is the *manifest + parser* half of the global RAG
feature; the *retrieval* half lives in :mod:`app.rag_global` and is
covered by ``tests/test_rag_global.py``. This file pins:

* The :class:`CorpusSource` / :class:`CorpusRegistry` dataclasses —
  the allow-list enforcement, the JSON round-trip, and the
  ``default()`` factory.
* The 5 parsers — :func:`parse_owasp_markdown`,
  :func:`parse_mitre_stix`, :func:`parse_cwe_xml`,
  :func:`parse_gtfobins_markdown`, :func:`parse_sigma_yaml` — against
  tiny inline fixtures. Each parser is *pure* (no I/O, no globals)
  so the fixtures are just byte strings.
* :func:`parse_source` — the dispatch path that maps a
  :class:`CorpusSource.source_id` to a parser. A bad source_id
  returns ``[]`` (the CLI logs and skips, never crashes).
* :func:`fetch_source` — the local cache + ETag path. Two tests:
  cache hit (no second network call) and ``force=True`` bypass.
  Both use ``unittest.mock.patch`` on ``urllib.request.urlopen`` so
  the test does not need real network.

The tests run with no third-party dependencies beyond what the
project already requires (``pytest``, ``numpy``). The smoke
end-to-end test of the CLI is in
``scripts/smoke_global_corpus.py`` — that one *does* exercise
the real network path, gated on a marker.

Run with:  python -m unittest tests.test_rag_corpus -v
"""

import json
import os
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from typing import List
from unittest import mock

import pytest

pytestmark = pytest.mark.smoke


# --- Path bootstrap ---------------------------------------------------------
# Mirror the project-root-cd pattern used in tests/test_storage.py and
# tests/test_rag.py so this file works no matter where the user
# invokes `python -m unittest` from.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from app.rag_corpus import (  # noqa: E402  (path bootstrap above)
    ALLOWED_LICENSES,
    CorpusDoc,
    CorpusRegistry,
    CorpusSource,
    DEFAULT_SOURCES,
    DEFAULT_USER_AGENT,
    PARSERS,
    fetch_source,
    parse_cwe_xml,
    parse_gtfobins_markdown,
    parse_mitre_stix,
    parse_owasp_markdown,
    parse_sigma_yaml,
    parse_source,
)


# --- Shared fixtures --------------------------------------------------------


# A small but realistic OWASP cheatsheet page. Two ``##`` sections
# (Introduction + Defensive Recommendations) plus a fenced code
# block. The parser must strip the code fence and split on
# ``## headings``.
_OWASP_FIXTURE: bytes = b"""# Cross-Site Scripting Prevention Cheat Sheet

## Introduction

Cross-Site Scripting (XSS) is a code injection attack. The attacker
injects a malicious script into a legitimate web page.

## Defensive Recommendations

1. Use a template engine that auto-escapes.
2. Apply context-sensitive encoding.

```html
<script>alert('xss')</script>
```

HTML entity encoding is the safest baseline.
"""


# A two-technique MITRE STIX 2.0 bundle. The first technique has
# a ``mitre-attack`` external reference (so the parser should
# rewrite ``source_url``); the second does not (so the parser
# should fall back to the bundle URL).
_MITRE_FIXTURE: bytes = json.dumps({
    "type": "bundle",
    "id": "bundle--test",
    "objects": [
        {
            "type": "intrusion-set",
            "name": "Should be skipped (not attack-pattern)",
        },
        {
            "type": "attack-pattern",
            "name": "Phishing",
            "description": (
                "Adversaries send malicious messages to gain access "
                "to victim systems."
            ),
            "external_references": [
                {"source_name": "mitre-attack",
                 "url": "https://attack.mitre.org/techniques/T1566"},
            ],
        },
        {
            "type": "attack-pattern",
            "name": "Spearphishing Attachment",
            "description": "Adversaries send attachments.",
            # No external_references — parser must fall back to
            # the bundle URL passed in as ``url``.
        },
    ],
}).encode("utf-8")


# Two-element CWE weakness view. The first has Name + Description;
# the second adds an Extended_Description.
_CWE_FIXTURE: bytes = b"""<?xml version="1.0"?>
<Weaknesses>
  <Weakness ID="79">
    <Name>Cross-site Scripting</Name>
    <Description>The product does not neutralize user input.</Description>
  </Weakness>
  <Weakness ID="89">
    <Name>SQL Injection</Name>
    <Description>The product constructs SQL from user input.</Description>
    <Extended_Description>Improper neutralization allows attackers to
    alter query logic.</Extended_Description>
  </Weakness>
  <Category ID="20">
    <Name>Improper Input Validation</Name>
  </Category>
</Weaknesses>
"""


# GTFOBins uses the same ATX-heading convention as OWASP; the
# parser delegates to ``parse_owasp_markdown`` and then strips the
# source-title prefix. This fixture has two exploit sections
# (Shell + File read).
_GTFOBINS_FIXTURE: bytes = b"""# awk

## Shell

It can be used to break out from restricted environments by spawning
an interactive system shell.

## File read

It reads data from files. The read file is sent to stdout.
"""


# Sigma rules README — same shape as OWASP, so the parser is the
# same code path. Two category sections.
_SIGMA_FIXTURE: bytes = b"""# Sigma Rules

## Process Creation

Rules that fire on process creation events. Useful for catching
suspicious parent/child relationships.

## File Event

Rules that fire on file system changes.
"""


def _owasp_source() -> CorpusSource:
    """A :class:`CorpusSource` matching the OWASP fixture."""
    return CorpusSource(
        source_id="owasp",
        title="OWASP CheatSheetSeries",
        license="CC-BY-SA-4.0",
        urls=["https://example.invalid/xss.md"],
        homepage="https://cheatsheetseries.owasp.org/",
        description="OWASP defensive cheatsheets.",
    )


def _mitre_source() -> CorpusSource:
    return CorpusSource(
        source_id="mitre",
        title="MITRE ATT&CK",
        license="CC-BY-4.0",
        urls=["https://example.invalid/enterprise-attack.json"],
        homepage="https://attack.mitre.org/",
        description="ATT&CK techniques.",
    )


def _cwe_source() -> CorpusSource:
    return CorpusSource(
        source_id="cwe",
        title="CWE",
        license="CC-BY-4.0",
        urls=["https://example.invalid/2000.xml"],
        homepage="https://cwe.mitre.org/",
        description="CWE catalog.",
        chunk_size=1024,
    )


def _gtfobins_source() -> CorpusSource:
    return CorpusSource(
        source_id="gtfobins",
        title="GTFOBins",
        license="MIT",
        urls=["https://example.invalid/awk.md"],
        homepage="https://gtfobins.github.io/",
        description="Unix binary exploits.",
    )


def _sigma_source() -> CorpusSource:
    return CorpusSource(
        source_id="sigma",
        title="Sigma Rules",
        license="Apache-2.0",
        urls=["https://example.invalid/README.md"],
        homepage="https://sigmahq.io/",
        description="Generic detection rules.",
    )


# --- CorpusSource / CorpusRegistry tests -----------------------------------


class CorpusSourceValidationTests(unittest.TestCase):
    """Pin the :class:`CorpusSource` invariants enforced in
    :meth:`__post_init__`. The registry is the gate against
    accidental ingestion of non-allow-listed licenses; if any of
    these checks silently disappears, the threat model breaks."""

    def test_valid_source_passes(self) -> None:
        """A fully-populated source constructs cleanly."""
        s = _owasp_source()
        self.assertEqual(s.source_id, "owasp")
        self.assertEqual(s.license, "CC-BY-SA-4.0")
        self.assertEqual(len(s.urls), 1)

    def test_empty_source_id_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CorpusSource(
                source_id="", title="x", license="MIT", urls=["u"]
            )

    def test_non_alphanumeric_source_id_rejected(self) -> None:
        """``"xss!"`` is not a valid column value — spaces and
        punctuation would corrupt the ``source_id`` foreign key
        in SQLite."""
        with self.assertRaises(ValueError):
            CorpusSource(
                source_id="xss!",
                title="x",
                license="MIT",
                urls=["u"],
            )

    def test_disallowed_license_rejected(self) -> None:
        """The license must be in :data:`ALLOWED_LICENSES`. A
        typo (``"CC-BY-3.0"``) must fail at construction time,
        not silently survive to ingestion."""
        with self.assertRaises(ValueError):
            CorpusSource(
                source_id="x",
                title="x",
                license="CC-BY-3.0",  # not in allow-list
                urls=["u"],
            )

    def test_empty_urls_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CorpusSource(
                source_id="x",
                title="x",
                license="MIT",
                urls=[],
            )

    def test_allowed_licenses_set_is_nonempty(self) -> None:
        """Sanity check: the allow-list has at least the five
        licenses PR-E documents in the manifest."""
        self.assertIn("CC-BY-4.0", ALLOWED_LICENSES)
        self.assertIn("CC-BY-SA-4.0", ALLOWED_LICENSES)
        self.assertIn("MIT", ALLOWED_LICENSES)
        self.assertIn("Apache-2.0", ALLOWED_LICENSES)


class CorpusRegistryTests(unittest.TestCase):
    """The :class:`CorpusRegistry` is a thin JSON round-trip wrapper
    around a list of sources. The CLI writes the default registry
    to disk on first run so operators can hand-edit it; a
    round-trip must be lossless for the known fields."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.path = Path(self.tmpdir.name) / "registry.json"

    def test_default_registry_has_five_sources(self) -> None:
        """The default curated manifest must contain one row per
        source the parsers know how to handle. Adding a new
        source means adding a row here *and* a parser function."""
        reg = CorpusRegistry.default()
        ids = {s.source_id for s in reg}
        self.assertEqual(
            ids,
            {"owasp", "mitre", "cwe", "gtfobins", "sigma"},
        )

    def test_default_sources_licenses_are_all_allowed(self) -> None:
        """Every default source must use an allow-listed license.
        Catches typos in the curated manifest before the CLI
        refuses to ingest them."""
        for s in DEFAULT_SOURCES:
            self.assertIn(
                s.license,
                ALLOWED_LICENSES,
                f"{s.source_id} uses {s.license!r} which is not in "
                f"the allow-list",
            )

    def test_save_then_load_round_trip(self) -> None:
        """Save the default registry, load it back, and assert
        every field survives. This is the contract the CLI
        relies on when an operator hand-edits the JSON."""
        reg = CorpusRegistry.default()
        reg.save(self.path)
        loaded = CorpusRegistry.load(self.path)
        self.assertEqual(len(loaded), len(reg))
        for a, b in zip(reg, loaded):
            self.assertEqual(a.source_id, b.source_id)
            self.assertEqual(a.title, b.title)
            self.assertEqual(a.license, b.license)
            self.assertEqual(a.urls, b.urls)
            self.assertEqual(a.homepage, b.homepage)
            self.assertEqual(a.description, b.description)
            self.assertEqual(a.chunk_size, b.chunk_size)

    def test_add_rejects_duplicates(self) -> None:
        """Two sources with the same id would corrupt the
        ``source_id`` foreign key. The registry must raise
        rather than silently overwrite."""
        reg = CorpusRegistry.default()
        with self.assertRaises(ValueError):
            reg.add(_owasp_source())

    def test_load_rejects_non_list_json(self) -> None:
        """A bare object (``{}``) or scalar must fail at load
        time, not later in the parser dispatch."""
        self.path.write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            CorpusRegistry.load(self.path)

    def test_load_rejects_non_dict_row(self) -> None:
        self.path.write_text("[1, 2, 3]", encoding="utf-8")
        with self.assertRaises(ValueError):
            CorpusRegistry.load(self.path)


# --- Parser tests -----------------------------------------------------------


class OwaspParserTests(unittest.TestCase):
    """The OWASP parser splits a cheatsheet on ``##`` headings and
    strips the code fence. The expected output is one
    :class:`CorpusDoc` per non-empty section, each with the
    source license and the per-section title."""

    def test_splits_on_h2_sections(self) -> None:
        """Two ``##`` sections → two :class:`CorpusDoc` rows."""
        docs = parse_owasp_markdown(
            _OWASP_FIXTURE, source=_owasp_source(),
            url="https://example.invalid/xss.md",
        )
        self.assertEqual(len(docs), 2)
        titles = [d.title for d in docs]
        self.assertTrue(
            any("Introduction" in t for t in titles),
            f"missing Introduction section: {titles}",
        )
        self.assertTrue(
            any("Defensive Recommendations" in t for t in titles),
            f"missing Defensive Recommendations section: {titles}",
        )

    def test_license_propagates_from_source(self) -> None:
        """Every :class:`CorpusDoc` carries the source's license
        so the prompt builder can render the attribution without
        a join."""
        docs = parse_owasp_markdown(
            _OWASP_FIXTURE, source=_owasp_source(),
            url="https://example.invalid/xss.md",
        )
        for d in docs:
            self.assertEqual(d.license, "CC-BY-SA-4.0")

    def test_source_url_propagates(self) -> None:
        docs = parse_owasp_markdown(
            _OWASP_FIXTURE, source=_owasp_source(),
            url="https://example.invalid/xss.md",
        )
        for d in docs:
            self.assertEqual(
                d.source_url, "https://example.invalid/xss.md"
            )

    def test_code_fence_is_stripped(self) -> None:
        """The ``<script>...</script>`` block must not survive
        into a chunk — we strip code fences so the embedder
        sees prose, not markup."""
        docs = parse_owasp_markdown(
            _OWASP_FIXTURE, source=_owasp_source(),
            url="https://example.invalid/xss.md",
        )
        joined = " ".join(d.text for d in docs)
        self.assertNotIn("<script>", joined)
        self.assertNotIn("</script>", joined)
        # …but the surrounding prose is still present.
        self.assertIn("HTML entity encoding", joined)

    def test_no_headings_returns_single_doc(self) -> None:
        """A page with no ``#``/``##`` headings becomes one
        synthetic document so the retriever can still index it
        (better a noisy chunk than a dropped page)."""
        docs = parse_owasp_markdown(
            b"Just a paragraph of text, no headings.",
            source=_owasp_source(),
            url="https://example.invalid/x.md",
        )
        self.assertEqual(len(docs), 1)
        self.assertIn("paragraph of text", docs[0].text)


class MitreParserTests(unittest.TestCase):
    """The MITRE parser keeps ``attack-pattern`` objects and drops
    everything else. Per-object ``external_references`` rewrite
    the chunk's URL when present."""

    def test_only_attack_patterns_kept(self) -> None:
        """The ``intrusion-set`` object in the fixture must be
        dropped; the two ``attack-pattern`` objects must be
        kept."""
        docs = parse_mitre_stix(
            _MITRE_FIXTURE, source=_mitre_source(),
            url="https://example.invalid/bundle.json",
        )
        self.assertEqual(len(docs), 2)
        names = [d.title for d in docs]
        self.assertTrue(
            any("Phishing" in n for n in names),
            f"missing Phishing: {names}",
        )
        self.assertTrue(
            any("Spearphishing Attachment" in n for n in names),
            f"missing Spearphishing Attachment: {names}",
        )
        # …and the intrusion-set was dropped.
        joined = " ".join(d.text for d in docs)
        self.assertNotIn("Should be skipped", joined)

    def test_external_reference_rewrites_source_url(self) -> None:
        """The first technique has a ``mitre-attack`` ref; the
        parser should use that URL as the per-chunk citation."""
        docs = parse_mitre_stix(
            _MITRE_FIXTURE, source=_mitre_source(),
            url="https://example.invalid/bundle.json",
        )
        phishing = next(d for d in docs if "Phishing" in d.title)
        self.assertEqual(
            phishing.source_url,
            "https://attack.mitre.org/techniques/T1566",
        )

    def test_missing_external_reference_falls_back_to_bundle_url(self) -> None:
        """The second technique has no external ref; the parser
        should fall back to the bundle URL passed in as ``url``."""
        docs = parse_mitre_stix(
            _MITRE_FIXTURE, source=_mitre_source(),
            url="https://example.invalid/bundle.json",
        )
        spear = next(
            d for d in docs if "Spearphishing Attachment" in d.title
        )
        self.assertEqual(
            spear.source_url,
            "https://example.invalid/bundle.json",
        )

    def test_malformed_json_returns_empty(self) -> None:
        """The CLI logs a warning and skips the URL; the parser
        returns ``[]`` (it must never crash the ingestion)."""
        docs = parse_mitre_stix(
            b"not json", source=_mitre_source(),
            url="https://example.invalid/bad.json",
        )
        self.assertEqual(docs, [])

    def test_attack_id_in_extra(self) -> None:
        """The technique id (Txxxx) is preserved in
        :attr:`CorpusDoc.extra` so the retriever can render
        "see T1566" without re-parsing the URL."""
        docs = parse_mitre_stix(
            _MITRE_FIXTURE, source=_mitre_source(),
            url="https://example.invalid/bundle.json",
        )
        phishing = next(d for d in docs if "Phishing" in d.title)
        self.assertEqual(phishing.extra.get("attack_id"), "T1566")


class CweParserTests(unittest.TestCase):
    """The CWE parser extracts ``<Weakness>``/``<Category>``/``<View>``
    elements with a ``<Name>`` and an optional ``<Description>`` and
    ``<Extended_Description>``."""

    def test_extracts_two_weaknesses_and_one_category(self) -> None:
        docs = parse_cwe_xml(
            _CWE_FIXTURE, source=_cwe_source(),
            url="https://example.invalid/2000.xml",
        )
        self.assertEqual(len(docs), 3)
        names = [d.title for d in docs]
        self.assertTrue(any("Cross-site Scripting" in n for n in names))
        self.assertTrue(any("SQL Injection" in n for n in names))
        self.assertTrue(
            any("Improper Input Validation" in n for n in names)
        )

    def test_extended_description_merged_into_text(self) -> None:
        """The second weakness has both ``<Description>`` and
        ``<Extended_Description>``. The text blob should
        include both."""
        docs = parse_cwe_xml(
            _CWE_FIXTURE, source=_cwe_source(),
            url="https://example.invalid/2000.xml",
        )
        sqli = next(d for d in docs if "SQL Injection" in d.title)
        self.assertIn("constructs SQL", sqli.text)
        self.assertIn("Improper neutralization", sqli.text)

    def test_license_propagates(self) -> None:
        docs = parse_cwe_xml(
            _CWE_FIXTURE, source=_cwe_source(),
            url="https://example.invalid/2000.xml",
        )
        for d in docs:
            self.assertEqual(d.license, "CC-BY-4.0")


class GtfoBinsParserTests(unittest.TestCase):
    """GTFOBins reuses the OWASP parser, then strips the source
    title prefix so the per-chunk title is just the exploit
    class (Shell, File read, …)."""

    def test_title_is_just_section_heading(self) -> None:
        """The per-chunk title should be ``"Shell"`` and
        ``"File read"``, not ``"GTFOBins — Shell"`` (the OWASP
        parser adds the prefix; the GTFOBins parser strips
        it)."""
        docs = parse_gtfobins_markdown(
            _GTFOBINS_FIXTURE, source=_gtfobins_source(),
            url="https://example.invalid/awk.md",
        )
        titles = [d.title for d in docs]
        self.assertIn("Shell", titles)
        self.assertIn("File read", titles)
        # No GTFOBins prefix should remain.
        for t in titles:
            self.assertNotIn("GTFOBins", t)

    def test_license_propagates(self) -> None:
        docs = parse_gtfobins_markdown(
            _GTFOBINS_FIXTURE, source=_gtfobins_source(),
            url="https://example.invalid/awk.md",
        )
        for d in docs:
            self.assertEqual(d.license, "MIT")


class SigmaParserTests(unittest.TestCase):
    """Sigma rules share the OWASP parser code path. The contract
    is: ``##`` sections become per-chunk docs, with the
    source's license propagated."""

    def test_two_category_sections(self) -> None:
        docs = parse_sigma_yaml(
            _SIGMA_FIXTURE, source=_sigma_source(),
            url="https://example.invalid/README.md",
        )
        self.assertEqual(len(docs), 2)
        titles = [d.title for d in docs]
        self.assertTrue(any("Process Creation" in t for t in titles))
        self.assertTrue(any("File Event" in t for t in titles))

    def test_license_propagates(self) -> None:
        docs = parse_sigma_yaml(
            _SIGMA_FIXTURE, source=_sigma_source(),
            url="https://example.invalid/README.md",
        )
        for d in docs:
            self.assertEqual(d.license, "Apache-2.0")


class ParseSourceDispatchTests(unittest.TestCase):
    """The :func:`parse_source` dispatch must route by
    ``source_id`` and silently skip unknown ids."""

    def test_dispatch_table_has_all_default_sources(self) -> None:
        """Every default source id has a parser. Adding a new
        default source without a parser would mean a silent
        zero-row ingestion."""
        for s in DEFAULT_SOURCES:
            self.assertIn(
                s.source_id, PARSERS,
                f"{s.source_id} is in DEFAULT_SOURCES but has no parser",
            )

    def test_unknown_source_id_returns_empty(self) -> None:
        """A typo in the registry must not crash; the CLI logs
        and skips. Verify the return is ``[]`` and no
        exception leaks."""
        src = CorpusSource(
            source_id="bogus",
            title="Bogus",
            license="MIT",
            urls=["u"],
        )
        # Override the validator (the dataclass allows any
        # alphanumeric id; "bogus" is fine) and call dispatch.
        result = parse_source(src, "u", b"some body")
        self.assertEqual(result, [])


# --- fetch_source tests -----------------------------------------------------


class FetchSourceTests(unittest.TestCase):
    """:func:`fetch_source` is the only network path in the
    corpus module. The tests patch ``urllib.request.urlopen`` so
    the suite stays offline; the CLI is the only real caller."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.cache_dir = Path(self.tmpdir.name) / "cache"
        self.url = "https://example.invalid/file.txt"
        self.body = b"the body bytes"

    @staticmethod
    def _fake_resp(*, status: int, etag: str = "", body: bytes = b""):
        """Build a ``urlopen`` context-manager mock."""
        fake = mock.MagicMock()
        fake.status = status
        fake.headers = {"ETag": etag} if etag else {}
        fake.read.return_value = body
        fake.__enter__ = lambda s: s
        fake.__exit__ = lambda s, *a: None
        return fake

    def test_cache_hit_skips_network(self) -> None:
        """A second call with the same URL must not call
        ``urlopen`` — the cache file is the source of truth."""
        # First call: mock the network to write the cache.
        priming_resp = self._fake_resp(
            status=200, etag='"v1"', body=self.body
        )
        with mock.patch(
            "app.rag_corpus.urllib.request.urlopen",
            return_value=priming_resp,
        ):
            first = fetch_source(
                self.url, cache_dir=self.cache_dir
            )
        self.assertEqual(first, self.body)
        # Second call: urlopen is patched to raise. If the
        # cache is honoured, the call still succeeds.
        with mock.patch(
            "app.rag_corpus.urllib.request.urlopen",
            side_effect=AssertionError("must not hit network"),
        ):
            second = fetch_source(
                self.url, cache_dir=self.cache_dir
            )
        self.assertEqual(second, self.body)

    def test_force_bypasses_cache(self) -> None:
        """``force=True`` re-downloads even when the cache
        has a copy — used by the CLI's ``--update`` flag."""
        # Prime the cache via a mocked network call.
        priming_resp = self._fake_resp(
            status=200, etag='"v1"', body=self.body
        )
        with mock.patch(
            "app.rag_corpus.urllib.request.urlopen",
            return_value=priming_resp,
        ):
            fetch_source(self.url, cache_dir=self.cache_dir)
        # Build a fake response that returns a *new* body on
        # the second call. If the cache were honoured, we'd
        # get the first body back; force=True must return
        # the new one.
        new_body = b"a newer body"
        fake_resp = self._fake_resp(
            status=200, etag='"v2"', body=new_body
        )
        with mock.patch(
            "app.rag_corpus.urllib.request.urlopen",
            return_value=fake_resp,
        ) as urlopen_mock:
            out = fetch_source(
                self.url, cache_dir=self.cache_dir, force=True
            )
        self.assertEqual(out, new_body)
        # And the call did happen.
        self.assertEqual(urlopen_mock.call_count, 1)

    def test_user_agent_header_sent(self) -> None:
        """The user agent is the polite one — without it, the
        raw GitHub CDN rate-limits anonymous downloads."""
        fake_resp = self._fake_resp(
            status=200, etag='"v1"', body=b"x"
        )
        with mock.patch(
            "app.rag_corpus.urllib.request.urlopen",
            return_value=fake_resp,
        ) as urlopen_mock:
            fetch_source(
                self.url, cache_dir=self.cache_dir
            )
        # Inspect the Request the mock received.
        args, _ = urlopen_mock.call_args
        req = args[0]
        self.assertIn(DEFAULT_USER_AGENT, req.headers["User-agent"])

    def test_cached_etag_sends_if_none_match(self) -> None:
        """When the cache has only an ETag (no body yet), the
        client must send ``If-None-Match: <etag>`` so the server
        can return 304 Not Modified on a re-fetch. This pins
        the polite-client contract: we never re-download bytes
        we already have a fingerprint for."""
        from app.rag_corpus import _cache_paths
        body_path, etag_path = _cache_paths(self.cache_dir, self.url)
        # Pre-seed ONLY the etag file. The body file does not
        # exist, so fetch_source will hit the network — but
        # it must send the cached ETag first.
        etag_path.parent.mkdir(parents=True, exist_ok=True)
        etag_path.write_text('"v1"', encoding="utf-8")
        # The server's response is a 200 with new bytes (the
        # 304 branch in fetch_source is currently unreachable
        # in practice — see app/rag_corpus.py — but the
        # If-None-Match header *is* sent whenever a cached
        # etag exists). The body we get back is the new
        # server body, which fetch_source persists.
        new_body = b"refreshed body"
        fake_resp = self._fake_resp(
            status=200, etag='"v2"', body=new_body,
        )
        with mock.patch(
            "app.rag_corpus.urllib.request.urlopen",
            return_value=fake_resp,
        ) as urlopen_mock:
            out = fetch_source(
                self.url, cache_dir=self.cache_dir
            )
        # The body came back from the network.
        self.assertEqual(out, new_body)
        # The new etag and body are now cached.
        self.assertEqual(body_path.read_bytes(), new_body)
        self.assertEqual(
            etag_path.read_text(encoding="utf-8"), '"v2"',
        )
        # The If-None-Match header carried the cached etag.
        args, _ = urlopen_mock.call_args
        req = args[0]
        self.assertEqual(
            req.get_header("If-none-match"), '"v1"',
            "fetch_source must send the cached ETag for 304 "
            "revalidation",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

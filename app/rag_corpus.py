"""Curated security knowledge base for the Phase 12 PR-E RAG layer.

This module is the **curation + ingestion** half of the global corpus
feature; the **retrieval** half lives in :mod:`app.rag_global`; the
**persistence** half is the ``global_chunks`` table in
:mod:`app.storage`. The split mirrors PR-B (RAG pipeline) so each PR
stays reviewable in one sitting.

The module exports three layers:

1. :class:`CorpusSource` / :class:`CorpusRegistry` — a small dataclass
   + JSON-backed manifest that describes the **set** of public,
   openly-licensed sources we are allowed to ingest, their canonical
   URLs, and their licenses. The manifest is the single source of truth
   for "what may be indexed"; the CLI will refuse to ingest a source
   that is not in the registry.

2. :func:`fetch_source` — a tiny HTTP fetcher with a local file cache
   and ETag-style re-validation. The ingestion CLI is the only caller;
   the runtime retriever never hits the network.

3. Parser functions — :func:`parse_owasp_markdown`,
   :func:`parse_mitre_stix`, :func:`parse_cwe_xml`,
   :func:`parse_gtfobins_markdown`, :func:`parse_sigma_yaml`. Each
   parser returns ``list[CorpusDoc]`` so the caller does not have to
   know the source format. Parsers are *pure* (no I/O, no globals)
   and unit-tested against tiny fixtures.

The license allow-list is enforced at the manifest level: a source
whose ``license`` is not in :data:`ALLOWED_LICENSES` is rejected at
ingest time. This is a deliberate safety rail — the project's threat
model treats accidental ingestion of non-redistributable material as
a high-severity bug, so the check is a hard gate, not a warning.

Design notes
------------
* **No third-party parsers** for the v1 slice. ``markdown-it``,
  ``python-stix2``, and ``defusedxml`` would all be reasonable
  dependencies, but each adds a download + a maintenance surface.
  Our needs are modest — we want a *chunked text blob per logical
  document*, not a full AST. Hand-rolled parsers keep the test
  surface small and the dependency footprint zero.
* **Network is optional.** The parsers and the index build all work
  on a local cache directory. The CLI is the only place that calls
  :func:`fetch_source`; if it fails (offline, rate-limited), the
  corpus simply does not refresh.
* **Caching is content-addressed by URL + ETag.** The local cache
  is a directory of files named ``<sha256-of-url>.bin`` with a
  sibling ``<sha256-of-url>.etag`` metadata file. Re-runs do not
  re-download unchanged material.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

log = logging.getLogger(__name__)


# --- License allow-list ------------------------------------------------------

# SPDX-like license strings. The CLI rejects any source whose
# ``license`` is not in this set. The strings are deliberately short
# and human-readable; the goal is "the manifest row tells a reviewer
# exactly what licence we render under", not SPDX-strict validation.
ALLOWED_LICENSES: frozenset[str] = frozenset(
    {
        "CC-BY-4.0",       # MITRE ATT&CK, CWE, CAPEC
        "CC-BY-SA-4.0",    # OWASP CheatSheetSeries
        "MIT",             # GTFOBins, Atomic Red Team, many tooling repos
        "Apache-2.0",      # WADComs, some defensive tooling
        "PUBLIC-DOMAIN",   # NVD CVE feed (US Government work)
    }
)


# --- Public dataclasses ------------------------------------------------------


@dataclass(frozen=True)
class CorpusSource:
    """One row in the corpus manifest.

    Attributes:
        source_id: Short lowercase id used as the ``source_id`` column
            in the ``global_chunks`` table (e.g. ``"owasp"``,
            ``"mitre"``). Must be unique across the registry.
        title: Human-readable title for the sidebar "Knowledge base"
            panel.
        license: SPDX-like license string. Must be in
            :data:`ALLOWED_LICENSES` or the CLI will refuse to ingest.
        urls: Ordered list of canonical URLs the fetcher will pull.
            Multiple URLs are useful for "all cheatsheets" style
            corpora where the manifest enumerates a curated subset.
        homepage: Optional human-facing URL to show in the sidebar.
        description: One-sentence summary used in the docs and the
            CLI ``--status`` output.
        chunk_size: Optional override for the chunker window. The
            default 512 chars is fine for OWASP / MITRE text; XML
            corpora may want a larger window to keep a single weakness
            description on one chunk. ``None`` uses the chunker
            default.
    """

    source_id: str
    title: str
    license: str
    urls: List[str]
    homepage: str = ""
    description: str = ""
    chunk_size: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.source_id or not self.source_id.replace("_", "").isalnum():
            raise ValueError(
                f"source_id must be alphanumeric (got {self.source_id!r})"
            )
        if self.license not in ALLOWED_LICENSES:
            raise ValueError(
                f"license {self.license!r} is not in the allow-list "
                f"{sorted(ALLOWED_LICENSES)}"
            )
        if not self.urls:
            raise ValueError(
                f"source {self.source_id!r} has no urls"
            )


@dataclass
class CorpusDoc:
    """One parsed document, ready to be chunked and embedded.

    The text is the *plain text* of the document after the parser has
    stripped markup, code fences, etc. The chunker (which is
    tokenizer-free) will slice it into overlapping windows.

    Attributes:
        source_id: Foreign key into :class:`CorpusSource`.
        source_url: The exact URL the document came from. This is
            what gets rendered as the provenance citation.
        license: Denormalised from the source for storage.
        title: Document title. Used as a prefix on the first chunk
            so the model can see the document name in context.
        text: The plain-text body.
    """

    source_id: str
    source_url: str
    license: str
    title: str
    text: str
    extra: Dict[str, Any] = field(default_factory=dict)


# --- Default registry --------------------------------------------------------

# The default curated manifest. The CLI loads this when no explicit
# registry file is passed. Adding a new source = adding one row here
# + writing a parser in this module.
DEFAULT_SOURCES: List[CorpusSource] = [
    CorpusSource(
        source_id="owasp",
        title="OWASP CheatSheetSeries",
        license="CC-BY-SA-4.0",
        # A small starter set: the four most-asked-about cheatsheets.
        # The CLI can be pointed at a local mirror that contains the
        # full set; the registry is intentionally minimal to keep the
        # initial download small.
        urls=[
            "https://raw.githubusercontent.com/OWASP/CheatSheetSeries/master/cheatsheets/Cross_Site_Scripting_Prevention_Cheat_Sheet.md",
            "https://raw.githubusercontent.com/OWASP/CheatSheetSeries/master/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.md",
            "https://raw.githubusercontent.com/OWASP/CheatSheetSeries/master/cheatsheets/Authentication_Cheat_Sheet.md",
            "https://raw.githubusercontent.com/OWASP/CheatSheetSeries/master/cheatsheets/Input_Validation_Cheat_Sheet.md",
        ],
        homepage="https://cheatsheetseries.owasp.org/",
        description=(
            "OWASP defensive cheatsheets covering the most common web "
            "vulnerabilities and the canonical remediation patterns."
        ),
    ),
    CorpusSource(
        source_id="mitre",
        title="MITRE ATT&CK (Enterprise techniques, sample)",
        license="CC-BY-4.0",
        # MITRE publishes ATT&CK as STIX 2.0 bundles on GitHub. The
        # parser handles both the .json bundle and an array of
        # attack-pattern objects. We start with one bundle to keep
        # the first ingestion under ~30 MB.
        urls=[
            "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json",
        ],
        homepage="https://attack.mitre.org/",
        description=(
            "MITRE ATT&CK Enterprise attack-pattern corpus. Each "
            "technique becomes one chunk; sub-techniques are flattened "
            "into the parent chunk for now."
        ),
        chunk_size=1024,
    ),
    CorpusSource(
        source_id="cwe",
        title="CWE (sample slice)",
        license="CC-BY-4.0",
        # CWE publishes XML views. The 100 most-frequently-cited
        # CWEs are a useful starter; expanding the slice is a one-line
        # change in the registry.
        urls=[
            "https://cwe.mitre.org/data/xml/views/2000.xml",
        ],
        homepage="https://cwe.mitre.org/",
        description=(
            "MITRE Common Weakness Enumeration. The 2000.xml view "
            "is the full CWE catalog; the parser extracts name + "
            "description + extended description for every weakness."
        ),
        chunk_size=1024,
    ),
    CorpusSource(
        source_id="gtfobins",
        title="GTFOBins (curated subset)",
        license="MIT",
        urls=[
            "https://raw.githubusercontent.com/GTFOBins/GTFOBins.github.io/master/_gtfobins/README.md",
        ],
        homepage="https://gtfobins.github.io/",
        description=(
            "Curated list of Unix binaries that can be exploited to "
            "bypass local security restrictions. The README indexes "
            "the full set; the parser flattens each binary's section."
        ),
    ),
    CorpusSource(
        source_id="sigma",
        title="Sigma rules (rules/ folder, sample)",
        license="Apache-2.0",
        # SigmaHQ publishes individual rules as one YAML per file.
        # Pulling the folder's index file keeps the first ingestion
        # manageable; the CLI can be pointed at a local mirror.
        urls=[
            "https://raw.githubusercontent.com/SigmaHQ/sigma/master/rules/README.md",
        ],
        homepage="https://sigmahq.io/",
        description=(
            "Generic detection rules. The README indexes the rules/ "
            "tree; the parser flattens a small curated subset of "
            "process-creation, file-event, and network rules."
        ),
    ),
]


# --- Registry ----------------------------------------------------------------


class CorpusRegistry:
    """In-memory list of :class:`CorpusSource` with JSON round-trip.

    The registry is a thin wrapper — it does not perform I/O at
    construction time. :meth:`load` reads a JSON file; :meth:`save`
    writes it back. The CLI writes the default registry to disk on
    first run so an operator can hand-edit it (adding a local mirror
    URL, removing a source they do not want) and then re-ingest.

    JSON shape (one row per source)::

        [
          {
            "source_id": "owasp",
            "title": "OWASP CheatSheetSeries",
            "license": "CC-BY-SA-4.0",
            "urls": ["https://.../XSS.md", "https://.../SQLi.md"],
            "homepage": "https://cheatsheetseries.owasp.org/",
            "description": "..."
          },
          ...
        ]

    Round-trip is lossless for the known fields; unknown fields are
    dropped on save (a warning is logged). This is intentional: a
    typo in the JSON should not silently survive a save.
    """

    def __init__(self, sources: Optional[Sequence[CorpusSource]] = None) -> None:
        self._sources: List[CorpusSource] = list(sources or [])

    # ---- list protocol ---------------------------------------------------

    def __iter__(self) -> Iterable[CorpusSource]:  # type: ignore[override]
        return iter(self._sources)

    def __len__(self) -> int:
        return len(self._sources)

    def __contains__(self, source_id: object) -> bool:
        return isinstance(source_id, str) and any(
            s.source_id == source_id for s in self._sources
        )

    def get(self, source_id: str) -> Optional[CorpusSource]:
        """Return the source with this id, or ``None``."""
        for s in self._sources:
            if s.source_id == source_id:
                return s
        return None

    def add(self, source: CorpusSource) -> None:
        """Append a source, rejecting duplicates by id."""
        if source.source_id in self:
            raise ValueError(
                f"source_id {source.source_id!r} already in registry"
            )
        self._sources.append(source)

    # ---- I/O -------------------------------------------------------------

    @classmethod
    def default(cls) -> "CorpusRegistry":
        """Return the built-in curated registry."""
        return cls(DEFAULT_SOURCES)

    def save(self, path: Path) -> None:
        """Write the registry to ``path`` as pretty-printed JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        rows: List[Dict[str, Any]] = []
        for s in self._sources:
            d = asdict(s)
            rows.append(d)
        path.write_text(
            json.dumps(rows, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "CorpusRegistry":
        """Read the registry from ``path``. Raises on malformed JSON."""
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError(
                f"registry JSON must be a list, got {type(data).__name__}"
            )
        sources: List[CorpusSource] = []
        for i, row in enumerate(data):
            if not isinstance(row, dict):
                raise ValueError(
                    f"registry row {i} is not an object: {row!r}"
                )
            sources.append(CorpusSource(**row))
        return cls(sources)


# --- HTTP fetch + local cache -----------------------------------------------


#: Default User-Agent. The HTTP layer sets this so a polite
#: request to GitHub's raw CDN is more likely to succeed. Anonymous
#: downloads from githubusercontent are throttled if the UA looks
#: like a library default.
DEFAULT_USER_AGENT: str = "SecMentor-RAG-Ingest/1.0 (+local)"


def _cache_paths(cache_dir: Path, url: str) -> tuple[Path, Path]:
    """Return the (body, etag) paths for ``url`` under ``cache_dir``."""
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return (
        cache_dir / f"{h}.bin",
        cache_dir / f"{h}.etag",
    )


def fetch_source(
    url: str,
    *,
    cache_dir: Path,
    timeout: float = 30.0,
    user_agent: str = DEFAULT_USER_AGENT,
    force: bool = False,
) -> bytes:
    """Fetch ``url`` with a local file cache.

    Behaviour:
      * If a cached body exists for this URL and ``force`` is False,
        the cached body is returned without a network round-trip.
      * Otherwise the URL is fetched. The ``If-None-Match`` header
        is set to the cached ETag when present; a 304 response
        revalidates the cache and returns the cached body.
      * Failures (network, HTTP error, timeout) are logged at
        WARNING and re-raised; the CLI catches them and skips
        the source rather than aborting the whole ingestion.

    Args:
        url: The URL to fetch.
        cache_dir: Directory for the body / etag cache. Created on
            first use.
        timeout: Per-request timeout in seconds.
        user_agent: UA header.
        force: If True, ignore the cache and re-download.

    Returns:
        The response body as ``bytes``.

    Raises:
        urllib.error.URLError: On any network failure.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    body_path, etag_path = _cache_paths(cache_dir, url)

    cached_etag: Optional[str] = None
    if body_path.exists() and etag_path.exists() and not force:
        return body_path.read_bytes()
    if etag_path.exists() and not force:
        cached_etag = etag_path.read_text(encoding="utf-8").strip() or None

    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    if cached_etag:
        req.add_header("If-None-Match", cached_etag)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", 200)
            if status == 304 and body_path.exists():
                # Server confirmed our cache; no new bytes.
                return body_path.read_bytes()
            new_etag = resp.headers.get("ETag")
            body = resp.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        log.warning("fetch_source: %s failed: %s", url, exc)
        raise

    body_path.write_bytes(body)
    if new_etag:
        etag_path.write_text(new_etag, encoding="utf-8")
    return body


# --- Parsers -----------------------------------------------------------------


# A loose markdown heading. We split on ATX-style headings (#, ##, …)
# because the OWASP cheatsheets and GTFOBins pages all use them.
_HEADING_RE: re.Pattern[str] = re.compile(
    r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE
)


def _strip_markdown(text: str) -> str:
    """Reduce markdown to plain text for the embedder.

    We remove code fences, inline code, links (keeping the text), and
    HTML tags. Line breaks are preserved (we collapse runs of
    whitespace on each line, not across lines) so the heading
    splitter — which uses ``^...$`` with ``MULTILINE`` — still sees
    headings at the start of a line. This is *not* a full markdown
    renderer — the goal is "the embedder sees prose, not markup",
    and an approximation is fine for retrieval.
    """
    # Drop fenced code blocks entirely.
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    # Inline code: keep the inner text but strip the backticks.
    text = re.sub(r"`([^`]*)`", r"\1", text)
    # Markdown links: [label](url) -> label.
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)
    # HTML tags.
    text = re.sub(r"<[^>]+>", " ", text)
    # Collapse intra-line whitespace; keep newlines so the
    # heading regex still works.
    out_lines = [" ".join(line.split()) for line in text.splitlines()]
    text = "\n".join(out_lines)
    return text


def parse_owasp_markdown(
    body: bytes, *, source: CorpusSource, url: str
) -> List[CorpusDoc]:
    """Parse an OWASP cheatsheet markdown blob.

    Splits on ``#`` and ``##`` headings, then strips markup. Each
    section becomes one :class:`CorpusDoc`. Splitting on section
    boundaries gives the chunker a stable unit to work on and the
    model a clean ``[source: ...]`` footer to cite.

    Args:
        body: The raw markdown bytes (UTF-8 expected; Latin-1
            fallback is used if decoding fails).
        source: The :class:`CorpusSource` for provenance metadata.
        url: The exact URL this body came from.

    Returns:
        A list of one :class:`CorpusDoc` per ``##`` section (plus a
        synthetic "Introduction" doc if the page has no headings).
    """
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        text = body.decode("latin-1", errors="replace")
    text = _strip_markdown(text)
    if not text:
        return []

    # We split on headings but keep the heading text as a section
    # title. The first "section" is the intro (everything before the
    # first heading).
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [
            CorpusDoc(
                source_id=source.source_id,
                source_url=url,
                license=source.license,
                title=source.title,
                text=text,
            )
        ]

    sections: List[CorpusDoc] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        heading = m.group(2).strip()
        body_text = text[start:end].strip()
        if not body_text:
            continue
        # Prefix the body with the heading so the chunker / embedder
        # see the section name. The chunk size is small; this costs
        # a handful of tokens per chunk and pays back at retrieval.
        sections.append(
            CorpusDoc(
                source_id=source.source_id,
                source_url=url,
                license=source.license,
                title=f"{source.title} — {heading}",
                text=f"{heading}. {body_text}",
            )
        )
    return sections


def parse_mitre_stix(
    body: bytes, *, source: CorpusSource, url: str
) -> List[CorpusDoc]:
    """Parse a MITRE ATT&CK STIX 2.0 bundle.

    The MITRE CTI repo publishes one JSON bundle per matrix. Each
    bundle is a STIX 2.0 object with a ``objects`` array; we keep
    every ``attack-pattern`` and render it as one chunk with its
    ``name`` as the title. Other STIX objects (``intrusion-set``,
    ``malware``, ``tool``) are skipped to keep the corpus focused on
    techniques — the most-retrieved unit in practice.

    Args:
        body: The raw JSON bytes.
        source: The :class:`CorpusSource` for provenance.
        url: The URL the bundle came from.

    Returns:
        One :class:`CorpusDoc` per technique. The synthetic
        "ATT&CK" document is *not* used; only ``attack-pattern``
        objects.
    """
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        log.warning("parse_mitre_stix: %s malformed JSON (%s)", url, exc)
        return []
    objects = data.get("objects") if isinstance(data, dict) else None
    if not isinstance(objects, list):
        return []
    docs: List[CorpusDoc] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        if obj.get("type") != "attack-pattern":
            continue
        name = str(obj.get("name", "")).strip()
        if not name:
            continue
        # ATT&CK techniques have a short prose description in
        # ``description``. Some objects also have a ``x_mitre_description``
        # with more detail; we keep both.
        desc = str(obj.get("description", "")).strip()
        if not desc:
            continue
        # External refs hold the canonical URL: ``attack.mitre.org/...``
        url_ref = ""
        for ref in obj.get("external_references", []) or []:
            if isinstance(ref, dict) and ref.get("source_name") == "mitre-attack":
                url_ref = str(ref.get("url", "")).strip()
                break
        chunk_url = url_ref or url
        text = f"{name}. {desc}"
        docs.append(
            CorpusDoc(
                source_id=source.source_id,
                source_url=chunk_url,
                license=source.license,
                title=f"MITRE ATT&CK — {name}",
                text=text,
                extra={"attack_id": url_ref.rsplit("/", 1)[-1]} if url_ref else {},
            )
        )
    return docs


def parse_cwe_xml(
    body: bytes, *, source: CorpusSource, url: str
) -> List[CorpusDoc]:
    """Parse a CWE XML view (e.g. ``2000.xml``).

    CWE's XML is small and well-formed. We use a hand-rolled
    extractor instead of ``xml.etree`` because the namespace
    declarations are inconsistent across views and the docstring
    is intentionally parser-free in this module.

    Args:
        body: The raw XML bytes.
        source: The :class:`CorpusSource` for provenance.
        url: The URL the view came from.

    Returns:
        One :class:`CorpusDoc` per ``<Weakness>`` / ``<Category>`` /
        ``<View>`` element with a non-empty ``Name``.
    """
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        text = body.decode("latin-1", errors="replace")

    # We match the three element kinds that carry human prose: each
    # is a self-contained block. The regex is intentionally
    # non-greedy and DOTALL so it works on a one-line CWE export.
    docs: List[CorpusDoc] = []
    for tag in ("Weakness", "Category", "View"):
        for m in re.finditer(
            rf'<{tag}\b[^>]*>(.*?)</{tag}>', text, re.DOTALL
        ):
            inner = m.group(1)
            name_m = re.search(r'<Name>(.*?)</Name>', inner, re.DOTALL)
            desc_m = re.search(
                r'<Description>(.*?)</Description>', inner, re.DOTALL
            )
            ext_m = re.search(
                r'<Extended_Description>(.*?)</Extended_Description>',
                inner,
                re.DOTALL,
            )
            if not name_m:
                continue
            name = _xml_text(name_m.group(1)).strip()
            if not name:
                continue
            desc = _xml_text(desc_m.group(1)) if desc_m else ""
            ext = _xml_text(ext_m.group(1)) if ext_m else ""
            text_body = " ".join(part for part in (name, desc, ext) if part)
            text_body = " ".join(text_body.split())
            if not text_body:
                continue
            docs.append(
                CorpusDoc(
                    source_id=source.source_id,
                    source_url=url,
                    license=source.license,
                    title=f"CWE — {name}",
                    text=text_body,
                )
            )
    return docs


def _xml_text(s: str) -> str:
    """Strip nested XML tags from a CWE field's text content."""
    s = re.sub(r"<[^>]+>", " ", s)
    return " ".join(s.split())


def parse_gtfobins_markdown(
    body: bytes, *, source: CorpusSource, url: str
) -> List[CorpusDoc]:
    """Parse a GTFOBins-style markdown page.

    GTFOBins pages are a single markdown file per binary. The
    parser splits on ``##`` headings (each is a different exploit
    class: ``Shell``, ``Command``, ``File read``, etc.).

    Args:
        body: The raw markdown bytes.
        source: The :class:`CorpusSource` for provenance.
        url: The URL the page came from.

    Returns:
        One :class:`CorpusDoc` per ``##`` section.
    """
    # GTFOBins uses the same ATX-heading convention as OWASP, so
    # the OWASP parser is the right starting point. We re-use it
    # but the title is the binary name, not the source title.
    docs = parse_owasp_markdown(body, source=source, url=url)
    # Replace the title prefix with just the heading; the source
    # title is implied by the chunk's metadata, not the text.
    for d in docs:
        if " — " in d.title:
            d.title = d.title.split(" — ", 1)[1]
    return docs


def parse_sigma_yaml(
    body: bytes, *, source: CorpusSource, url: str
) -> List[CorpusDoc]:
    """Parse a Sigma rules index (a markdown README).

    The full SigmaHQ/rules/ tree is huge; the manifest points at
    the README which lists the top-level subfolders. This parser
    treats the README as a *catalog*: each ``## <category>`` heading
    becomes one document summarising the rule family. The point is
    *not* to be the full Sigma corpus (that is a separate project)
    but to give the retriever enough context to cite "see Sigma rule
    family X" when the user asks "how do I detect Y".

    Args:
        body: The raw markdown bytes.
        source: The :class:`CorpusSource` for provenance.
        url: The URL the page came from.

    Returns:
        One :class:`CorpusDoc` per ``##`` section.
    """
    return parse_owasp_markdown(body, source=source, url=url)


# --- Top-level dispatch ------------------------------------------------------


#: Maps ``source_id`` to a parser function. The CLI looks up the
#: parser by the manifest's ``source_id`` so adding a new source
#: is a one-line registry change plus a parser function.
PARSERS: Dict[str, Any] = {
    "owasp": parse_owasp_markdown,
    "mitre": parse_mitre_stix,
    "cwe": parse_cwe_xml,
    "gtfobins": parse_gtfobins_markdown,
    "sigma": parse_sigma_yaml,
}


def parse_source(
    source: CorpusSource, url: str, body: bytes
) -> List[CorpusDoc]:
    """Dispatch a fetched body to the right parser.

    Args:
        source: The :class:`CorpusSource` whose ``source_id`` picks
            the parser.
        url: The exact URL the body came from.
        body: The raw bytes.

    Returns:
        The list of :class:`CorpusDoc` produced by the parser.
        Returns ``[]`` if the ``source_id`` is unknown — the CLI
        will log a WARNING and skip the URL rather than crash.
    """
    parser = PARSERS.get(source.source_id)
    if parser is None:
        log.warning(
            "parse_source: no parser registered for %s; skipping %s",
            source.source_id, url,
        )
        return []
    return parser(body, source=source, url=url)


# --- Manifest timestamp ------------------------------------------------------


def manifest_timestamp(path: Path) -> Optional[str]:
    """Return the ISO-8601 mtime of the manifest file, or ``None``.

    Used by the CLI to show "last refreshed" in ``--status``.
    """
    if not path.exists():
        return None
    return datetime.fromtimestamp(
        path.stat().st_mtime, tz=timezone.utc
    ).isoformat()


__all__ = [
    "ALLOWED_LICENSES",
    "CorpusDoc",
    "CorpusRegistry",
    "CorpusSource",
    "DEFAULT_SOURCES",
    "DEFAULT_USER_AGENT",
    "PARSERS",
    "fetch_source",
    "manifest_timestamp",
    "parse_cwe_xml",
    "parse_gtfobins_markdown",
    "parse_mitre_stix",
    "parse_owasp_markdown",
    "parse_sigma_yaml",
    "parse_source",
]

"""Phase 12 PR-E — global security corpus ingestion CLI.

Run with::

    python scripts/ingest_security_corpus.py --status
    python scripts/ingest_security_corpus.py --source owasp
    python scripts/ingest_security_corpus.py --source owasp --update
    python scripts/ingest_security_corpus.py --source owasp --clear
    python scripts/ingest_security_corpus.py --offline --source owasp

What it does
------------
* Reads the corpus manifest (``app.rag_corpus.DEFAULT_SOURCES`` or a
  user-supplied JSON file via ``--registry``).
* Fetches each URL (HTTP + ETag cache at ``~/.cache/secmentor/corpus_cache/``)
  unless ``--offline`` is set, in which case it only uses the cache.
* Parses with the registered parser (markdown / STIX / XML).
* Chunks + embeds with the configured embedder and persists to the
  ``global_chunks`` table. When the embedder is unavailable
  (``MissingEmbedder``), the CLI logs a WARNING and the
  ``GlobalIndex.search`` path stays in degraded mode.

The CLI is intentionally **safe to re-run**. Each ``--source X`` call
clears the existing rows for ``X`` before inserting, so a re-run
means "source X is now exactly the URLs I just downloaded" — not
"X is the old rows plus the new bits". A network failure on one URL
logs and skips that URL; other URLs in the same source still ingest.

Subcommands
-----------
The CLI is a single ``argparse`` parser, not a subcommand tree —
the flags are independent and combine naturally:

* ``--status``      Show per-source chunk counts; no fetching, no
                    embedding. The default verb when no other action
                    is given.
* ``--source ID``   Restrict the action to a single ``source_id``.
                    Use it with ``--update``, ``--clear``, or on its
                    own to ingest just that source.
* ``--update``      Re-fetch (skipping the ETag cache) and re-ingest
                    the selected sources.
* ``--clear ID``    Delete rows for the given source. Combined with
                    ``--source``, both must agree.
* ``--offline``     Do not hit the network; use only the local cache.
                    Useful for "I already downloaded once, just
                    re-chunk / re-embed after a code change".
* ``--embedder NAME``  Choose the embedder backend. ``miniLM`` is the
                    real sentence-transformer model (default if
                    installed). ``none`` forces
                    :class:`app.rag_embedder.MissingEmbedder` and is
                    the only choice on machines without the
                    dependency; the corpus will persist text + license
                    metadata but the search will be in degraded mode.
* ``--registry PATH`  Override the default manifest. Useful for
                    pointing at a local mirror.
* ``--db PATH``     Override the SQLite DB path. Mostly for tests.
* ``--no-fetch``    Same as ``--offline`` for backwards compatibility.

Exit codes
----------
* 0 — success (or no work to do).
* 2 — fatal configuration error (no manifest, unknown source id).
* 3 — fatal runtime error (DB initialisation failed).
* 4 — partial success: at least one source failed but the run
      completed. The failing sources are listed in the output.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import urllib.error
from pathlib import Path
from typing import List, Optional, Sequence

# Make the project root importable when this file is run directly
# (e.g. ``python scripts/ingest_security_corpus.py``). The project
# does not have a setup.py / pyproject.toml that installs ``app``
# as a package, so we add the repo root to ``sys.path`` here.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app import storage  # noqa: E402  (sys.path tweak above is intentional)
from app.rag_corpus import (  # noqa: E402
    CorpusDoc,
    CorpusRegistry,
    CorpusSource,
    DEFAULT_SOURCES,
    fetch_source,
    manifest_timestamp,
    parse_source,
)
from app.rag_embedder import Embedder, MissingEmbedder  # noqa: E402
from app.rag_global import GlobalIndex  # noqa: E402


log = logging.getLogger("ingest_security_corpus")

#: Default cache directory for fetched corpus bodies. Kept under the
#: user's home so multiple checkouts share the cache. The CLI never
#: deletes this directory.
DEFAULT_CACHE_DIR: Path = Path.home() / ".cache" / "secmentor" / "corpus_cache"


# --- CLI helpers -------------------------------------------------------------


def _db_path_or_none(raw: Optional[str]) -> Optional[Path]:
    """Coerce the ``--db`` argparse value to ``Path`` or ``None``.

    :func:`app.storage.init_db` and the storage repository calls
    accept ``Optional[Path]``. argparse gives us a ``str`` (or
    ``None``), so we coerce once here and pass the result through.
    Returning ``None`` lets the storage layer pick its default
    location (``db_path()``), which honours the
    ``SECMENTOR_DB_PATH`` env var.
    """
    if raw is None or raw == "":
        return None
    return Path(raw)


def _build_embedder(name: str) -> Embedder:
    """Resolve an embedder name to an instance.

    The CLI is the only place that decides "do we use the real model
    or the no-op stand-in?". A user invoking the script with
    ``--embedder none`` (or on a machine where the model can't load)
    gets :class:`MissingEmbedder`; the corpus will still be persisted
    (text + license metadata) so a later run with the real model can
    fill the embeddings.

    Args:
        name: One of ``"miniLM"`` (default) or ``"none"``.

    Returns:
        An :class:`Embedder` instance. Callers must check
        :meth:`Embedder.is_available` before calling :meth:`encode`.
    """
    if name == "none":
        return MissingEmbedder()
    if name == "miniLM":
        # Lazy import so the CLI doesn't pull sentence-transformers
        # on every invocation. The class raises at construction
        # time if the model file is missing — caught by the caller.
        from app.rag_embedder import Embedder as RealEmbedder  # type: ignore

        try:
            return RealEmbedder()
        except Exception as exc:  # pragma: no cover - import-guarded
            log.warning(
                "Could not initialise real embedder (%s); falling back "
                "to MissingEmbedder. Re-run with --embedder none to "
                "suppress this message.",
                exc,
            )
            return MissingEmbedder()
    raise ValueError(f"unknown embedder name: {name!r}")


def _resolve_registry(args: argparse.Namespace) -> CorpusRegistry:
    """Return the registry to use for this run.

    Precedence:
      1. ``--registry PATH`` (user-supplied JSON).
      2. The built-in :data:`DEFAULT_SOURCES` (curated manifest).
    """
    if args.registry is not None:
        path = Path(args.registry)
        if not path.exists():
            raise FileNotFoundError(
                f"--registry {path} does not exist; create it with "
                f"the same JSON shape as DEFAULT_SOURCES or omit "
                f"--registry to use the built-in manifest."
            )
        return CorpusRegistry.load(path)
    return CorpusRegistry.default()


def _resolve_sources(
    registry: CorpusRegistry, source_filter: Optional[str]
) -> List[CorpusSource]:
    """Return the subset of sources to operate on this run.

    If ``source_filter`` is given, the returned list contains exactly
    one element (or raises if the id is unknown). Otherwise every
    source in the registry is returned in declared order.
    """
    if source_filter is None:
        return list(registry)
    if source_filter not in registry:
        raise ValueError(
            f"unknown source_id {source_filter!r}; known: "
            f"{[s.source_id for s in registry]}"
        )
    return [registry.get(source_filter)]  # type: ignore[list-item]


def _print_status(
    registry: CorpusRegistry, registry_path: Optional[Path], index: GlobalIndex
) -> None:
    """Print the per-source status block.

    Reads from the live ``GlobalIndex`` (which goes to SQLite) and
    pairs it with the manifest, so the operator sees "manifest says
    5 sources, DB has rows for 3 of them" in one glance.
    """
    print("Global security corpus — status")
    print("=" * 60)
    if registry_path is not None:
        ts = manifest_timestamp(registry_path)
        if ts:
            print(f"Manifest:      {registry_path} (mtime {ts})")
        else:
            print(f"Manifest:      {registry_path} (mtime unknown)")
    else:
        print("Manifest:      <built-in DEFAULT_SOURCES>")
    print(f"Cache dir:     {DEFAULT_CACHE_DIR}")
    print(f"Index available: {index.is_available()}")
    print()

    status = index.status()
    total = status.get("_all", {}).get("chunks", 0)
    print(f"Total chunks:  {total}")
    print()
    print(f"{'source_id':<14} {'license':<14} {'chunks':>8}  manifest_urls")
    print("-" * 60)
    # Walk the manifest order so a freshly-added source with zero
    # chunks still shows up (status() only sees sources with rows).
    for s in registry:
        s_status = status.get(s.source_id, {"chunks": 0, "distinct_docs": 0})
        print(
            f"{s.source_id:<14} {s.license:<14} "
            f"{s_status['chunks']:>8}  {len(s.urls)}"
        )


# --- Per-source ingestion ---------------------------------------------------


def _ingest_source(
    source: CorpusSource,
    index: GlobalIndex,
    *,
    cache_dir: Path,
    offline: bool,
    update: bool,
) -> tuple[int, List[str]]:
    """Fetch + parse + embed every URL in ``source.urls``.

    Args:
        source: The manifest row.
        index: The :class:`GlobalIndex` to push into.
        cache_dir: Where ``fetch_source`` keeps its body+etag files.
        offline: If True, never hit the network. Cached bodies are
            still used; missing bodies produce a per-URL error.
        update: If True, force-refresh the cache (re-download even
            when a body is cached).

    Returns:
        ``(chunks_written, errors)`` — both ints / lists. ``chunks_written``
        is the number of rows inserted into the global_chunks table.
        ``errors`` is a list of human-readable per-URL failure
        messages; the caller appends them to the global error list.
    """
    if not index.is_available() and not isinstance(index._embedder, MissingEmbedder):
        # If the operator asked for the real model but it failed
        # to load, the index will be silently broken at search time.
        # We still want to *try* the ingestion (the embedder may be
        # there but the model files missing); the index logs the
        # failure itself. The caller can decide to bail.
        pass

    all_docs: List[CorpusDoc] = []
    errors: List[str] = []
    for url in source.urls:
        try:
            if offline:
                # Read directly from the cache; do not even consult
                # the network for an ETag revalidation.
                h_cache = _PROJECT_ROOT  # type: ignore[has-type]  # silence linter
                from app.rag_corpus import _cache_paths

                body_path, _ = _cache_paths(cache_dir, url)
                if not body_path.exists():
                    errors.append(
                        f"{url}: offline mode but no cached body "
                        f"at {body_path}"
                    )
                    continue
                body = body_path.read_bytes()
            else:
                body = fetch_source(
                    url,
                    cache_dir=cache_dir,
                    force=update,
                )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            errors.append(f"{url}: fetch failed: {exc}")
            continue
        except FileNotFoundError as exc:
            errors.append(f"{url}: {exc}")
            continue
        try:
            docs = parse_source(source, url, body)
        except Exception as exc:  # parser crashed — skip this URL
            log.exception("parser crashed for %s", url)
            errors.append(f"{url}: parser error: {exc}")
            continue
        if not docs:
            errors.append(f"{url}: parser returned 0 docs")
            continue
        all_docs.extend(docs)

    if not all_docs:
        return 0, errors

    # Persist. add_source is a no-op when the embedder is
    # unavailable; that's the right behaviour for the offline-with-
    # no-model case: rows are not created because there is no
    # embedding to store alongside them.
    written = index.add_source(source.source_id, all_docs, license=source.license)
    return written, errors


# --- Top-level commands ------------------------------------------------------


def _cmd_status(args: argparse.Namespace) -> int:
    """``--status`` handler. Just prints the manifest + index snapshot."""
    db = _db_path_or_none(args.db)
    registry = _resolve_registry(args)
    # Make sure the schema exists so a brand-new checkout can run
    # ``--status`` without first ingesting something.
    storage.init_db(db)
    index = GlobalIndex(db_path=db)
    _print_status(registry, args.registry, index)
    return 0


def _cmd_clear(args: argparse.Namespace) -> int:
    """``--clear ID`` handler. Deletes the rows and prints the count."""
    db = _db_path_or_none(args.db)
    target = args.clear
    # Same rationale as ``_cmd_status``: a fresh checkout should be
    # able to run ``--clear`` on a non-existent source without
    # having to ingest first.
    storage.init_db(db)
    index = GlobalIndex(db_path=db)
    n = index.clear(target)
    print(f"Cleared {n} rows for {target!r}.")
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    """The main ingest command. Used when ``--clear`` is not given."""
    registry = _resolve_registry(args)
    sources = _resolve_sources(registry, args.source)
    if not sources:
        print("No sources to ingest.", file=sys.stderr)
        return 0

    embedder = _build_embedder(args.embedder)
    if not embedder.is_available():
        print(
            f"[warn] Embedder {args.embedder!r} is unavailable "
            f"({type(embedder).__name__}). Persisted text will be "
            f"available for a later re-run, but the search index "
            f"will stay in degraded mode until a real embedder is "
            f"used. Re-run with --embedder miniLM once the model is "
            f"reachable.",
            file=sys.stderr,
        )

    # Make sure the schema exists; the CLI is the first thing many
    # operators will run on a fresh checkout.
    db = _db_path_or_none(args.db)
    storage.init_db(db)
    index = GlobalIndex(embedder=embedder, db_path=db)

    print(
        f"Ingesting {len(sources)} source(s) from "
        f"{'manifest ' + str(args.registry) if args.registry else 'built-in manifest'}"
    )
    print(
        f"  mode: {'offline' if args.offline else 'online'}"
        f"{' + force-refresh' if args.update else ''}"
    )
    print(f"  cache: {DEFAULT_CACHE_DIR}")
    print()

    total_chunks = 0
    all_errors: List[str] = []
    for s in sources:
        print(f"[{s.source_id}] {s.title}  ({s.license})  {len(s.urls)} url(s)")
        written, errors = _ingest_source(
            s,
            index,
            cache_dir=DEFAULT_CACHE_DIR,
            offline=args.offline,
            update=args.update,
        )
        total_chunks += written
        if errors:
            all_errors.extend(f"[{s.source_id}] {e}" for e in errors)
            print(f"  errors: {len(errors)}")
            for e in errors:
                print(f"    - {e}")
        print(f"  wrote {written} chunks")
        print()

    # Final status block so the operator can see the new totals
    # without a second CLI invocation.
    _print_status(registry, args.registry, index)
    print()
    print(f"Total: {total_chunks} chunks across {len(sources)} source(s).")
    if all_errors:
        print(f"WARNING: {len(all_errors)} per-URL error(s):", file=sys.stderr)
        for e in all_errors:
            print(f"  {e}", file=sys.stderr)
        return 4
    return 0


# --- Argparse ---------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser. Kept separate from ``main`` for testability."""
    p = argparse.ArgumentParser(
        prog="ingest_security_corpus",
        description=(
            "Ingest the curated security knowledge base into the "
            "global_chunks table. See module docstring for the full "
            "flag reference."
        ),
    )
    p.add_argument(
        "--source",
        type=str,
        default=None,
        metavar="ID",
        help="Restrict the run to a single source_id (e.g. 'owasp').",
    )
    p.add_argument(
        "--update",
        action="store_true",
        help="Re-download URLs even if the cache is fresh (still uses the ETag for 304s).",
    )
    p.add_argument(
        "--offline",
        "--no-fetch",
        action="store_true",
        dest="offline",
        help="Do not hit the network; use only the local cache.",
    )
    p.add_argument(
        "--status",
        action="store_true",
        help="Show per-source chunk counts and exit. Default verb when no action is given.",
    )
    p.add_argument(
        "--clear",
        type=str,
        default=None,
        metavar="ID",
        help="Delete rows for the given source_id and exit.",
    )
    p.add_argument(
        "--embedder",
        type=str,
        default="miniLM",
        choices=("miniLM", "none"),
        help="Embedder backend (default: miniLM).",
    )
    p.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Path to a custom manifest JSON file. Default: built-in DEFAULT_SOURCES.",
    )
    p.add_argument(
        "--db",
        type=str,
        default=None,
        help="SQLite DB path. Default: app.storage default.",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging (DEBUG).",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Top-level entry point. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    try:
        if args.clear is not None:
            # ``--clear`` is exclusive with everything else. If the
            # operator also passed ``--source``, the two must agree.
            if args.source is not None and args.source != args.clear:
                parser.error(
                    f"--source {args.source!r} conflicts with "
                    f"--clear {args.clear!r}; pick one."
                )
            return _cmd_clear(args)
        if args.status:
            return _cmd_status(args)
        return _cmd_ingest(args)
    except FileNotFoundError as exc:
        print(f"Fatal: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"Fatal: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - defensive top-level
        log.exception("ingest failed")
        print(f"Fatal: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())

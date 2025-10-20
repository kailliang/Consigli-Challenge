"""Unified ingestion script: chunking + embeddings + persistence.

This script consolidates the old quick Markdown chunker and the Typer-based
CLI into a single entry point that:
 - Parses supported documents (Markdown, PDF, DOCX)
 - Splits into chunks and writes `data/chunks.json`
 - Generates embeddings and upserts to Chroma
 - Persists table summaries to SQLite

Defaults mirror prior behavior so `python scripts/quick_ingest_markdown.py`
continues to regenerate `data/chunks.json` using the sample `data/*` folders.
Use `--dry-run` to only write chunks without embeddings.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

# Ensure ingest sources are importable without installing a package
ROOT = Path(__file__).resolve().parents[1]
INGEST_SRC = ROOT / "ingest" / "src"
sys.path.insert(0, str(INGEST_SRC))

from ingest.pipeline import IngestionPipeline, PipelineConfig  # noqa: E402
from ingest.chunking import Chunk  # noqa: E402


ENV_LOCATIONS: Tuple[Path, ...] = (ROOT / "backend" / ".env", ROOT / ".env")
DOC_SUFFIXES = {".md", ".pdf", ".docx"}


def _configure_logging(level_name: str) -> None:
    """Configure logging with compact debug output that highlights key details."""

    level = getattr(logging, level_name.upper(), logging.INFO)
    # When debugging, keep output to bare messages so only key facts surface.
    format_str = "%(message)s" if level <= logging.DEBUG else "%(asctime)s %(levelname)s %(name)s - %(message)s"
    logging.basicConfig(level=level, format=format_str)

    # Suppress verbose dependency logs that drown out the useful debug lines.
    for noisy_logger in ("httpx", "urllib3", "openai"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def _load_env_files(locations: Iterable[Path]) -> None:
    for location in locations:
        if not location.exists():
            continue
        for raw_line in location.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = value


def _resolve_input_paths(paths: list[Path]) -> list[Path]:
    resolved: list[Path] = []
    for path in paths:
        if path.is_dir():
            resolved.extend(sorted(p for p in path.rglob("*") if p.is_file()))
        else:
            resolved.append(path)

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in resolved:
        if path.exists() and path.is_file() and path.suffix.lower() in DOC_SUFFIXES and path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def _infer_year_from_name(path: Path) -> Optional[int]:
    import re

    match = re.search(r"(19|20)\d{2}", path.stem)
    return int(match.group(0)) if match else None


def _group_documents_by_metadata(
    files: Iterable[Path],
    *,
    company_override: Optional[str] = None,
    year_override: Optional[int] = None,
) -> Dict[tuple[str | None, int | None], list[Path]]:
    grouped: Dict[tuple[str | None, int | None], list[Path]] = {}
    for file_path in files:
        company = company_override or file_path.parent.name
        year = year_override if year_override is not None else _infer_year_from_name(file_path)
        key = (company, year)
        grouped.setdefault(key, []).append(file_path)
    return grouped


def _load_chunk_items_from_json(chunks_path: Path) -> list[tuple[str, Chunk]]:
    import json

    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks file not found: {chunks_path}")

    data = json.loads(chunks_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("chunks file must contain a list of chunk records")

    items: list[tuple[str, Chunk]] = []
    for rec in data:
        if not isinstance(rec, dict):
            continue
        chunk_id = str(rec.get("chunk_id") or "").strip()
        text = str(rec.get("text") or "")
        if not chunk_id or not text.strip():
            continue
        # Reconstruct metadata by excluding known fields
        metadata = {k: v for k, v in rec.items() if k not in {"chunk_id", "text", "order"}}
        items.append((chunk_id, Chunk(text=text, metadata=metadata)))
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="Chunk, embed and ingest reports into local stores.")
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=[
            ROOT / "data" / "BMW",
            ROOT / "data" / "Tesla",
            ROOT / "data" / "Ford",
            ROOT / "data" / "General",
        ],
        help="Files or directories to ingest (default: sample data folders).",
    )
    parser.add_argument("--company", "-c", type=str, default=None, help="Company short name (override).")
    parser.add_argument("--year", "-y", type=int, default=None, help="Reporting year (override).")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=ROOT / "data",
        help="Output directory for manifests and chunks.json (default: data/).",
    )
    parser.add_argument("--currency", type=str, default=None, help="Document currency (e.g., USD).")
    parser.add_argument("--reporting-basis", type=str, default=None, help="Accounting basis (IFRS, US GAAP, etc.).")
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=ROOT / "data" / "chroma",
        help="Directory for Chroma persistence (default: data/chroma).",
    )
    parser.add_argument(
        "--structured-db",
        type=Path,
        default=ROOT / "data" / "metrics.db",
        help="Path to SQLite structured store (default: data/metrics.db).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Skip embeddings and database writes.")
    parser.add_argument("--chunk-size", type=int, default=400, help="Target chunk size in tokens (default: 400).")
    parser.add_argument("--chunk-overlap", type=int, default=40, help="Token overlap between chunks (default: 40).")
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Embedding model name (default: $EMBEDDINGS_MODEL or text-embedding-3-large).",
    )
    parser.add_argument("--openai-api-key", type=str, default=None, help="OpenAI API key (falls back to env).")
    parser.add_argument("--openai-api-base", type=str, default=None, help="Optional OpenAI API base URL.")
    parser.add_argument("--chroma-batch-size", type=int, default=32, help="Embedding batch size for Chroma upserts.")
    parser.add_argument("--chroma-concurrency", type=int, default=32, help="Maximum concurrent embedding requests.")
    parser.add_argument("--collection-name", type=str, default=None, help="Override the Chroma collection name.")
    parser.add_argument(
        "--mode",
        choices=["chunk", "ingest", "both"],
        default="both",
        help="What to run: chunk only, ingest from chunks.json only, or both (default).",
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=None,
        help="Path to chunks.json when using --mode ingest (defaults to <output-dir>/chunks.json).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO or $LOG_LEVEL.",
    )

    args = parser.parse_args()

    _configure_logging(args.log_level)

    _load_env_files(ENV_LOCATIONS)

    # Resolve and validate inputs (for modes that parse/chunk)
    resolved_paths = _resolve_input_paths([Path(p) for p in args.inputs])
    chunks_path = args.chunks_path or (args.output_dir / "chunks.json")

    api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    api_base = args.openai_api_base or os.getenv("OPENAI_API_BASE")
    embedding_model = args.embedding_model or os.getenv("EMBEDDINGS_MODEL") or "text-embedding-3-large"

    collection_name = (
        args.collection_name or (f"annual-reports-{args.company.lower()}" if args.company else "annual-reports")
    )

    if args.mode in {"both", "chunk"}:
        if not resolved_paths:
            print("No eligible files found for chunking.")
            return 1

        # Prepare common config for pipeline
        common_kwargs = dict(
            output_dir=args.output_dir,
            currency=args.currency,
            reporting_basis=args.reporting_basis,
            dry_run=(args.mode == "chunk") or args.dry_run,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            embedding_model=embedding_model,
            chroma_path=args.chroma_dir,
            structured_db_path=args.structured_db,
            collection_name=collection_name,
            openai_api_key=api_key,
            openai_api_base=api_base,
            chroma_batch_size=args.chroma_batch_size,
            chroma_concurrency=args.chroma_concurrency,
        )

        # If both company and year provided, run as a single group
        if args.company is not None and args.year is not None:
            label = f"{args.company} {args.year}"
            print(f"Starting ingestion for {label}...")
            pipeline = IngestionPipeline(
                config=PipelineConfig(
                    input_paths=resolved_paths,
                    company=args.company,
                    year=args.year,
                    append_chunks=False,
                    **common_kwargs,  # type: ignore[arg-type]
                )
            )
            result = pipeline.run()
            print("Pipeline finished")
            print(f"Documents indexed: {result.documents_indexed}")
            print(f"Tables indexed: {result.tables_indexed}")
            print(f"Tokens used (approx): {result.tokens_used}")
            print(f"Manifest path: {result.manifest_path}")
            print(f"Chunks path: {result.chunks_path}")
        else:
            # Group by inferred company/year and ingest sequentially
            grouped = _group_documents_by_metadata(
                resolved_paths,
                company_override=args.company,
                year_override=args.year,
            )
            if not grouped:
                print("No eligible documents discovered for bulk ingestion.")
                return 1

            if args.mode == "both" and not api_key:
                print("OpenAI API key not provided via flag or environment.")
                return 2

            # For fresh runs, overwrite chunks.json
            if args.mode == "both" and chunks_path.exists():
                try:
                    chunks_path.unlink()
                except Exception:
                    pass

            total_docs = 0
            total_tables = 0
            total_tokens = 0
            last_chunks_path: Optional[Path] = None
            manifest_paths: list[Path] = []

            print(f"Bulk ingestion detected ({len(grouped)} company/year groups).")

            sorted_groups = sorted(
                grouped.items(),
                key=lambda item: (
                    item[0][0] or "",
                    item[0][1] if item[0][1] is not None else -1,
                ),
            )

            for index, ((auto_company, auto_year), files) in enumerate(sorted_groups):
                effective_company = args.company or auto_company
                effective_year = args.year or auto_year
                display_year = effective_year if effective_year is not None else "unknown"
                print(f"-> {effective_company} {display_year} ({len(files)} files)")

                pipeline = IngestionPipeline(
                    config=PipelineConfig(
                        input_paths=files,
                        company=effective_company,
                        year=effective_year,
                        append_chunks=index > 0,
                        **common_kwargs,  # type: ignore[arg-type]
                    )
                )
                result = pipeline.run()
                total_docs += result.documents_indexed
                total_tables += result.tables_indexed
                total_tokens += result.tokens_used
                manifest_paths.append(result.manifest_path)
                last_chunks_path = result.chunks_path

            print("Bulk pipeline complete")
            print(f"Total documents indexed: {total_docs}")
            print(f"Total tables indexed: {total_tables}")
            print(f"Total tokens used (approx): {total_tokens}")
            if manifest_paths:
                print(f"Manifest paths: {', '.join(str(path) for path in manifest_paths)}")
            if last_chunks_path:
                print(f"Combined chunks path: {last_chunks_path}")

        if args.mode == "chunk":
            # Already wrote chunks.json via pipeline; exit now
            return 0

    if args.mode == "ingest":
        # Ingest from existing chunks.json into Chroma only
        if not api_key:
            print("OpenAI API key not provided via flag or environment.")
            return 2
        try:
            items = _load_chunk_items_from_json(chunks_path)
        except Exception as exc:
            print(f"Failed to load chunks: {exc}")
            return 1
        if not items:
            print("No chunks found to ingest.")
            return 1

        from ingest.stores.chroma_writer import ChromaWriter  # local import to avoid side-effects

        writer = ChromaWriter(
            persist_dir=args.chroma_dir,
            collection_name=collection_name,
            embedding_model=embedding_model,
            openai_api_key=api_key,
            openai_api_base=api_base,
            batch_size=args.chroma_batch_size,
            concurrency=args.chroma_concurrency,
        )
        tokens = writer.upsert_chunks(items)
        print(f"Embedded {len(items)} chunks (approx {tokens} tokens) into collection '{collection_name}'.")
        print(f"Chroma path: {args.chroma_dir}")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

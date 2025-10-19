"""High-level ingestion workflow orchestration."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:  # pragma: no cover - fallback when rich is unavailable
    from rich.console import Console
    from rich.table import Table
except ImportError:  # pragma: no cover
    class Console:  # type: ignore[override]
        def rule(self, message: str) -> None:
            print(f"=== {message} ===")

        def print(self, message: str) -> None:
            print(message)

    class Table:  # type: ignore[override]
        def __init__(self, show_header: bool = True, header_style: str = "") -> None:
            self.rows: list[list[str]] = []

        def add_column(self, _: str) -> None:  # noqa: D401 - no-op
            return None

        def add_row(self, *values: str) -> None:
            self.rows.append(list(values))

        def __str__(self) -> str:
            return "\n".join([" | ".join(row) for row in self.rows])

from .parsers.documents import ParsedDocument, parse_document, serialize_manifest
from .chunking import Chunk, chunk_table_rows, chunk_text
from .stores.chroma_writer import ChromaWriter
from .stores.sqlite_writer import StructuredStore

console = Console()


def _infer_company(path: Path) -> str:
    company = path.parent.name
    if not company:
        raise ValueError(f"Unable to infer company from path {path}")
    return company


def _infer_year(path: Path) -> int | None:
    match = re.search(r"(19|20)\d{2}", path.stem)
    return int(match.group(0)) if match else None


@dataclass(slots=True)
class PipelineConfig:
    input_paths: list[Path]
    output_dir: Path
    company: str | None = None
    year: int | None = None
    currency: str | None = None
    reporting_basis: str | None = None
    dry_run: bool = False
    # Updated strategy: 300–400 target, max 600, overlap 30–50
    chunk_size: int = 400
    chunk_overlap: int = 40
    embedding_model: str = "text-embedding-3-small"
    chroma_path: Path = Path("./data/chroma")
    structured_db_path: Path = Path("./data/metrics.db")
    collection_name: str = "annual-reports"
    openai_api_key: str | None = None
    openai_api_base: str | None = None
    chroma_batch_size: int = 32
    chroma_concurrency: int = 4
    append_chunks: bool = False


@dataclass(slots=True)
class IngestionResult:
    documents_indexed: int
    tables_indexed: int
    tokens_used: int
    manifest_path: Path
    chunks_path: Path


@dataclass(slots=True)
class IngestionPipeline:
    config: PipelineConfig

    def run(self) -> IngestionResult:
        label_company = self.config.company or "auto"
        label_year = self.config.year or "auto"
        console.rule(f"[bold cyan]Ingestion start[/] :: {label_company} {label_year}")

        parsed_docs = list(self._parse_documents(self.config.input_paths))

        manifest_table = Table(show_header=True, header_style="bold magenta")
        manifest_table.add_column("File")
        manifest_table.add_column("SHA256")
        manifest_table.add_column("Pages")
        manifest_table.add_column("Tables")
        manifest_table.add_column("Sections")

        for parsed in parsed_docs:
            manifest_table.add_row(
                parsed.name,
                parsed.sha256[:12],
                str(parsed.page_count),
                str(parsed.table_count),
                str(len(parsed.sections)),
            )

        console.print(manifest_table)

        manifest_path = self._write_manifest(parsed_docs)

        chunk_items, table_records = self._prepare_ingest_assets(parsed_docs)
        chunks_path = self._write_chunks_json(chunk_items)

        tokens_used = 0

        if self.config.dry_run:
            console.print("[yellow]Dry run enabled: skipping embedding and database writes.[/]")
        else:
            tokens_used = self._embed_and_persist(chunk_items, table_records)

        return IngestionResult(
            documents_indexed=len(parsed_docs),
            tables_indexed=sum(doc.table_count for doc in parsed_docs),
            tokens_used=tokens_used,
            manifest_path=manifest_path,
            chunks_path=chunks_path,
        )

    def _parse_documents(self, paths: Iterable[Path]) -> Iterable[ParsedDocument]:
        for path in paths:
            try:
                yield parse_document(path)
            except Exception as exc:  # noqa: BLE001 - log and continue for now
                console.print(f"[red]Failed to parse {path}: {exc}")

    def _write_manifest(self, parsed_docs: list[ParsedDocument]) -> Path:
        manifest_dir = self.config.output_dir / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"{(self.config.company or 'auto')}-{(self.config.year or 'auto')}.json"

        manifest_data = [parsed.model_dump() for parsed in parsed_docs]
        manifest_path.write_text(serialize_manifest(manifest_data))
        return manifest_path

    def _prepare_ingest_assets(
        self, parsed_docs: Sequence[ParsedDocument]
    ) -> tuple[list[tuple[str, Chunk]], list[dict]]:
        chunk_items: list[tuple[str, Chunk]] = []
        table_records: list[dict] = []

        for doc_index, parsed in enumerate(parsed_docs):
            company = self.config.company or _infer_company(parsed.path)
            year = self.config.year or _infer_year(parsed.path)

            base_metadata = {
                "company": company,
                "year": year,
                "document_name": parsed.name,
                "doc_type": parsed.doc_type,
                "reporting_basis": self.config.reporting_basis,
                "document_sha": parsed.sha256,
                "source_path": str(parsed.path),
            }

            text_chunks = chunk_text(
                parsed.full_text,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                base_metadata=base_metadata,
            )

            for chunk_idx, chunk in enumerate(text_chunks):
                chunk_id = f"{parsed.sha256}-text-{chunk_idx}"
                chunk.metadata.update({"chunk_id": chunk_id})
                chunk_items.append((chunk_id, chunk))

            for table_index, table in enumerate(parsed.tables):
                table_metadata = {
                    **base_metadata,
                    "table_id": table.table_id,
                    "page_range": table.page_range,
                }

                row_chunks = chunk_table_rows(table.rows or [], base_metadata=table_metadata)

                for chunk_idx, chunk in enumerate(row_chunks):
                    chunk_id = f"{parsed.sha256}-table-{table_index}-{chunk_idx}"
                    chunk.metadata.update({"chunk_id": chunk_id})
                    chunk_items.append((chunk_id, chunk))

                table_records.append(
                    {
                        "company": company,
                        "year": year,
                        "document_name": parsed.name,
                        "table_id": table.table_id,
                        "row_count": table.row_count,
                        "column_count": table.column_count,
                        "page_range": table.page_range,
                        "caption": table.caption,
                        "rows": table.rows,
                    }
                )

        return chunk_items, table_records

    def _embed_and_persist(self, chunk_items: list[tuple[str, Chunk]], table_records: list[dict]) -> int:
        console.print(f"[cyan]Preparing {len(chunk_items)} chunks for embedding.[/]")

        if not self.config.openai_api_key:
            raise RuntimeError("OpenAI API key required for embeddings. Set in PipelineConfig.openai_api_key")

        chroma_writer = ChromaWriter(
            persist_dir=self.config.chroma_path,
            collection_name=self.config.collection_name,
            embedding_model=self.config.embedding_model,
            openai_api_key=self.config.openai_api_key,
            openai_api_base=self.config.openai_api_base,
            batch_size=self.config.chroma_batch_size,
            concurrency=self.config.chroma_concurrency,
        )

        tokens_used = chroma_writer.upsert_chunks(chunk_items)

        structured_store = StructuredStore(self.config.structured_db_path)
        structured_store.bulk_upsert_table_summaries(table_records)

        console.print(
            f"[green]Embedded {len(chunk_items)} chunks (approx {tokens_used} tokens) and upserted {len(table_records)} tables.[/]"
        )

        return tokens_used

    def _write_chunks_json(self, chunk_items: list[tuple[str, Chunk]]) -> Path:
        from json import dumps, loads

        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        chunk_records: list[dict] = []

        for order, (chunk_id, chunk) in enumerate(chunk_items, start=1):
            record = {
                "chunk_id": chunk_id,
                "text": chunk.text,
                "order": order,
                **chunk.metadata,
            }
            chunk_records.append(record)

        chunks_path = output_dir / "chunks.json"
        if self.config.append_chunks and chunks_path.exists():
            try:
                existing = loads(chunks_path.read_text(encoding="utf-8"))
                if isinstance(existing, list):
                    chunk_records = existing + chunk_records
            except Exception:
                pass
        chunks_path.write_text(dumps(chunk_records, indent=2, ensure_ascii=False), encoding="utf-8")
        return chunks_path

"""High-level ingestion workflow orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from rich.console import Console
from rich.table import Table

from .parsers.documents import ParsedDocument, parse_document, serialize_manifest
from .chunking import Chunk, chunk_table_rows, chunk_text
from .stores.chroma_writer import ChromaWriter
from .stores.sqlite_writer import StructuredStore

console = Console()


@dataclass(slots=True)
class PipelineConfig:
    input_paths: list[Path]
    output_dir: Path
    company: str
    year: int
    sector: str
    currency: str | None = None
    reporting_basis: str | None = None
    dry_run: bool = False
    chunk_size: int = 1200
    chunk_overlap: int = 200
    embedding_model: str = "text-embedding-3-small"
    chroma_path: Path = Path("./data/chroma")
    structured_db_path: Path = Path("./data/metrics.db")
    collection_name: str = "annual-reports"
    openai_api_key: str | None = None


@dataclass(slots=True)
class IngestionResult:
    documents_indexed: int
    tables_indexed: int
    tokens_used: int
    manifest_path: Path


@dataclass(slots=True)
class IngestionPipeline:
    config: PipelineConfig

    def run(self) -> IngestionResult:
        console.rule(f"[bold cyan]Ingestion start[/] :: {self.config.company} {self.config.year}")

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

        tokens_used = 0

        if self.config.dry_run:
            console.print("[yellow]Dry run enabled: skipping embedding and database writes.[/]")
        else:
            tokens_used = self._embed_and_persist(parsed_docs)

        return IngestionResult(
            documents_indexed=len(parsed_docs),
            tables_indexed=sum(doc.table_count for doc in parsed_docs),
            tokens_used=tokens_used,
            manifest_path=manifest_path,
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
        manifest_path = manifest_dir / f"{self.config.company}-{self.config.year}.json"

        manifest_data = [parsed.model_dump() for parsed in parsed_docs]
        manifest_path.write_text(serialize_manifest(manifest_data))
        return manifest_path

    def _embed_and_persist(self, parsed_docs: Sequence[ParsedDocument]) -> int:
        chunk_items: list[tuple[str, Chunk]] = []
        table_records: list[dict] = []

        for doc_index, parsed in enumerate(parsed_docs):
            base_metadata = {
                "company": self.config.company,
                "year": self.config.year,
                "sector": self.config.sector,
                "document_name": parsed.name,
                "doc_type": parsed.doc_type,
                "reporting_basis": self.config.reporting_basis,
                "document_sha": parsed.sha256,
            }

            text_chunks = chunk_text(
                parsed.full_text,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                base_metadata={**base_metadata, "chunk_type": "text"},
            )

            for chunk_idx, chunk in enumerate(text_chunks):
                chunk_id = f"{parsed.sha256}-text-{chunk_idx}"
                chunk.metadata.update({"chunk_id": chunk_id})
                chunk_items.append((chunk_id, chunk))

            for table_index, table in enumerate(parsed.tables):
                table_metadata = {
                    **base_metadata,
                    "chunk_type": "table",
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
                        "company": self.config.company,
                        "year": self.config.year,
                        "document_name": parsed.name,
                        "table_id": table.table_id,
                        "row_count": table.row_count,
                        "column_count": table.column_count,
                        "page_range": table.page_range,
                        "caption": table.caption,
                        "rows": table.rows,
                    }
                )

        console.print(f"[cyan]Preparing {len(chunk_items)} chunks for embedding.[/]")

        if not self.config.openai_api_key:
            raise RuntimeError("OpenAI API key required for embeddings. Set in PipelineConfig.openai_api_key")

        chroma_writer = ChromaWriter(
            persist_dir=self.config.chroma_path,
            collection_name=self.config.collection_name,
            embedding_model=self.config.embedding_model,
            openai_api_key=self.config.openai_api_key,
        )

        tokens_used = chroma_writer.upsert_chunks(chunk_items)

        structured_store = StructuredStore(self.config.structured_db_path)
        structured_store.bulk_upsert_table_summaries(table_records)

        console.print(
            f"[green]Embedded {len(chunk_items)} chunks (approx {tokens_used} tokens) and upserted {len(table_records)} tables.[/]"
        )

        return tokens_used

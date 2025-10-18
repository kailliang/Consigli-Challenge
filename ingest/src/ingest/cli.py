"""Typer-based CLI for running ingestion pipelines."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import typer

from ingest.pipeline import IngestionPipeline, PipelineConfig

app = typer.Typer(help="Ingestion utilities for the annual report analyst")

_ENV_LOCATIONS: Tuple[Path, ...] = (Path("./backend/.env"), Path("./.env"))
_DOC_SUFFIXES = {".md", ".pdf", ".docx"}


def _load_env_files(locations: Iterable[Path]) -> None:
    """Populate os.environ from simple KEY=VALUE lines in the provided files."""

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
        if (
            path.exists()
            and path.is_file()
            and path.suffix.lower() in _DOC_SUFFIXES
            and path not in seen
        ):
            unique.append(path)
            seen.add(path)
    return unique


def _infer_year_from_name(path: Path) -> Optional[int]:
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


def _prepare_common_config_kwargs(
    *,
    output_dir: Path,
    currency: Optional[str],
    reporting_basis: Optional[str],
    dry_run: bool,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
    chroma_dir: Path,
    structured_db: Path,
    collection_name: Optional[str],
    company_hint: Optional[str],
    openai_api_key: Optional[str],
    openai_api_base: Optional[str],
    chroma_batch_size: int,
    chroma_concurrency: int,
) -> dict:
    _load_env_files(_ENV_LOCATIONS)

    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    api_base = openai_api_base or os.getenv("OPENAI_API_BASE")

    if not api_key:
        raise typer.BadParameter("OpenAI API key not provided via flag or environment.")

    kwargs: dict = {
        "output_dir": output_dir,
        "currency": currency,
        "reporting_basis": reporting_basis,
        "dry_run": dry_run,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embedding_model,
        "chroma_path": chroma_dir,
        "structured_db_path": structured_db,
        "collection_name": collection_name
        or (f"annual-reports-{company_hint.lower()}" if company_hint else "annual-reports"),
        "openai_api_key": api_key,
        "openai_api_base": api_base,
        "chroma_batch_size": chroma_batch_size,
        "chroma_concurrency": chroma_concurrency,
    }
    return kwargs


@app.command()
def run(
    input_paths: list[Path] = typer.Argument(..., help="Files or directories containing reports."),
    company: Optional[str] = typer.Option(None, "--company", "-c", help="Company short name (e.g., Tesla). Leave unset to infer from directory name."),
    year: Optional[int] = typer.Option(None, "--year", "-y", help="Reporting year. Leave unset to infer from file name."),
    output_dir: Path = typer.Option(Path("./data"), "--output-dir", "-o", help="Output directory for manifests and stores."),
    currency: Optional[str] = typer.Option(None, "--currency", help="Document currency if known (e.g., USD)."),
    reporting_basis: Optional[str] = typer.Option(None, "--reporting-basis", help="Accounting basis (IFRS, US GAAP, etc.)."),
    chroma_dir: Path = typer.Option(Path("./data/chroma"), "--chroma-dir", help="Directory for Chroma persistence."),
    structured_db: Path = typer.Option(Path("./data/metrics.db"), "--structured-db", help="Path to SQLite structured store."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Skip embeddings and database writes."),
    chunk_size: int = typer.Option(800, "--chunk-size", help="Target chunk size in tokens."),
    chunk_overlap: int = typer.Option(80, "--chunk-overlap", help="Desired token overlap between chunks."),
    embedding_model: str = typer.Option("text-embedding-3-small", "--embedding-model", help="Embedding model name."),
    openai_api_key: Optional[str] = typer.Option(None, "--openai-api-key", envvar="OPENAI_API_KEY", help="OpenAI API key (falls back to env)."),
    openai_api_base: Optional[str] = typer.Option(None, "--openai-api-base", envvar="OPENAI_API_BASE", help="Optional OpenAI API base URL."),
    chroma_batch_size: int = typer.Option(32, "--chroma-batch-size", help="Embedding batch size used for Chroma upserts."),
    chroma_concurrency: int = typer.Option(32, "--chroma-concurrency", help="Maximum concurrent embedding requests."),
    collection_name: Optional[str] = typer.Option(None, "--collection-name", help="Override the Chroma collection name."),
):
    """Run the ingestion pipeline."""

    resolved_paths = _resolve_input_paths(input_paths)
    if not resolved_paths:
        typer.secho("No files found for ingestion.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    common_kwargs = _prepare_common_config_kwargs(
        output_dir=output_dir,
        currency=currency,
        reporting_basis=reporting_basis,
        dry_run=dry_run,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        chroma_dir=chroma_dir,
        structured_db=structured_db,
        collection_name=collection_name,
        company_hint=company,
        openai_api_key=openai_api_key,
        openai_api_base=openai_api_base,
        chroma_batch_size=chroma_batch_size,
        chroma_concurrency=chroma_concurrency,
    )

    if company is not None and year is not None:
        label = f"{company} {year}"
        typer.secho(f"Starting ingestion for {label}...", fg=typer.colors.CYAN)
        pipeline = IngestionPipeline(
            config=PipelineConfig(
                input_paths=resolved_paths,
                company=company,
                year=year,
                append_chunks=False,
                **common_kwargs,
            )
        )
        result = pipeline.run()
        typer.secho("Ingestion complete", fg=typer.colors.GREEN)
        typer.echo(f"Documents indexed: {result.documents_indexed}")
        typer.echo(f"Tables indexed: {result.tables_indexed}")
        typer.echo(f"Tokens used (approx): {result.tokens_used}")
        typer.echo(f"Manifest path: {result.manifest_path}")
        typer.echo(f"Chunks path: {result.chunks_path}")
        return

    grouped = _group_documents_by_metadata(
        resolved_paths,
        company_override=company,
        year_override=year,
    )
    if not grouped:
        typer.secho("No eligible documents discovered for bulk ingestion.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    chunks_path = common_kwargs["output_dir"] / "chunks.json"
    if not dry_run and chunks_path.exists():
        chunks_path.unlink()

    total_docs = 0
    total_tables = 0
    total_tokens = 0
    last_chunks_path: Optional[Path] = None
    manifest_paths: list[Path] = []

    typer.secho(f"Bulk ingestion detected ({len(grouped)} company/year groups).", fg=typer.colors.CYAN)

    sorted_groups = sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0] or "",
            item[0][1] if item[0][1] is not None else -1,
        ),
    )

    for index, ((auto_company, auto_year), files) in enumerate(sorted_groups):
        effective_company = company or auto_company
        effective_year = year or auto_year
        display_year = effective_year if effective_year is not None else "unknown"
        typer.secho(f"-> {effective_company} {display_year} ({len(files)} files)", fg=typer.colors.BLUE)

        pipeline = IngestionPipeline(
            config=PipelineConfig(
                input_paths=files,
                company=effective_company,
                year=effective_year,
                append_chunks=index > 0,
                **common_kwargs,
            )
        )
        result = pipeline.run()
        total_docs += result.documents_indexed
        total_tables += result.tables_indexed
        total_tokens += result.tokens_used
        manifest_paths.append(result.manifest_path)
        last_chunks_path = result.chunks_path

    typer.secho("Bulk ingestion complete", fg=typer.colors.GREEN)
    typer.echo(f"Total documents indexed: {total_docs}")
    typer.echo(f"Total tables indexed: {total_tables}")
    typer.echo(f"Total tokens used (approx): {total_tokens}")
    if manifest_paths:
        typer.echo(f"Manifest paths: {', '.join(str(path) for path in manifest_paths)}")
    if last_chunks_path:
        typer.echo(f"Combined chunks path: {last_chunks_path}")


if __name__ == "__main__":  # pragma: no cover
    app()

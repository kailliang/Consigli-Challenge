"""Typer-based CLI for running ingestion pipelines."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer

from ingest.pipeline import IngestionPipeline, PipelineConfig

app = typer.Typer(help="Ingestion utilities for the annual report analyst")


def _resolve_input_paths(paths: list[Path]) -> list[Path]:
    resolved: list[Path] = []
    for path in paths:
        if path.is_dir():
            resolved.extend(sorted(p for p in path.rglob("*") if p.is_file()))
        else:
            resolved.append(path)
    unique = []
    seen = set()
    for path in resolved:
        if path.exists() and path.is_file() and path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


@app.command()
def run(
    input_paths: list[Path] = typer.Argument(..., help="Files or directories containing reports."),
    company: str = typer.Option(..., "--company", "-c", help="Company short name (e.g., Tesla)."),
    year: int = typer.Option(..., "--year", "-y", help="Reporting year."),
    sector: str = typer.Option(..., "--sector", "-s", help="Sector tag (e.g., automotive)."),
    output_dir: Path = typer.Option(Path("./data"), "--output-dir", "-o", help="Output directory for manifests and stores."),
    currency: Optional[str] = typer.Option(None, "--currency", help="Document currency if known (e.g., USD)."),
    reporting_basis: Optional[str] = typer.Option(None, "--reporting-basis", help="Accounting basis (IFRS, US GAAP, etc.)."),
    chroma_dir: Path = typer.Option(Path("./data/chroma"), "--chroma-dir", help="Directory for Chroma persistence."),
    structured_db: Path = typer.Option(Path("./data/metrics.db"), "--structured-db", help="Path to SQLite structured store."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Skip embeddings and database writes."),
    chunk_size: int = typer.Option(1200, "--chunk-size", help="Text chunk size in characters."),
    chunk_overlap: int = typer.Option(200, "--chunk-overlap", help="Text chunk overlap in characters."),
    embedding_model: str = typer.Option("text-embedding-3-small", "--embedding-model", help="Embedding model name."),
    openai_api_key: Optional[str] = typer.Option(None, "--openai-api-key", envvar="OPENAI_API_KEY", help="OpenAI API key (falls back to env)."),
):
    """Run the ingestion pipeline for one or more reports."""

    resolved_paths = _resolve_input_paths(input_paths)
    if not resolved_paths:
        typer.secho("No files found for ingestion.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if not openai_api_key:
        openai_api_key = os.getenv("OPENAI_API_KEY")

    config = PipelineConfig(
        input_paths=resolved_paths,
        output_dir=output_dir,
        company=company,
        year=year,
        sector=sector,
        currency=currency,
        reporting_basis=reporting_basis,
        dry_run=dry_run,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        chroma_path=chroma_dir,
        structured_db_path=structured_db,
        collection_name=f"annual-reports-{company.lower()}",
        openai_api_key=openai_api_key,
    )

    typer.secho(f"Starting ingestion for {company} {year}...", fg=typer.colors.CYAN)
    pipeline = IngestionPipeline(config=config)

    try:
        result = pipeline.run()
    except Exception as exc:  # pragma: no cover
        typer.secho(f"Pipeline failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.secho("Ingestion complete", fg=typer.colors.GREEN)
    typer.echo(f"Documents indexed: {result.documents_indexed}")
    typer.echo(f"Tables indexed: {result.tables_indexed}")
    typer.echo(f"Tokens used (approx): {result.tokens_used}")
    typer.echo(f"Manifest path: {result.manifest_path}")


if __name__ == "__main__":  # pragma: no cover
    app()

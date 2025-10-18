"""Quick ingestion script for Markdown annual reports."""

from __future__ import annotations

import json
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
INGEST_SRC = ROOT / "ingest" / "src"


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"Could not load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


chunking = _load_module(INGEST_SRC / "ingest" / "chunking.py", "quick_chunking")
markdown_parser = _load_module(INGEST_SRC / "ingest" / "parsers" / "markdown_parser.py", "quick_markdown_parser")

chunk_text = chunking.chunk_text  # type: ignore[attr-defined]
parse_markdown = markdown_parser.parse_markdown  # type: ignore[attr-defined]


@dataclass(slots=True)
class QuickConfig:
    input_dirs: list[Path]
    output_path: Path
    company: str | None = None
    year: int = 2023
    chunk_size: int = 800
    chunk_overlap: int = 80


def iter_markdown_files(directories: Iterable[Path]) -> Iterable[Path]:
    for directory in directories:
        for path in directory.rglob("*.md"):
            yield path


def run_ingestion(config: QuickConfig) -> Path:
    chunk_records: list[dict] = []

    for path in iter_markdown_files(config.input_dirs):
        parsed = parse_markdown(path)
        company = config.company or path.parent.name

        base_metadata = {
            "company": company,
            "year": config.year,
            "document_name": path.name,
            "source_path": str(path),
            "doc_type": "markdown",
        }

        text_chunks = chunk_text(
            parsed.full_text,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            base_metadata=base_metadata,
        )

        for index, chunk in enumerate(text_chunks, start=1):
            chunk_id = f"{path.stem}-chunk-{index}"
            record = {
                "chunk_id": chunk_id,
                "order": index,
                "text": chunk.text,
                **chunk.metadata,
            }
            chunk_records.append(record)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(chunk_records, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Ingested {len(chunk_records)} chunks -> {config.output_path}")
    return config.output_path


if __name__ == "__main__":
    data_dirs = [
        ROOT / "data" / "BMW",
        ROOT / "data" / "Tesla",
        ROOT / "data" / "Ford",
        ROOT / "data" / "General",
    ]
    output = ROOT / "data" / "chunks.json"
    run_ingestion(QuickConfig(input_dirs=data_dirs, output_path=output))

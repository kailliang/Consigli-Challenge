# Ingestion Pipeline (WIP)

Current capabilities:

- Layout-aware document manifesting via `pdfplumber` (PDF), `python-docx` (DOCX), and lightweight Markdown parsing.
- Table summaries (row/column counts, per-page identifiers) and heading extraction heuristics for section trees.
- Deterministic file hashing and JSON manifest generation for reproducibility.
- Sentence-level text chunking: target 600–800 tokens, hard max 1000, with 60–100 token overlap.
- Tables are not split; each table is serialized as a single chunk to preserve units/currency context.
- Character-based chunking for text and table rows, followed by embedding upserts into Chroma and table summary persistence in SQLite.
- Typer CLI (`ingest-cli run ...`) for batch-ingesting reports directly from the command line.

Planned enhancements:

- OCR microservice for figures and scanned documents.
- Embedding + Chroma upsert workers with batch orchestration.
- Structured fact loader targeting SQLite (`data/metrics.db`).
- Typer CLI commands for bulk ingestion and status reporting.

Chunking parameters (defaults):
- `chunk_size` (target_max): 800
- `chunk_target_min` (derived): 600
- `chunk_max` (derived): 1000
- `chunk_overlap`: 80

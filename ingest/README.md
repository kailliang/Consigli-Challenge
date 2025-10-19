# Ingestion Pipeline (WIP)

Current capabilities:

- Layout-aware document manifesting via `pdfplumber` (PDF), `python-docx` (DOCX), and lightweight Markdown parsing.
- Table summaries (row/column counts, per-page identifiers) and heading extraction heuristics for section trees.
- Deterministic file hashing and JSON manifest generation for reproducibility.
- Sentence-level text chunking: target 300–400 tokens, hard max 600, with 30–50 token overlap.
- Tables are not split; each table is serialized as a single chunk to preserve units/currency context.
- Character-based chunking for text and table rows, followed by embedding upserts into Chroma and table summary persistence in SQLite.
- Unified ingestion script under `scripts/quick_ingest_markdown.py` for chunking, embeddings (Chroma), and SQLite persistence.
  - Modes:
    - Chunk only: `python scripts/quick_ingest_markdown.py --mode chunk`
    - Ingest only (from chunks.json): `python scripts/quick_ingest_markdown.py --mode ingest --chunks-path data/chunks.json`
    - Both (default): `python scripts/quick_ingest_markdown.py --mode both`
  - Set `OPENAI_API_KEY` via environment or `backend/.env`/`.env` for ingestion.

Planned enhancements:

- OCR microservice for figures and scanned documents.
- Embedding + Chroma upsert workers with batch orchestration.
- Structured fact loader targeting SQLite (`data/metrics.db`).
- Typer CLI commands for bulk ingestion and status reporting.

Chunking parameters (defaults):
- `chunk_size` (target_max): 800
- `chunk_target_min` (derived): 600
- `chunk_max` (derived): 600
- `chunk_overlap`: 40

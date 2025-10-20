# Ingestion Pipeline (WIP)

Current capabilities:

- High-level goal: convert Markdown/PDF/DOCX annual reports into searchable text/table chunks, embeddings, and structured summaries for downstream RAG.
- Table summaries (row/column counts, per-page identifiers) and heading extraction heuristics for section trees.
- Deterministic file hashing and JSON manifest generation for reproducibility.
- Sentence-level text chunking: target 300–400 tokens, hard max 600, with 30–50 token overlap.
- Tables are not split; each table is serialized as a single chunk to preserve units/currency context.
- Character-based chunking for text and table rows, followed by embedding upserts into Chroma and table summary persistence in SQLite.
- Unified ingestion script under `scripts/quick_ingest_markdown.py` for chunking, embeddings (Chroma), and SQLite persistence.
  - Modes:
    - Chunk only: `python scripts/quick_ingest_markdown.py --mode chunk`
    - Ingest only (from chunks.json): `python scripts/quick_ingest_markdown.py --mode ingest --chunks-path data/chunks.json`
    
    `python scripts/quick_ingest_markdown.py --embedding-model text-embedding-3-large --mode ingest --chunks-path data/chunks.json`
    
    - Both (default): `python scripts/quick_ingest_markdown.py --mode both`
  - Set `OPENAI_API_KEY` via environment or `backend/.env`/`.env` for ingestion.

### Converting PDFs/DOCX to Markdown

Use `scripts/pdf_to_markdown.py` to batch convert source documents before ingestion:

```bash
python scripts/pdf_to_markdown.py \
  --input-dir ingest/data \
  --output-dir ingest/data_markdown \
  --include-metadata \
  --timeout 120
```

- Defaults scan `ingest/data` recursively and write `.md` files alongside the originals.
- Provide your API key via `--api-key` or the `AGENTIC_DOCUMENT_ANALYSIS_API_KEY` environment variable.
- Flags: `--include-metadata` embeds extra metadata (like detected headings, page numbers, section info), 
  `--include-marginalia`, 
  `--disable-rotation-detection` 
  control Landing AI options.

### Chunking and Ingestion

Most common invocation (parse + embed + persist all sample data):

```bash
python scripts/quick_ingest_markdown.py
```

Target a specific file or directory:

```bash
python scripts/quick_ingest_markdown.py data/Ford/Ford_Annual_Report_2021.md --mode both
```

`OPENAI_API_KEY` (and optional `OPENAI_API_BASE`) are read automatically from `.env` files. Override table-summary settings if needed:

```bash
python scripts/quick_ingest_markdown.py \
  --mode both \
  --table-summary-model gpt-5-mini \
```

Chunking parameters (defaults):
- `chunk_size` (target_max): 400
- `chunk_target_min` (derived): 300
- `chunk_max` (derived): 500
- `chunk_overlap`: 40

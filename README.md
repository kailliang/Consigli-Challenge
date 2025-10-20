# LLM-Powered Annual Report Analyst

Prototype workspace for a multi-modal retrieval-augmented generation (RAG) system that analyzes automotive annual reports. The backend is scaffolded with FastAPI + LangChain; the frontend is a modern React + Vite chat shell ready for streaming integration. 

## Application Overview

- **Ingestion pipeline** (Python, `ingest/`) parses Markdown/PDF/DOCX annual reports, builds table/context summaries with OpenAI, embeds text/table chunks via `text-embedding-3-large`, and persists data to Chroma + SQLite.
- **Backend service** (`backend/`) exposes a chat-oriented API that orchestrates retrieval, conversation memory, query expansion, reranking, and answer generation using configurable OpenAI models.
- **Frontend client** (`frontend/`) provides a chat UI for analysts to ask questions, review responses, and inspect supporting citations.

With ingestion completed, the application can answer revenue/profit and trend questions across recent reports for Tesla, BMW, Ford, and other supported companies.

## RAG Workflow

1. **User prompt + memory** – Each message is combined with short conversation memory so pronouns and follow-ups inherit context.
2. **Retrieval gating** – A lightweight LLM decides if Chroma lookup is required; small talk bypasses retrieval.
3. **Query expansion** – An LLM (with current time tool access) emits a rewritten, self-contained query plus diverse variations to cover alternate phrasing and fiscal periods.
4. **Candidate retrieval** – Expanded queries run against the vector store; results are deduplicated by chunk identifier.
5. **Metadata-aware reranking** – Each chunk is re-embedded, scored against the rewritten query, boosted for matching years, and penalized for cross-company leakage.
6. **Answer generation** – The top-ranked context is fed to the main LLM, which produces a cited answer returned to the frontend.

LangSmith tracing is enabled to capture chunk scores, query expansions, and decision logs for debugging.

## Repository Layout

- `backend/`: FastAPI service, LangChain orchestration, vector/structured store clients.
- `frontend/`: React + TypeScript chat UI scaffold powered by Vite and TailwindCSS.
- `ingest/`: Placeholder for document ingestion pipelines (parsing, embeddings, OCR workers).
- `infra/`: Deployment manifests, container definitions, and infrastructure-as-code (upcoming).
- `docs/`: Supplemental architecture notes and design references.

## Getting Started

> Tooling versions: Python 3.13, Node.js 20+, uv or pipx for backend packaging, pnpm/npm for frontend tooling.

### Backend

1. Create and activate a virtual environment.
2. Install dependencies: `uv pip install -r backend/pyproject.toml` *(or use `pip install -e .[dev]` from inside `backend/` once the environment is active).* 
3. Copy `backend/.env.example` to `.env` and populate required keys (OpenAI, LangSmith, storage paths).
4. Run the API: `uvicorn app.main:app --reload --app-dir backend/app`.
   - Streaming endpoint: `WS /v1/query/ws`

### Ingestion

- Unified script (chunks + embeddings + persistence): `python scripts/quick_ingest_markdown.py`
  - Dry run (chunks only): `python scripts/quick_ingest_markdown.py --dry-run`
  - Full run: ensure `OPENAI_API_KEY` is set in your shell or in `backend/.env`/`.env`
  - Inputs default to `data/BMW data/Tesla data/Ford data/General`; you can pass specific files/dirs as args
  - Useful flags: `--company`, `--year`, `--output-dir`, `--chroma-dir`, `--structured-db`, `--embedding-model`, `--chroma-batch-size`, `--chroma-concurrency`, `--collection-name`, `--chunk-size`, `--chunk-overlap`, `--openai-api-base`, `--openai-api-key`

 - Modes (`--mode`):
   - Chunk only: `python scripts/quick_ingest_markdown.py --mode chunk`
   - Ingest only (from an existing chunks.json): `python scripts/quick_ingest_markdown.py --mode ingest --chunks-path data/chunks.json`
   - Both (default): `python scripts/quick_ingest_markdown.py --mode both`
   - Notes:
     - `--chunks-path` defaults to `<output-dir>/chunks.json`.
     - For `ingest`/`both`, set `OPENAI_API_KEY` (env or `--openai-api-key`).

### Frontend

1. Navigate to `frontend/` and install dependencies (`pnpm install` recommended).
2. Copy `.env.example` to `.env.local` and adjust `VITE_API_BASE_URL` if needed.
3. Start the dev server: `pnpm dev`.

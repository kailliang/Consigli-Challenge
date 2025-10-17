# LLM-Powered Annual Report Analyst

Prototype workspace for a multi-modal retrieval-augmented generation (RAG) system that analyzes automotive annual reports. The backend is scaffolded with FastAPI + LangChain; the frontend is a modern React + Vite chat shell ready for streaming integration. This repository follows the implementation path described in `guide.md`.

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
   - JSON endpoint: `POST /v1/query`
   - Streaming endpoint: `POST /v1/query/stream` (SSE)

### Frontend

1. Navigate to `frontend/` and install dependencies (`pnpm install` recommended).
2. Copy `.env.example` to `.env.local` and adjust `VITE_API_BASE_URL` if needed.
3. Start the dev server: `pnpm dev`.

## Next Steps

1. **Ingestion pipeline**: add OCR/table normalization, better unit parsing, and CLI orchestration for large batches.
2. **Vector + structured stores**: expose retrieval clients, add migrations, and keep Chroma/SQLite schemas in sync across environments.
3. **RAG orchestration**: replace the placeholder generator with LangChain RetrievalQA, numeric reasoning, and LangSmith tracing.
4. **Frontend integration**: upgrade to streamed responses, richer citation views, ingestion controls, and comparison visualizations.
5. **Validation & observability**: harden numeric checks against structured datasets, add pytest coverage (API + ingestion), contract fixtures, and frontend e2e smoke tests.

Refer to `guide.md` for detailed requirements, success metrics, and long-term roadmap.

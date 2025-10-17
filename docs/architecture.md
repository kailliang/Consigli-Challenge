# Architecture Notes

This document tracks early decisions while the system is under construction. Refer to `guide.md` for full requirements.

## Backend

- **FastAPI + LangChain**: `app/main.py` wires a versioned API router (`/v1`) and prepares the ground for LangSmith tracing.
- **Configuration**: Pydantic settings (`app/core/config.py`) centralize model names, store locations, and feature flags (memory windows, tracing toggles, etc.).
- **Logging**: Structlog-based JSON logging (`app/core/logging.py`) is ready for shipping to centralized observability platforms.
- **Data access**: Chroma client helper (`app/services/vectorstore.py`) and async SQLAlchemy engine (`app/core/db.py`) prepare persistence layers for embeddings and structured facts.
- **Query API**: `/v1/query` route now runs a LangChain-based RetrievalQA pipeline (`app/services/rag.py`) backed by Chroma + structured table lookups.

### Upcoming

- Harden the RAG prompt, add numeric consistency checks against structured tables, and expand comparison/time-series pathways.
- Session memory store aligning with the 10-turn window + rolling summary requirement.
- Build comparison/time-series services that combine vector + structured store data.

## Frontend

- **React + Vite**: Chat shell with Tailwind styling, query caching, and placeholder interactions (`src/components/chat`).
- **State Management**: `useChatSession` now calls the `/v1/query` backend and surfaces citation stubs in the UI.

### Upcoming

- Real-time streaming responses via Server-Sent Events or WebSockets.
- Citation drawer with precise page/table references.
- Upload + ingest control surface for analysts.

## Ingestion & Infra

- `ingest/` houses the Typer-ready pipeline with pdfplumber/python-docx parsing, heading summaries, manifest generation, char-level chunking, and direct upserts into Chroma + SQLite; `infra/` remains reserved for deployment tooling.
- Next: upgrade parsers with OCR/table normalization, add worker containers, and orchestrate pipeline steps via distributed queue.
- Plan to introduce Dockerfiles, Helm charts, and CI workflows once the pipelines stabilize.

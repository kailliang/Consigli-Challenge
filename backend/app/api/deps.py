"""Shared FastAPI dependencies."""

from __future__ import annotations

from fastapi import Depends

from app.core.config import AppSettings, get_settings
from app.core.db import get_session
from app.models.chat import QueryRequest
from app.services import rag, vectorstore


def get_app_settings() -> AppSettings:
    """Expose application settings as a dependency."""

    return get_settings()


async def get_db_session():
    """Provide an async SQLAlchemy session."""

    async with get_session() as session:
        yield session


async def get_vector_client():
    """Provide a Chroma client instance."""

    return vectorstore.get_vector_client()


def get_rag_context(
    settings: AppSettings = Depends(get_app_settings),
) -> rag.RAGContext:
    """Construct or retrieve a cached RAG context."""

    return rag.get_rag_context(settings)


def validate_query(request: QueryRequest) -> QueryRequest:
    """Ensure prompt formatting and basic defensive checks."""

    prompt = request.prompt.strip()
    if not prompt:
        raise ValueError("Prompt must not be empty.")
    return QueryRequest(prompt=prompt, session_id=request.session_id)

"""Vector store clients and helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chromadb import PersistentClient
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_client: PersistentClient | None = None


def get_vector_client() -> PersistentClient:
    """Return a singleton Chroma persistent client."""

    global _client
    if _client is None:
        settings = get_settings()
        persist_dir = Path(settings.chroma_persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        logger.info("vectorstore.init", persist_dir=str(persist_dir))
        _client = PersistentClient(path=str(persist_dir))
    return _client


def get_collection(
    name: str,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Fetch or create a named collection."""

    client = get_vector_client()
    collection = client.get_or_create_collection(name=name, metadata=metadata)
    return collection


def get_langchain_vectorstore(collection_name: str | None = None) -> Chroma:
    """Return a LangChain-ready Chroma vector store."""

    settings = get_settings()
    collection = collection_name or settings.vector_collection_name

    if not settings.openai_api_key:
        raise RuntimeError("OpenAI API key required for vector store embeddings.")

    embeddings = OpenAIEmbeddings(
        model=settings.embeddings_model,
        openai_api_key=settings.openai_api_key,
        openai_api_base=settings.openai_api_base,
    )

    vectorstore = Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )
    return vectorstore

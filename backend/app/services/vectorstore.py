"""Vector store clients and helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from chromadb import PersistentClient
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from app.core.config import AppSettings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_client: PersistentClient | None = None
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_persist_dir(raw_path: str | Path) -> Path:
    """Convert configured paths to absolute locations under the project root."""

    path = Path(raw_path)
    if path.is_absolute():
        return path

    resolved = (_REPO_ROOT / path).resolve()
    return resolved


def get_vector_client() -> PersistentClient:
    """Return a singleton Chroma persistent client."""

    global _client
    if _client is None:
        settings = get_settings()
        persist_dir = _resolve_persist_dir(settings.chroma_persist_dir)
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


def create_embeddings(settings: AppSettings | None = None) -> OpenAIEmbeddings:
    """Instantiate an OpenAI embeddings client using application settings."""

    settings = settings or get_settings()

    if not settings.openai_api_key:
        raise RuntimeError("OpenAI API key required for vector store embeddings.")

    return OpenAIEmbeddings(
        model=settings.embeddings_model,
        openai_api_key=settings.openai_api_key,
        openai_api_base=settings.openai_api_base,
    )


def get_langchain_vectorstore(
    collection_name: str | None = None,
    *,
    settings: AppSettings | None = None,
    embeddings: OpenAIEmbeddings | None = None,
) -> Chroma:
    """Return a LangChain-ready Chroma vector store."""

    settings = settings or get_settings()
    collection = collection_name or settings.vector_collection_name
    embeddings = embeddings or create_embeddings(settings)

    persist_dir = _resolve_persist_dir(settings.chroma_persist_dir)

    vectorstore = Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )
    return vectorstore

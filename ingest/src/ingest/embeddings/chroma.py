"""Embedding and vector store utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from chromadb import PersistentClient

try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:  # pragma: no cover - optional dependency during scaffolding
    OpenAIEmbeddings = None  # type: ignore


def embed_and_upsert(
    texts: Iterable[str],
    *,
    collection_name: str,
    client: PersistentClient,
    metadata: list[dict] | None = None,
) -> None:
    """Placeholder for embedding and upsert workflow."""

    if OpenAIEmbeddings is None:
        raise RuntimeError("langchain-openai must be installed to run embeddings")

    # TODO: implement chunking, metadata alignment, and batch upsert.
    _ = list(texts)
    return None

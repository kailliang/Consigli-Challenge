"""Chroma vector store writer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from chromadb import PersistentClient

try:
    from langchain_openai import OpenAIEmbeddings
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("langchain-openai must be installed for embeddings") from exc

from ..chunking import Chunk


@dataclass(slots=True)
class ChromaWriter:
    persist_dir: Path
    collection_name: str
    embedding_model: str
    openai_api_key: str | None = None

    def __post_init__(self) -> None:
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = PersistentClient(path=str(self.persist_dir))
        self._collection = self._client.get_or_create_collection(name=self.collection_name)
        self._embeddings = OpenAIEmbeddings(model=self.embedding_model, openai_api_key=self.openai_api_key)

    def upsert_chunks(self, chunk_items: Iterable[tuple[str, Chunk]]) -> int:
        chunk_list = list(chunk_items)
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for chunk_id, chunk in chunk_list:
            ids.append(chunk_id)
            documents.append(chunk.content)
            metadatas.append({**chunk.metadata, "token_count": chunk.token_count})

        if not documents:
            return 0

        embeddings = self._embeddings.embed_documents(documents)
        self._collection.upsert(ids=ids, metadatas=metadatas, documents=documents, embeddings=embeddings)
        return sum(chunk.token_count for _, chunk in chunk_list)

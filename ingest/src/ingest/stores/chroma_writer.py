from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence, Tuple

from chromadb import PersistentClient
from openai import AsyncOpenAI

from ..chunking import Chunk


def _batched(seq: Sequence, size: int) -> Iterator[tuple[int, int]]:
    """Yield start/end indices for slicing in fixed-size batches."""

    if size <= 0:
        raise ValueError("Batch size must be positive")
    start = 0
    length = len(seq)
    while start < length:
        end = min(start + size, length)
        yield start, end
        start = end


@dataclass(slots=True)
class ChromaWriter:
    persist_dir: Path
    collection_name: str
    embedding_model: str
    openai_api_key: str
    openai_api_base: str | None = None
    batch_size: int = 8
    concurrency: int = 32
    _client: PersistentClient = field(init=False)
    _collection: any = field(init=False)
    _openai_kwargs: dict[str, str] = field(init=False)

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.concurrency <= 0:
            raise ValueError("concurrency must be positive")

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = PersistentClient(path=str(self.persist_dir))
        self._collection = self._client.get_or_create_collection(name=self.collection_name)

        self._openai_kwargs = {"api_key": self.openai_api_key}
        if self.openai_api_base:
            self._openai_kwargs["base_url"] = self.openai_api_base

    def upsert_chunks(self, items: Iterable[Tuple[str, Chunk]]) -> int:
        """Generate embeddings and upsert them into Chroma."""

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for chunk_id, chunk in items:
            text = chunk.text.strip()
            if not text:
                continue
            ids.append(str(chunk_id))
            documents.append(text)
            metadata = dict(chunk.metadata)
            metadata.setdefault("chunk_id", str(chunk_id))
            metadatas.append(metadata)

        if not ids:
            return 0

        batches = [
            (ids[start:end], documents[start:end], metadatas[start:end])
            for start, end in _batched(ids, self.batch_size)
        ]

        embeddings_results = asyncio.run(self._embed_batches_async(batches))

        total_tokens = 0
        for batch_ids, batch_docs, batch_meta, embeddings, tokens in embeddings_results:
            total_tokens += tokens
            self._collection.upsert(
                ids=batch_ids,
                embeddings=embeddings,
                documents=batch_docs,
                metadatas=batch_meta,
            )

        return total_tokens

    async def _embed_batches_async(
        self,
        batches: list[tuple[list[str], list[str], list[dict]]],
    ) -> list[tuple[list[str], list[str], list[dict], list[list[float]], int]]:
        client = AsyncOpenAI(**self._openai_kwargs)
        semaphore = asyncio.Semaphore(self.concurrency)

        async def embed_batch(
            batch_ids: list[str],
            batch_docs: list[str],
            batch_meta: list[dict],
        ) -> tuple[list[str], list[str], list[dict], list[list[float]], int]:
            async with semaphore:
                response = await client.embeddings.create(
                    model=self.embedding_model,
                    input=batch_docs,
                )
            embeddings = [entry.embedding for entry in response.data]
            usage = getattr(response, "usage", None)
            token_usage = int(getattr(usage, "total_tokens", 0) or 0) if usage is not None else 0
            return batch_ids, batch_docs, batch_meta, embeddings, token_usage

        tasks = [embed_batch(batch_ids, batch_docs, batch_meta) for batch_ids, batch_docs, batch_meta in batches]
        results = await asyncio.gather(*tasks)
        await client.close()
        return list(results)

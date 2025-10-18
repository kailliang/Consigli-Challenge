from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

from ..chunking import Chunk


@dataclass(slots=True)
class ChromaWriter:
    persist_dir: Path
    collection_name: str
    embedding_model: str
    openai_api_key: str

    def upsert_chunks(self, items: Iterable[Tuple[str, Chunk]]) -> int:
        """Quick-and-dirty placeholder: return approximate token usage.

        Replace with real embeddings + Chroma upsert later.
        """

        total_tokens = 0
        for _id, chunk in items:
            total_tokens += int(chunk.metadata.get("token_estimate", len(chunk.text) // 4))
        return total_tokens


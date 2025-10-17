"""Chunking utilities for text and table content."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(slots=True)
class Chunk:
    content: str
    metadata: dict
    token_count: int


def chunk_text(
    text: str,
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    base_metadata: dict | None = None,
) -> list[Chunk]:
    """Chunk text by characters as a proxy for tokens."""

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    start = 0
    length = len(text)
    chunks: list[Chunk] = []

    while start < length:
        end = min(start + chunk_size, length)
        chunk_content = text[start:end]
        metadata = {**(base_metadata or {}), "char_start": start, "char_end": end}
        chunks.append(Chunk(content=chunk_content, metadata=metadata, token_count=len(chunk_content.split())))

        if end >= length:
            break
        start = max(end - chunk_overlap, 0)

    return chunks


def chunk_table_rows(
    rows: Iterable[dict],
    *,
    max_rows: int = 20,
    base_metadata: dict | None = None,
) -> list[Chunk]:
    combined_rows: list[str] = []
    chunks: list[Chunk] = []

    for row in rows:
        row_repr = " | ".join(f"{k}: {v}" for k, v in row.items())
        combined_rows.append(row_repr)

        if len(combined_rows) >= max_rows:
            content = "\n".join(combined_rows)
            chunks.append(
                Chunk(
                    content=content,
                    metadata={**(base_metadata or {}), "rows": len(combined_rows)},
                    token_count=len(content.split()),
                )
            )
            combined_rows = []

    if combined_rows:
        content = "\n".join(combined_rows)
        chunks.append(
            Chunk(
                content=content,
                metadata={**(base_metadata or {}), "rows": len(combined_rows)},
                token_count=len(content.split()),
            )
        )

    return chunks

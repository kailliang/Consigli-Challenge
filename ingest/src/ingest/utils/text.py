"""Text utilities for chunking."""

from __future__ import annotations

from typing import Iterable


def chunk_text(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> Iterable[str]:
    """Yield character-based chunks with overlap."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    length = len(text)
    if length == 0:
        return

    start = 0
    while start < length:
        end = min(length, start + chunk_size)
        yield text[start:end].strip()
        if end >= length:
            break
        start = max(0, end - chunk_overlap)

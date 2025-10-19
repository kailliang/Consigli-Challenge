from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(slots=True)
class Chunk:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


_ANCHOR_ID_RE = re.compile(
    r"<a\s+id=['\"][^'\"]+['\"]></a>",
    re.IGNORECASE,
)

_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[\.!?;])\s+|(?<=\n\n)\n*",
    re.VERBOSE,
)


def _estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p and p.strip()]
    return parts if parts else ([text.strip()] if text.strip() else [])


def chunk_text(
    text: str,
    *,
    chunk_size: int = 400,
    chunk_overlap: int = 40,
    base_metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Chunk narrative text.

    Strategy:
    - Target 300–400 tokens (chunk_size is treated as target_max)
    - Hard max 600 tokens
    - Overlap 30–50 tokens (sentence-level)
    """

    target_max = chunk_size
    target_min = int(0.75 * target_max)
    max_tokens = 600
    overlap = chunk_overlap

    cleaned = _ANCHOR_ID_RE.sub("", text)

    sentences = _split_sentences(cleaned)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    start = 0
    meta = base_metadata or {}

    while start < len(sentences):
        token_count = 0
        end = start

        while end < len(sentences):
            candidate = " ".join(sentences[start : end + 1])
            tokens = _estimate_tokens(candidate)
            if tokens > max_tokens:
                if end == start:
                    piece = sentences[start][: max_tokens * 4]
                    chunks.append(Chunk(text=piece.strip(), metadata={**meta, "token_estimate": _estimate_tokens(piece)}))
                    start += 1
                else:
                    prev_text = " ".join(sentences[start:end])
                    chunks.append(Chunk(text=prev_text.strip(), metadata={**meta, "token_estimate": _estimate_tokens(prev_text)}))
                    start = end
                break

            if target_min <= tokens <= target_max:
                chunks.append(Chunk(text=candidate.strip(), metadata={**meta, "token_estimate": tokens}))
                # Overlap by sentences to approximately reach desired tokens
                start = _advance_with_overlap(sentences, start, end, overlap)
                break

            end += 1
        else:
            remainder = " ".join(sentences[start:])
            chunks.append(Chunk(text=remainder.strip(), metadata={**meta, "token_estimate": _estimate_tokens(remainder)}))
            start = len(sentences)

    return chunks


def _advance_with_overlap(sentences: list[str], start: int, end: int, overlap_tokens: int) -> int:
    keep_tokens = 0
    idx = end
    while idx >= start:
        keep_tokens += _estimate_tokens(sentences[idx])
        if keep_tokens >= overlap_tokens:
            break
        idx -= 1
    return max(start + 1, idx + 1)


def chunk_table_rows(rows: list[dict[str, Any]], *, base_metadata: dict[str, Any] | None = None) -> list[Chunk]:
    """Serialize an entire table as a single chunk (no splitting)."""

    meta = base_metadata or {}
    table_id = meta.get("table_id", "table")
    company = meta.get("company")
    year = meta.get("year")
    document_name = meta.get("document_name")
    page_range = meta.get("page_range")
    table_context = meta.get("table_context")
    caption = meta.get("caption")

    header_keys: list[str] = list(rows[0].keys()) if rows else []
    preview_rows = rows  # assume tables are not very large per requirement
    header_parts = [f"Table {table_id}"]
    if company:
        header_parts.append(f"company={company}")
    if year is not None:
        header_parts.append(f"year={year}")
    if document_name:
        header_parts.append(f"source={document_name}")
    if page_range:
        header_parts.append(f"page={page_range}")

    lines = [" | ".join(header_parts)]

    if table_context:
        lines.append(f"context: {table_context}")
    if caption:
        lines.append(f"caption: {caption}")

    lines.append(f"Columns: {', '.join(header_keys)} | Rows: {len(rows)}")
    lines.extend(", ".join(f"{k}={v}" for k, v in row.items()) for row in preview_rows)

    text = "\n".join(lines)
    return [Chunk(text=text, metadata={**meta, "token_estimate": _estimate_tokens(text)})]

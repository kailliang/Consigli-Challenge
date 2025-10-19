"""Sentence-level text chunking with token estimates.

Implements the updated strategy:
- Target chunk size: 300–400 tokens; hard max 600 tokens
- Overlap: 30–50 tokens (sentence-level)
- Do not split tables (handled by caller)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable


# Simple token estimator: ~4 characters/token is a common heuristic.
def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[\.!?;])\s+|(?<=\n\n)\n*",
    re.VERBOSE,
)


def split_sentences(text: str) -> list[str]:
    """Split text by sentence-ish boundaries including Chinese/English punctuation.

    Keeps punctuation with the sentence and trims whitespace.
    """

    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p and p.strip()]
    return parts if parts else ([text.strip()] if text.strip() else [])


@dataclass(slots=True)
class ChunkConfig:
    target_min_tokens: int = 300
    target_max_tokens: int = 400
    max_tokens: int = 600
    overlap_tokens: int = 40  # 30–50


@dataclass(slots=True)
class TextChunk:
    text: str
    token_estimate: int
    start_sentence: int
    end_sentence: int


def chunk_text(text: str, *, config: ChunkConfig | None = None) -> list[TextChunk]:
    cfg = config or ChunkConfig()
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[TextChunk] = []
    start = 0

    while start < len(sentences):
        token_count = 0
        end = start

        while end < len(sentences):
            candidate = " ".join(sentences[start : end + 1])
            tokens = estimate_tokens(candidate)
            if tokens > cfg.max_tokens:
                if end == start:
                    # Single sentence too long: hard cut by characters to respect max.
                    piece = sentences[start][: cfg.max_tokens * 4]
                    chunks.append(
                        TextChunk(text=piece.strip(), token_estimate=estimate_tokens(piece), start_sentence=start, end_sentence=start)
                    )
                    start += 1
                else:
                    # finalize previous end
                    prev_text = " ".join(sentences[start:end])
                    chunks.append(
                        TextChunk(text=prev_text.strip(), token_estimate=estimate_tokens(prev_text), start_sentence=start, end_sentence=end - 1)
                    )
                    start = end  # overlap handled below
                break

            if tokens >= cfg.target_min_tokens and (tokens <= cfg.target_max_tokens):
                chunks.append(
                    TextChunk(text=candidate.strip(), token_estimate=tokens, start_sentence=start, end_sentence=end)
                )
                # advance start with overlap (sentence-level)
                start = _advance_with_overlap(sentences, start, end, cfg.overlap_tokens)
                break

            end += 1

        else:
            # Exhausted sentences; flush remainder even if below min
            remainder = " ".join(sentences[start:])
            chunks.append(
                TextChunk(text=remainder.strip(), token_estimate=estimate_tokens(remainder), start_sentence=start, end_sentence=len(sentences) - 1)
            )
            start = len(sentences)

    return chunks


def _advance_with_overlap(sentences: list[str], start: int, end: int, overlap_tokens: int) -> int:
    # Determine how many sentences from the tail to keep as overlap by token estimate.
    keep_tokens = 0
    idx = end
    while idx >= start:
        keep_tokens += estimate_tokens(sentences[idx])
        if keep_tokens >= overlap_tokens:
            break
        idx -= 1

    next_start = max(start, idx)
    return next_start

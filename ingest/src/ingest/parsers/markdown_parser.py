"""Markdown parsing utilities."""

from __future__ import annotations

import re
from bisect import bisect_right
from html import unescape
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence, Tuple


@dataclass(slots=True)
class MarkdownParseResult:
    page_count: int
    tables: list[dict[str, Any]]
    sections: list[dict[str, Any]]
    full_text: str


def parse_markdown(path: Path) -> MarkdownParseResult:
    text = path.read_text(encoding="utf-8")
    segments = text.splitlines(keepends=True)
    lines = [segment.rstrip("\r\n") for segment in segments]

    markdown_tables, table_index, markdown_ranges = _extract_markdown_tables(lines, path)
    html_tables, table_index, html_spans = _extract_html_tables(text, path, start_index=table_index)
    tables = markdown_tables + html_tables

    sections = _extract_sections(lines)

    # Markdown files lack pages; treat entire file as one logical page.
    page_count = max(1, len(sections))

    line_offsets = _compute_line_offsets(segments)
    html_line_ranges = _convert_spans_to_line_ranges(html_spans, line_offsets, len(lines))

    lines_to_skip: set[int] = set()
    for start, end in markdown_ranges:
        lines_to_skip.update(range(start, end + 1))
    for start, end in html_line_ranges:
        lines_to_skip.update(range(start, end + 1))

    sanitized_segments = [
        segment for idx, segment in enumerate(segments) if idx not in lines_to_skip
    ]
    sanitized_text = "".join(sanitized_segments)

    return MarkdownParseResult(
        page_count=page_count,
        tables=tables,
        sections=sections,
        full_text=sanitized_text,
    )


def _extract_sections(lines: list[str]) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            title = stripped.lstrip("#").strip() or "Untitled"
            sections.append({"level": level, "title": title})
        elif stripped.isupper() and len(stripped) <= 80:
            sections.append({"level": 1, "title": stripped.title()})

    if not sections and lines:
        sections.append({"level": 1, "title": lines[0].strip() or "Document"})

    return sections


def _extract_markdown_tables(
    lines: list[str],
    path: Path,
    *,
    start_index: int = 0,
) -> Tuple[list[dict[str, Any]], int, list[tuple[int, int]]]:
    tables: list[dict[str, Any]] = []
    buffer: list[str] = []
    in_table = False
    table_index = start_index
    table_ranges: list[tuple[int, int]] = []
    buffer_start: int | None = None
    buffer_end: int | None = None

    def flush_buffer() -> None:
        nonlocal buffer, in_table, table_index, buffer_start, buffer_end
        if not buffer or buffer_start is None:
            buffer = []
            buffer_start = None
            buffer_end = None
            in_table = False
            return
        header_line = buffer[0]
        if "|" not in header_line:
            buffer = []
            buffer_start = None
            buffer_end = None
            in_table = False
            return
        headers = [cell.strip() or f"col_{idx}" for idx, cell in enumerate(header_line.split("|")) if cell.strip()]
        if not headers:
            buffer = []
            buffer_start = None
            buffer_end = None
            in_table = False
            return
        data_rows = [row for row in buffer[2:] if row.strip()]
        structured_rows: list[dict[str, str]] = []
        for row in data_rows:
            values = [cell.strip() for cell in row.split("|") if cell.strip()]
            structured_rows.append(
                {
                    headers[idx] if idx < len(headers) else f"col_{idx}": values[idx] if idx < len(values) else ""
                    for idx in range(len(headers))
                }
            )
        table_index += 1
        context, surrounding_sentences = _select_context_sentence(
            lines=lines,
            table_start=buffer_start or 0,
            table_end=buffer_end if buffer_end is not None else (buffer_start or 0),
            headers=headers,
            rows=structured_rows,
        )
        caption = _build_caption(
            surrounding_sentences=surrounding_sentences,
            heading=_find_nearest_heading(lines, buffer_start or 0),
        )
        tables.append(
            {
                "table_id": f"{path.stem}-table-{table_index}",
                "caption": caption,
                "context": context,
                "page_range": "1",
                "row_count": len(structured_rows),
                "column_count": len(headers),
                "rows": structured_rows,
            }
        )
        end_index = buffer_end if buffer_end is not None else buffer_start
        table_ranges.append((buffer_start, end_index))
        buffer = []
        buffer_start = None
        buffer_end = None
        in_table = False

    for idx, line in enumerate(lines):
        if "|" in line:
            if not in_table:
                buffer = []
                buffer_start = idx
            buffer.append(line)
            buffer_end = idx
            in_table = True
        else:
            if in_table:
                flush_buffer()
            in_table = False

    if in_table:
        flush_buffer()

    return tables, table_index, table_ranges


_TABLE_RE = re.compile(r"<table.*?>.*?</table>", re.IGNORECASE | re.DOTALL)
_ROW_RE = re.compile(r"<tr.*?>.*?</tr>", re.IGNORECASE | re.DOTALL)
_CELL_RE = re.compile(r"<t[dh].*?>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<.*?>", re.DOTALL)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.!?])\s+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _strip_html(value: str) -> str:
    cleaned = _TAG_RE.sub(" ", value)
    return unescape(cleaned).strip()


def _select_context_sentence(
    *,
    lines: Sequence[str],
    table_start: int,
    table_end: int,
    headers: Sequence[str],
    rows: Sequence[dict[str, str]],
    window: int = 6,
    max_length: int = 100,
    support_sentences: int = 6,
) -> tuple[str | None, list[str]]:
    keywords = _collect_table_keywords(headers, rows)

    before_sentences, after_sentences = _gather_surrounding_sentences(
        lines=lines,
        table_start=table_start,
        table_end=table_end,
        window=window,
    )

    ordered_support: list[str] = []
    ordered_support.extend(reversed(before_sentences))
    ordered_support.extend(after_sentences)
    ordered_support = [sentence for sentence in ordered_support if sentence][:support_sentences]

    if not keywords:
        primary = ordered_support[0] if ordered_support else None
        return _truncate_text(primary, max_length=max_length) if primary else None, ordered_support

    best_sentence: str | None = None
    best_score: tuple[int, int] | None = None

    for rank, sentence in enumerate(ordered_support, start=1):
        tokens = _tokenize(sentence)
        if not tokens:
            continue
        overlap = len(tokens & keywords)
        numeric_bonus = 1 if any(char.isdigit() for char in sentence) else 0
        score = (overlap * 10 + numeric_bonus, -rank)
        if overlap == 0:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_sentence = sentence

    if not best_sentence:
        primary = ordered_support[0] if ordered_support else None
        truncated = _truncate_text(primary, max_length=max_length) if primary else None
        return truncated if truncated else None, ordered_support

    truncated = _truncate_text(best_sentence, max_length=max_length)
    return truncated if truncated else None, ordered_support


def _gather_surrounding_sentences(
    *,
    lines: Sequence[str],
    table_start: int,
    table_end: int,
    window: int,
) -> tuple[list[str], list[str]]:
    before_indices = range(max(0, table_start - window), table_start)
    before_sentences = _split_sentences(lines, before_indices)

    after_indices = range(table_end + 1, min(len(lines), table_end + 1 + window))
    after_sentences = _split_sentences(lines, after_indices)

    return before_sentences, after_sentences


def _split_sentences(lines: Sequence[str], indices: Iterable[int]) -> list[str]:
    fragments: list[str] = []
    for idx in indices:
        if 0 <= idx < len(lines):
            text = lines[idx].strip()
            if text:
                fragments.append(text)
    buffer = " ".join(fragments)
    if not buffer:
        return []
    sentences = _SENTENCE_SPLIT_RE.split(buffer)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _find_nearest_heading(lines: Sequence[str], table_start: int, search_window: int = 12) -> str | None:
    limit = max(-1, table_start - search_window)
    for idx in range(table_start - 1, limit, -1):
        if idx < 0 or idx >= len(lines):
            continue
        raw = lines[idx].strip()
        if not raw:
            continue
        if raw.startswith("#"):
            value = raw.lstrip("#").strip()
            if value:
                return value
            continue
        if raw.isupper() and len(raw) <= 80:
            return raw.title()
        if raw.endswith(":") and len(raw) <= 120:
            return raw.rstrip(":").strip()
    return None


def _build_caption(
    *,
    surrounding_sentences: Sequence[str],
    heading: str | None,
    max_length: int = 220,
) -> str | None:
    candidates: list[str] = []

    if heading:
        heading_clean = heading.strip()
        if heading_clean:
            candidates.append(heading_clean)

    for sentence in surrounding_sentences:
        if not sentence:
            continue
        if candidates and sentence.lower() == candidates[-1].lower():
            continue
        candidates.append(sentence)
        if len(candidates) >= 3:
            break

    if not candidates:
        return None

    caption = " ".join(candidates).strip()
    if not caption:
        return None

    caption = " ".join(caption.split())
    if len(caption) > max_length:
        caption = _truncate_text(caption, max_length=max_length)
    return caption


def _collect_table_keywords(headers: Sequence[str], rows: Sequence[dict[str, str]]) -> set[str]:
    keywords: set[str] = set()

    for header in headers:
        keywords.update(_tokenize(header))

    if rows:
        first_row = rows[0]
        first_key = next(iter(first_row.keys()), "")
        keywords.update(_tokenize(first_key))

        for row in rows[:3]:
            if first_key:
                keywords.update(_tokenize(str(row.get(first_key, ""))))

    return {token for token in keywords if token}


def _tokenize(value: str) -> set[str]:
    return {match.group(0).lower() for match in _TOKEN_RE.finditer(value)}


def _truncate_text(sentence: str, *, max_length: int) -> str:
    sentence = sentence.strip()
    if len(sentence) <= max_length:
        return sentence

    words = sentence.split()
    if not words:
        return ""

    result_words: list[str] = []
    total_length = 0

    for word in words:
        addition = len(word) if not result_words else len(word) + 1
        if total_length + addition > max_length:
            break
        result_words.append(word)
        total_length += addition

    return " ".join(result_words).strip()


def _extract_html_tables(
    text: str,
    path: Path,
    *,
    start_index: int = 0,
) -> Tuple[list[dict[str, Any]], int, list[tuple[int, int]]]:
    tables: list[dict[str, Any]] = []
    table_index = start_index
    spans: list[tuple[int, int]] = []

    for table_match in _TABLE_RE.finditer(text):
        table_html = table_match.group(0)
        spans.append(table_match.span())
        rows = _ROW_RE.findall(table_html)
        if not rows:
            continue

        parsed_rows: list[list[str]] = [
            [_strip_html(cell) for cell in _CELL_RE.findall(row_html)]
            for row_html in rows
        ]
        if not parsed_rows:
            continue

        max_cols = max(len(row) for row in parsed_rows)

        def pad(row: list[str]) -> list[str]:
            return row + [""] * (max_cols - len(row))

        normalized_rows = [pad(row) for row in parsed_rows]

        header_rows: list[list[str]] = []
        data_start = len(normalized_rows)
        for idx, row in enumerate(normalized_rows):
            first_cell = row[0].strip() if row else ""
            if idx == 0 or not first_cell:
                header_rows.append(row)
            else:
                data_start = idx
                break

        data_rows_raw = normalized_rows[data_start:]
        if not data_rows_raw:
            data_rows_raw = normalized_rows[len(header_rows):]

        headers: list[str] = []
        for col in range(max_cols):
            header_value = ""
            for header_row in header_rows:
                candidate = header_row[col].strip()
                if candidate:
                    header_value = candidate
            if not header_value:
                header_value = "category" if col == 0 else f"column_{col}"
            headers.append(header_value)

        data_rows: list[dict[str, str]] = []
        for row in data_rows_raw:
            row_dict: dict[str, str] = {}
            has_value = False
            for idx, header in enumerate(headers):
                value = row[idx].strip()
                if value:
                    has_value = True
                row_dict[header] = value
            if has_value:
                data_rows.append(row_dict)

        table_index += 1
        tables.append(
            {
                "table_id": f"{path.stem}-table-{table_index}",
                "caption": None,
                "page_range": "1",
                "row_count": len(data_rows),
                "column_count": len(headers),
                "rows": data_rows,
            }
        )

    return tables, table_index, spans


def _compute_line_offsets(segments: list[str]) -> list[int]:
    offsets: list[int] = []
    current = 0
    for segment in segments:
        offsets.append(current)
        current += len(segment)
    return offsets


def _convert_spans_to_line_ranges(
    spans: list[tuple[int, int]],
    line_offsets: list[int],
    line_count: int,
) -> list[tuple[int, int]]:
    if not spans or line_count == 0:
        return []

    def offset_to_index(offset: int) -> int:
        idx = bisect_right(line_offsets, offset) - 1
        if idx < 0:
            return 0
        if idx >= line_count:
            return line_count - 1
        return idx

    ranges: list[tuple[int, int]] = []
    for start, end in spans:
        start_idx = offset_to_index(start)
        end_idx = offset_to_index(max(start, end - 1))
        ranges.append((start_idx, end_idx))
    return ranges

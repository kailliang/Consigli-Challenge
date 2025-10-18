"""Markdown parsing utilities."""

from __future__ import annotations

import re
from html import unescape
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple


@dataclass(slots=True)
class MarkdownParseResult:
    page_count: int
    tables: list[dict[str, Any]]
    sections: list[dict[str, Any]]
    full_text: str


def parse_markdown(path: Path) -> MarkdownParseResult:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    tables, _ = _extract_markdown_tables(lines, path)
    html_tables, _ = _extract_html_tables(text, path, start_index=len(tables))
    tables.extend(html_tables)
    sections = _extract_sections(lines)

    # Markdown files lack pages; treat entire file as one logical page.
    page_count = max(1, len(sections))

    return MarkdownParseResult(
        page_count=page_count,
        tables=tables,
        sections=sections,
        full_text=text,
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
) -> Tuple[list[dict[str, Any]], int]:
    tables: list[dict[str, Any]] = []
    buffer: list[str] = []
    in_table = False
    table_index = start_index

    def flush_buffer() -> None:
        nonlocal buffer, in_table, table_index
        if not buffer:
            return
        header_line = buffer[0]
        delimiter_line = buffer[1] if len(buffer) > 1 else ""
        if "|" not in header_line:
            buffer = []
            in_table = False
            return
        headers = [cell.strip() or f"col_{idx}" for idx, cell in enumerate(header_line.split("|")) if cell.strip()]
        if not headers:
            buffer = []
            in_table = False
            return
        data_rows = [row for row in buffer[2:] if row.strip()]
        structured_rows: list[dict[str, str]] = []
        for row in data_rows:
            values = [cell.strip() for cell in row.split("|") if cell.strip()]
            structured_rows.append({headers[idx] if idx < len(headers) else f"col_{idx}": values[idx] if idx < len(values) else "" for idx in range(len(headers))})
        table_index += 1
        tables.append(
            {
                "table_id": f"{path.stem}-table-{table_index}",
                "caption": None,
                "page_range": "1",
                "row_count": len(structured_rows),
                "column_count": len(headers),
                "rows": structured_rows,
            }
        )
        buffer = []
        in_table = False

    for line in lines:
        if "|" in line:
            buffer.append(line)
            in_table = True
        else:
            if in_table:
                flush_buffer()
            in_table = False

    if in_table:
        flush_buffer()

    return tables, table_index


_TABLE_RE = re.compile(r"<table.*?>.*?</table>", re.IGNORECASE | re.DOTALL)
_ROW_RE = re.compile(r"<tr.*?>.*?</tr>", re.IGNORECASE | re.DOTALL)
_CELL_RE = re.compile(r"<t[dh].*?>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<.*?>", re.DOTALL)


def _strip_html(value: str) -> str:
    cleaned = _TAG_RE.sub(" ", value)
    return unescape(cleaned).strip()


def _extract_html_tables(
    text: str,
    path: Path,
    *,
    start_index: int = 0,
) -> Tuple[list[dict[str, Any]], int]:
    tables: list[dict[str, Any]] = []
    table_index = start_index

    for table_match in _TABLE_RE.finditer(text):
        table_html = table_match.group(0)
        rows = _ROW_RE.findall(table_html)
        if not rows:
            continue

        header_cells = _CELL_RE.findall(rows[0])
        headers = [
            (_strip_html(cell) or f"col_{idx}")
            for idx, cell in enumerate(header_cells)
        ]
        if not headers:
            continue

        data_rows: list[dict[str, str]] = []
        for row_html in rows[1:]:
            cells = _CELL_RE.findall(row_html)
            if not cells:
                continue
            row_dict: dict[str, str] = {}
            has_value = False
            for idx, header in enumerate(headers):
                value = _strip_html(cells[idx]) if idx < len(cells) else ""
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

    return tables, table_index

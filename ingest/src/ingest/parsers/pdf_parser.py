"""PDF parsing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pdfplumber
from pdfplumber.page import Page


@dataclass(slots=True)
class PDFParseResult:
    page_count: int
    tables: list[dict[str, Any]]
    sections: list[dict[str, Any]]
    full_text: str


def parse_pdf(path: Path) -> PDFParseResult:
    tables: list[dict[str, Any]] = []
    sections: list[dict[str, Any]] = []
    seen_headings: set[str] = set()
    full_text_parts: list[str] = []

    with pdfplumber.open(path) as pdf:
        page_count = len(pdf.pages)

        for page_index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text:
                full_text_parts.append(text)
            _extract_tables(path, page, page_index, tables)
            _extract_headings(page, page_index, sections, seen_headings)

    return PDFParseResult(
        page_count=page_count,
        tables=tables,
        sections=sections,
        full_text="\n\n".join(full_text_parts),
    )


def _extract_tables(path: Path, page: Page, page_number: int, tables: list[dict[str, Any]]) -> None:
    try:
        extracted_tables = page.extract_tables() or []
    except Exception:  # pragma: no cover - pdfplumber internal failures
        extracted_tables = []

    for table_index, raw_table in enumerate(extracted_tables, start=1):
        if not raw_table:
            continue

        filtered_rows = [row for row in raw_table if any(_sanitize_cell(cell) for cell in row)]
        if not filtered_rows:
            continue

        header = [cell or f"col_{index}" for index, cell in enumerate(filtered_rows[0])]
        data_rows = filtered_rows[1:] if len(filtered_rows) > 1 else []
        column_count = len(header)
        row_count = len(data_rows)
        structured_rows = [_map_row(header, row) for row in data_rows]

        tables.append(
            {
                "table_id": f"{path.stem}-p{page_number}-t{table_index}",
                "caption": None,
                "page_range": str(page_number),
                "row_count": row_count,
                "column_count": column_count,
                "rows": structured_rows,
            }
        )


def _extract_headings(
    page: Page,
    page_number: int,
    sections: list[dict[str, Any]],
    seen_headings: set[str],
) -> None:
    text = page.extract_text() or ""

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if _looks_like_heading(line) and line not in seen_headings:
            seen_headings.add(line)
            sections.append({"level": 1, "title": f"{line} (p.{page_number})"})


def _sanitize_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _looks_like_heading(line: str) -> bool:
    if len(line) > 120:
        return False
    if line.isupper():
        return True
    if line.endswith(":"):
        return True
    capitalised_ratio = sum(1 for char in line if char.isupper()) / max(len(line), 1)
    return capitalised_ratio > 0.6


def _map_row(headers: list[Any], row: list[Any]) -> dict[str, str]:
    mapped: dict[str, str] = {}
    for index, header in enumerate(headers):
        key = _sanitize_cell(header) or f"col_{index}"
        value = _sanitize_cell(row[index]) if index < len(row) else ""
        mapped[str(key)] = value
    return mapped

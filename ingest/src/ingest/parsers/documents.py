"""Document parsing facade for PDF, DOCX, and Markdown files."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ..utils.files import compute_sha256
from .markdown_parser import parse_markdown
from .pdf_parser import parse_pdf
from .word_parser import parse_docx

SupportedSuffix = Literal[".pdf", ".docx", ".md"]


class TableSummary(BaseModel):
    table_id: str
    caption: str | None = None
    page_range: str | None = None
    row_count: int
    column_count: int
    rows: list[dict[str, str]] = Field(default_factory=list)


class SectionSummary(BaseModel):
    level: int
    title: str


class ParsedDocument(BaseModel):
    name: str
    path: Path
    sha256: str
    doc_type: str
    page_count: int
    table_count: int
    tables: list[TableSummary]
    sections: list[SectionSummary]
    full_text: str


def parse_document(path: Path) -> ParsedDocument:
    suffix = path.suffix.lower()
    if suffix not in {".pdf", ".docx", ".md"}:
        raise ValueError(f"Unsupported document type: {suffix}")

    sha256 = compute_sha256(path)

    if suffix == ".pdf":
        summary = parse_pdf(path)
    elif suffix == ".docx":
        summary = parse_docx(path)
    else:
        summary = parse_markdown(path)

    return ParsedDocument(
        name=path.name,
        path=path,
        sha256=sha256,
        doc_type=suffix,
        page_count=summary.page_count,
        table_count=len(summary.tables),
        tables=[TableSummary(**table) for table in summary.tables],
        sections=[SectionSummary(**section) for section in summary.sections],
        full_text=summary.full_text,
    )


def serialize_manifest(records: list[dict]) -> str:
    from json import dumps

    return dumps(records, indent=2, default=str)

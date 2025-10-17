"""SQLAlchemy models for structured fact storage."""

from __future__ import annotations

from sqlalchemy import JSON, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base declarative model."""


class TableSummary(Base):
    """Normalized table metadata and rows."""

    __tablename__ = "table_summaries"
    __table_args__ = (UniqueConstraint("company", "year", "table_id", name="uq_table_identity"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    company: Mapped[str] = mapped_column(String, nullable=False)
    year: Mapped[int] = mapped_column(Integer, nullable=False)
    document_name: Mapped[str] = mapped_column(String, nullable=False)
    table_id: Mapped[str] = mapped_column(String, nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    column_count: Mapped[int] = mapped_column(Integer, nullable=False)
    page_range: Mapped[str | None] = mapped_column(String, nullable=True)
    caption: Mapped[str | None] = mapped_column(String, nullable=True)
    rows: Mapped[list[dict]] = mapped_column(JSON, default=list)

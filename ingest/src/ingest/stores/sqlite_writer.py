"""Structured store writer for table summaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sqlalchemy import JSON, Column, Integer, MetaData, String, Table, UniqueConstraint, create_engine
from sqlalchemy.dialects.sqlite import insert as sqlite_insert


@dataclass(slots=True)
class StructuredStore:
    db_path: Path

    def __post_init__(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(f"sqlite:///{self.db_path}")
        self._metadata = MetaData()
        self._table_summaries = Table(
            "table_summaries",
            self._metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("company", String, nullable=False),
            Column("year", Integer, nullable=False),
            Column("document_name", String, nullable=False),
            Column("table_id", String, nullable=False),
            Column("row_count", Integer, nullable=False),
            Column("column_count", Integer, nullable=False),
            Column("page_range", String),
            Column("caption", String),
            Column("rows", JSON, nullable=False),
            UniqueConstraint("company", "year", "table_id", name="uq_table_identity"),
        )
        self._metadata.create_all(self._engine)

    def bulk_upsert_table_summaries(self, records: Iterable[dict]) -> None:
        record_list = list(records)
        if not record_list:
            return

        stmt = sqlite_insert(self._table_summaries).values(record_list)
        update_target = {
            "row_count": stmt.excluded.row_count,
            "column_count": stmt.excluded.column_count,
            "page_range": stmt.excluded.page_range,
            "caption": stmt.excluded.caption,
            "rows": stmt.excluded.rows,
        }

        upsert_stmt = stmt.on_conflict_do_update(index_elements=[self._table_summaries.c.company, self._table_summaries.c.year, self._table_summaries.c.table_id], set_=update_target)

        with self._engine.begin() as connection:
            connection.execute(upsert_stmt)

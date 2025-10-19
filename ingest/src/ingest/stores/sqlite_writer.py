from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(slots=True)
class StructuredStore:
    db_path: Path

    def bulk_upsert_table_summaries(self, table_records: Iterable[dict]) -> None:
        """Create/update the table_summaries store with the provided records.

        The schema is designed to support finance-heavy queries such as
        multi-year revenue / profit lookups by company.
        """

        records = list(table_records)
        if not records:
            return

        with self._connect() as conn:
            self._ensure_schema(conn)
            payload = [
                (
                    record.get("company"),
                    record.get("year"),
                    record.get("document_name"),
                    record.get("table_id"),
                    record.get("row_count") or 0,
                    record.get("column_count") or 0,
                    record.get("page_range"),
                    record.get("caption"),
                    json.dumps(record.get("rows", []), ensure_ascii=False),
                )
                for record in records
                if record.get("table_id") and record.get("company") and record.get("year") is not None
            ]

            conn.executemany(
                """
                INSERT INTO table_summaries (
                    company,
                    year,
                    document_name,
                    table_id,
                    row_count,
                    column_count,
                    page_range,
                    caption,
                    rows,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(company, year, table_id) DO UPDATE SET
                    document_name = excluded.document_name,
                    row_count = excluded.row_count,
                    column_count = excluded.column_count,
                    page_range = excluded.page_range,
                    caption = excluded.caption,
                    rows = excluded.rows,
                    updated_at = CURRENT_TIMESTAMP
                """,
                payload,
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS table_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company TEXT NOT NULL,
                year INTEGER NOT NULL,
                document_name TEXT NOT NULL,
                table_id TEXT NOT NULL,
                row_count INTEGER NOT NULL DEFAULT 0,
                column_count INTEGER NOT NULL DEFAULT 0,
                page_range TEXT,
                caption TEXT,
                rows TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(company, year, table_id)
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_table_summaries_company_year
            ON table_summaries(company, year)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_table_summaries_table_id
            ON table_summaries(table_id)
            """
        )

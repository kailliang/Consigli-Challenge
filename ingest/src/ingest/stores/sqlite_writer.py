from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(slots=True)
class StructuredStore:
    db_path: Path

    def bulk_upsert_table_summaries(self, table_records: Iterable[dict]) -> None:
        """No-op placeholder for quick end-to-end iteration.

        Replace with real SQLite schema and upsert logic.
        """

        return None


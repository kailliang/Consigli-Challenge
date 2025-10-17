"""Structured store access helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Iterable
import re

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.tables import TableSummary


async def fetch_tables_by_ids(
    session: AsyncSession,
    table_ids: Iterable[str],
) -> dict[str, TableSummary]:
    """Fetch table summaries keyed by table_id."""

    ids = list({table_id for table_id in table_ids if table_id})

    if not ids:
        return {}

    stmt = select(TableSummary).where(TableSummary.table_id.in_(ids))
    results = await session.scalars(stmt)
    return {row.table_id: row for row in results}


async def fetch_recent_tables(
    session: AsyncSession,
    *,
    company: str | None = None,
    year: int | None = None,
    limit: int = 10,
) -> Sequence[TableSummary]:
    """Fetch recent table summaries for inspection."""

    stmt = select(TableSummary).order_by(TableSummary.id.desc()).limit(limit)

    if company:
        stmt = stmt.where(TableSummary.company == company)
    if year:
        stmt = stmt.where(TableSummary.year == year)

    results = await session.scalars(stmt)
    return list(results)


def summarize_time_series(
    table_summary: TableSummary,
    *,
    max_rows: int = 5,
    max_points: int = 3,
) -> list[str]:
    """Produce heuristic time-series summaries from table rows."""

    rows = table_summary.rows or []
    if not rows:
        return []

    summaries: list[str] = []

    for row in rows[:max_rows]:
        items = list(row.items())
        if not items:
            continue

        label_value = str(items[0][1]).strip()
        if not label_value:
            label_value = str(items[0][0]).strip()

        if not label_value:
            continue

        points: list[str] = []
        for key, value in items[1:]:
            if len(points) >= max_points:
                break

            if _looks_like_year(str(key)) and _looks_like_number(str(value)):
                points.append(f"{key}: {value}")

        if points:
            summaries.append(f"{label_value} -> {' | '.join(points)}")

    return summaries


_YEAR_PATTERN = re.compile(r"^\d{4}$")
_NUMBER_PATTERN = re.compile(r"^[+-]?(?:\d+[\d,\.]*|\d{1,3}(?:,\d{3})+(?:\.\d+)?)$")


def _looks_like_year(value: str) -> bool:
    return bool(_YEAR_PATTERN.match(value.strip()))


def _looks_like_number(value: str) -> bool:
    cleaned = value.strip()
    if not cleaned:
        return False
    return bool(_NUMBER_PATTERN.match(cleaned))

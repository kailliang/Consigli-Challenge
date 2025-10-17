"""Structured store access helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Iterable

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

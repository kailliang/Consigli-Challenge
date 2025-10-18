"""Structured store access helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
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


@dataclass(slots=True)
class MetricSeries:
    metric: str
    points: list[tuple[str, str]]
    source_table_id: str


def extract_metric_series(
    table_summary: TableSummary,
    *,
    aliases: dict[str, list[str]] | None = None,
    max_points: int = 4,
) -> list[MetricSeries]:
    """Extract metric time series for named metrics from a table summary."""

    if aliases is None:
        aliases = {
            "revenue": ["revenue", "sales"],
            "ebitda": ["ebitda"],
            "net_income": ["net", "profit"],
        }

    rows = table_summary.rows or []
    if not rows:
        return []

    series: list[MetricSeries] = []

    for row in rows:
        label = _extract_label(row)
        if not label:
            continue

        normalized_label = label.lower()
        target_metric = None
        for metric_name, alias_list in aliases.items():
            if any(alias in normalized_label for alias in alias_list):
                target_metric = metric_name
                break

        if not target_metric:
            continue

        points: list[tuple[str, str]] = []
        for key, value in row.items():
            if key == label:
                continue
            if _looks_like_year(str(key)) and _looks_like_number(str(value)):
                points.append((str(key), str(value)))
            if len(points) >= max_points:
                break

        if points:
            series.append(MetricSeries(metric=target_metric, points=points, source_table_id=table_summary.table_id))

    return series


_YEAR_PATTERN = re.compile(r"^\d{4}$")
_NUMBER_PATTERN = re.compile(r"^[+-]?(?:\d+[\d,\.]*|\d{1,3}(?:,\d{3})+(?:\.\d+)?)$")


def _looks_like_year(value: str) -> bool:
    return bool(_YEAR_PATTERN.match(value.strip()))


def _looks_like_number(value: str) -> bool:
    cleaned = value.strip()
    if not cleaned:
        return False
    return bool(_NUMBER_PATTERN.match(cleaned))


def _extract_label(row: dict) -> str | None:
    if not row:
        return None
    first_key = next(iter(row))
    raw = str(row.get(first_key) or first_key)
    return raw.strip() or None

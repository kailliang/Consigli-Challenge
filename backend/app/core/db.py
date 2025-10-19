"""Database utilities for structured fact storage."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from .config import get_settings

_async_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_structured_db_url(raw_url: str) -> str:
    """Ensure SQLite URLs point at the repository-level data directory."""

    url = make_url(raw_url)
    if "sqlite" not in url.drivername or url.database is None:
        return raw_url

    db_path = Path(url.database)
    if not db_path.is_absolute():
        db_path = (_REPO_ROOT / db_path).resolve()

    db_path.parent.mkdir(parents=True, exist_ok=True)
    url = url.set(database=str(db_path))
    return str(url)


def get_engine() -> AsyncEngine:
    """Return a singleton instance of the async engine."""

    global _async_engine
    if _async_engine is None:
        settings = get_settings()
        resolved_url = _resolve_structured_db_url(settings.structured_db_url)
        _async_engine = create_async_engine(resolved_url, echo=False, future=True)
    return _async_engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Create or return the shared async session factory."""

    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _session_factory


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """Provide an async database session via dependency injection."""

    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            yield session
        finally:
            await session.close()

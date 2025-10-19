"""FastAPI application entrypoint."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import api_router
from .core.config import AppSettings, get_settings
from .core.logging import get_logger, setup_logging

logger = get_logger(__name__)


def _configure_tracing(settings: AppSettings) -> None:
    """Propagate LangSmith settings into LangChain environment variables."""

    if not settings.enable_tracing:
        return

    if settings.langsmith_api_key:
        os.environ.setdefault("LANGCHAIN_API_KEY", settings.langsmith_api_key)
    if settings.langsmith_endpoint:
        os.environ.setdefault("LANGCHAIN_ENDPOINT", settings.langsmith_endpoint)
    if settings.langsmith_project:
        os.environ.setdefault("LANGCHAIN_PROJECT", settings.langsmith_project)

    if os.environ.get("LANGCHAIN_API_KEY"):
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifecycle hooks for startup and shutdown."""

    settings = get_settings()
    _configure_tracing(settings)
    setup_logging()

    logger.info(
        "application.startup",
        environment=settings.environment,
        version=settings.version,
        tracing_enabled=settings.enable_tracing,
    )

    try:
        yield
    finally:
        logger.info("application.shutdown")


def create_app(settings: AppSettings | None = None) -> FastAPI:
    """Construct the FastAPI application instance."""

    settings = settings or get_settings()

    application = FastAPI(
        title=settings.project_name,
        version=settings.version,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    if settings.cors_allowed_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_allowed_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    application.include_router(api_router, prefix="/v1")

    @application.get("/", tags=["meta"], summary="Service metadata")
    async def root() -> dict[str, str]:
        """Service metadata root endpoint."""

        return {"service": settings.project_name, "version": settings.version}

    return application


app = create_app()

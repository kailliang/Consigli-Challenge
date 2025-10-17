"""API routers for the backend service."""

from fastapi import APIRouter

from .routes import health_router
from .v1 import query

api_router = APIRouter()
api_router.include_router(health_router, prefix="/health", tags=["health"])
api_router.include_router(query.router, prefix="", tags=["query"])

__all__ = ["api_router"]

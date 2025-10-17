"""Primary API route definitions."""

from fastapi import APIRouter

health_router = APIRouter()


@health_router.get("/", summary="Readiness probe", tags=["health"])
async def healthcheck() -> dict[str, str]:
    """Report service readiness for liveness probes."""

    return {"status": "ok"}

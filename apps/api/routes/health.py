"""Health check route."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    """Liveness probe – returns 200 when the API is up."""
    return {"status": "ok"}

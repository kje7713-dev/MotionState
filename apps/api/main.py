"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from apps.api.routes import health, jobs, projects, videos, webhooks
from libs.db import engine
from libs.models import Base


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create DB tables on startup (dev convenience; use Alembic in prod)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(
    title="MotionState Pipeline API",
    description="Video ingest → normalization → queued processing → structured motion output.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(videos.router, prefix="/videos", tags=["videos"])
app.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
app.include_router(projects.router, prefix="/projects", tags=["projects"])
app.include_router(webhooks.router, prefix="/projects", tags=["webhooks"])

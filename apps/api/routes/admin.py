"""Admin-only operational visibility endpoints.

All routes require a valid ``X-Admin-Token`` header matching the configured
``admin_token`` setting.  When ``admin_token`` is empty the endpoints are
disabled (return 403) so they are safe to deploy without any additional
infrastructure.

Available endpoints:

- GET /admin/health/summary  — app liveness + dependency reachability
- GET /admin/runs/recent     — last N processing runs (ordered newest-first)
- GET /admin/failures/recent — last N errored runs with failure details
- GET /admin/projects/usage/top — top projects by usage in current month
- GET /admin/metrics         — JSON metrics summary
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from libs.config import settings
from libs.db import get_db
from libs.models import ProcessingRun, RunStatus, UsageEvent, UsageEventType, Video
from libs.storage import get_storage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])

_ADMIN_TOKEN_HEADER = APIKeyHeader(
    name="X-Admin-Token", scheme_name="AdminTokenAuth", auto_error=False
)

# ---------------------------------------------------------------------------
# Admin auth dependency
# ---------------------------------------------------------------------------


def require_admin(token: str | None = Security(_ADMIN_TOKEN_HEADER)) -> None:
    """Dependency that enforces admin token authentication.

    Raises:
        HTTPException 403: if admin_token is not configured or the provided
            token does not match.
    """
    configured = settings.admin_token
    if not configured:
        raise HTTPException(status_code=403, detail="Admin access is not configured")
    if not token or token != configured:
        raise HTTPException(status_code=403, detail="Invalid or missing admin token")


# ---------------------------------------------------------------------------
# Health summary
# ---------------------------------------------------------------------------


@router.get("/health/summary", dependencies=[Depends(require_admin)])
async def health_summary(db: AsyncSession = Depends(get_db)) -> dict:
    """Extended health check with per-dependency reachability.

    Checks:
    - app: always ``ok`` when this handler runs
    - db: executes a trivial ``SELECT 1``
    - redis: pings the Redis connection used by the queue
    - storage: confirms the configured storage backend is reachable

    Returns a machine-readable dict.  Individual checks return ``"ok"`` or
    ``"error: <detail>"``.
    """
    checks: dict[str, str] = {"app": "ok"}

    # --- DB check ---
    try:
        result = await db.execute(text("SELECT 1"))
        result.scalar()
        checks["db"] = "ok"
    except Exception as exc:
        logger.warning("Admin health: DB check failed: %s", exc)
        checks["db"] = f"error: {exc}"

    # --- Redis check ---
    try:
        r = aioredis.from_url(settings.redis_url, socket_connect_timeout=2)
        await r.ping()
        await r.aclose()
        checks["redis"] = "ok"
    except Exception as exc:
        logger.warning("Admin health: Redis check failed: %s", exc)
        checks["redis"] = f"error: {exc}"

    # --- Storage check ---
    try:
        storage = get_storage(settings)
        await storage.check_reachable()
        checks["storage"] = "ok"
    except Exception as exc:
        logger.warning("Admin health: Storage check failed: %s", exc)
        checks["storage"] = f"error: {exc}"

    overall = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
    return {
        "status": overall,
        "checks": checks,
        "checked_at": datetime.now(UTC).isoformat(),
    }


# ---------------------------------------------------------------------------
# Recent runs
# ---------------------------------------------------------------------------


@router.get("/runs/recent", dependencies=[Depends(require_admin)])
async def recent_runs(
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    """Return the most recent processing runs, newest first.

    Args:
        limit: Maximum number of runs to return (capped at 100).
    """
    limit = min(limit, 100)
    result = await db.execute(
        select(ProcessingRun, Video.project_id)
        .join(Video, ProcessingRun.video_id == Video.id)
        .order_by(ProcessingRun.id.desc())
        .limit(limit)
    )
    rows = result.all()
    return [
        {
            "run_id": run.id,
            "video_id": run.video_id,
            "project_id": project_id,
            "status": run.status,
            "trigger_type": run.trigger_type,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "duration_ms": (
                int((run.completed_at - run.started_at).total_seconds() * 1000)
                if run.started_at and run.completed_at
                else None
            ),
            "error": run.error,
            "created_at": run.created_at.isoformat(),
        }
        for run, project_id in rows
    ]


# ---------------------------------------------------------------------------
# Recent failures
# ---------------------------------------------------------------------------


@router.get("/failures/recent", dependencies=[Depends(require_admin)])
async def recent_failures(
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    """Return recent errored processing runs with failure details.

    Args:
        limit: Maximum number of failures to return (capped at 100).
    """
    limit = min(limit, 100)
    result = await db.execute(
        select(ProcessingRun, Video.project_id)
        .join(Video, ProcessingRun.video_id == Video.id)
        .where(ProcessingRun.status == RunStatus.error)
        .order_by(ProcessingRun.id.desc())
        .limit(limit)
    )
    rows = result.all()
    return [
        {
            "run_id": run.id,
            "video_id": run.video_id,
            "project_id": project_id,
            "error": run.error,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "failed_at": run.completed_at.isoformat() if run.completed_at else None,
            "created_at": run.created_at.isoformat(),
        }
        for run, project_id in rows
    ]


# ---------------------------------------------------------------------------
# Top projects by usage
# ---------------------------------------------------------------------------


@router.get("/projects/usage/top", dependencies=[Depends(require_admin)])
async def top_projects_usage(
    limit: int = 10,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return the top projects by videos processed in the current calendar month.

    Args:
        limit: Maximum number of projects to return (capped at 50).
    """
    limit = min(limit, 50)
    now = datetime.now(UTC)
    if now.month == 12:
        window_end = datetime(now.year + 1, 1, 1, tzinfo=UTC)
    else:
        window_end = datetime(now.year, now.month + 1, 1, tzinfo=UTC)
    window_start = datetime(now.year, now.month, 1, tzinfo=UTC)

    result = await db.execute(
        select(
            UsageEvent.project_id,
            func.sum(UsageEvent.quantity).label("total"),
        )
        .where(
            UsageEvent.event_type == UsageEventType.video_seconds_processed,
            UsageEvent.created_at >= window_start,
            UsageEvent.created_at < window_end,
        )
        .group_by(UsageEvent.project_id)
        .order_by(func.sum(UsageEvent.quantity).desc())
        .limit(limit)
    )
    rows = result.all()
    return {
        "year": now.year,
        "month": now.month,
        "metric": "video_seconds_processed",
        "projects": [
            {"project_id": project_id, "total": int(total)}
            for project_id, total in rows
        ],
    }


# ---------------------------------------------------------------------------
# Metrics summary
# ---------------------------------------------------------------------------


@router.get("/metrics", dependencies=[Depends(require_admin)])
async def metrics_summary(db: AsyncSession = Depends(get_db)) -> dict:
    """Return a JSON metrics summary of system-wide processing activity.

    Includes counts of runs by status, recent throughput, and webhook
    delivery totals.
    """
    # Run counts by status
    run_counts_result = await db.execute(
        select(ProcessingRun.status, func.count(ProcessingRun.id).label("count"))
        .group_by(ProcessingRun.status)
    )
    run_counts = {str(row.status): row.count for row in run_counts_result.all()}

    # Average processing duration for completed runs (started_at and completed_at set)
    avg_duration_result = await db.execute(
        select(
            func.avg(
                func.extract("epoch", ProcessingRun.completed_at)
                - func.extract("epoch", ProcessingRun.started_at)
            ).label("avg_seconds")
        ).where(
            ProcessingRun.status == RunStatus.completed,
            ProcessingRun.started_at.isnot(None),
            ProcessingRun.completed_at.isnot(None),
        )
    )
    avg_seconds = avg_duration_result.scalar()

    # Total usage event counts (all time)
    usage_totals_result = await db.execute(
        select(
            UsageEvent.event_type,
            func.sum(UsageEvent.quantity).label("total"),
        ).group_by(UsageEvent.event_type)
    )
    usage_totals = {row.event_type: int(row.total) for row in usage_totals_result.all()}

    return {
        "runs_by_status": run_counts,
        "avg_processing_duration_seconds": (
            round(float(avg_seconds), 2) if avg_seconds is not None else None
        ),
        "usage_totals_alltime": usage_totals,
        "generated_at": datetime.now(UTC).isoformat(),
    }

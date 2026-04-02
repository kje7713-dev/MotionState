"""Usage event emission and aggregation helpers.

This module provides the core accounting primitives:

- :func:`emit` — append a single UsageEvent row (fire-and-forget safe)
- :func:`monthly_totals` — sum events for the current calendar month
- :func:`alltime_totals` — sum events across all time
- :func:`latest_storage_bytes` — total storage bytes written to date
- :func:`project_usage_summary` — structured summary dict for API responses
"""

from __future__ import annotations

import logging
from calendar import monthrange
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.models import UsageEvent, UsageEventType

logger = logging.getLogger(__name__)

# Canonical units for each event type.
_UNITS: dict[str, str] = {
    UsageEventType.videos_uploaded: "count",
    UsageEventType.video_seconds_processed: "seconds",
    UsageEventType.frames_extracted: "count",
    UsageEventType.clips_generated: "count",
    UsageEventType.storage_bytes_written: "bytes",
    UsageEventType.webhook_deliveries: "count",
    UsageEventType.api_reads: "count",
}


async def emit(
    db: AsyncSession,
    *,
    project_id: int,
    event_type: UsageEventType | str,
    quantity: int,
    processing_run_id: int | None = None,
    metadata: dict | None = None,
) -> None:
    """Append a UsageEvent row.  Errors are logged but never re-raised.

    Args:
        db: Active async session.  The caller is responsible for committing.
        project_id: Project that consumed the resource.
        event_type: Which dimension is being metered.
        quantity: How many units were consumed (bytes, seconds, count, …).
        processing_run_id: Optional run that caused this consumption.
        metadata: Arbitrary key/value context stored as JSON.
    """
    try:
        unit = _UNITS.get(str(event_type), "count")
        event = UsageEvent(
            project_id=project_id,
            processing_run_id=processing_run_id,
            event_type=str(event_type),
            quantity=quantity,
            unit=unit,
            metadata_json=metadata,
        )
        db.add(event)
        # Flush so the row is visible within the same transaction; the caller
        # will commit (or the FastAPI get_db auto-commit will).
        await db.flush()
    except Exception:
        logger.exception(
            "Failed to emit usage event project=%s type=%s qty=%s",
            project_id,
            event_type,
            quantity,
        )


async def monthly_totals(
    db: AsyncSession,
    project_id: int,
    *,
    year: int | None = None,
    month: int | None = None,
) -> dict[str, int]:
    """Return a dict of {event_type: total_quantity} for the given month.

    Defaults to the current UTC calendar month when *year*/*month* are omitted.
    """
    now = datetime.now(UTC)
    target_year = year if year is not None else now.year
    target_month = month if month is not None else now.month

    # Inclusive window: first second of the month … first second of next month.
    monthrange(target_year, target_month)  # validate year/month combo
    window_start = datetime(target_year, target_month, 1, tzinfo=UTC)
    if target_month == 12:
        window_end = datetime(target_year + 1, 1, 1, tzinfo=UTC)
    else:
        window_end = datetime(target_year, target_month + 1, 1, tzinfo=UTC)

    result = await db.execute(
        select(UsageEvent.event_type, func.sum(UsageEvent.quantity).label("total"))
        .where(
            UsageEvent.project_id == project_id,
            UsageEvent.created_at >= window_start,
            UsageEvent.created_at < window_end,
        )
        .group_by(UsageEvent.event_type)
    )
    rows = result.all()
    return {r.event_type: int(r.total) for r in rows}


async def alltime_totals(db: AsyncSession, project_id: int) -> dict[str, int]:
    """Return a dict of {event_type: total_quantity} across all time."""
    result = await db.execute(
        select(UsageEvent.event_type, func.sum(UsageEvent.quantity).label("total"))
        .where(UsageEvent.project_id == project_id)
        .group_by(UsageEvent.event_type)
    )
    rows = result.all()
    return {r.event_type: int(r.total) for r in rows}


async def latest_storage_bytes(db: AsyncSession, project_id: int) -> int:
    """Return the cumulative storage bytes written for a project."""
    result = await db.execute(
        select(func.sum(UsageEvent.quantity)).where(
            UsageEvent.project_id == project_id,
            UsageEvent.event_type == UsageEventType.storage_bytes_written,
        )
    )
    total = result.scalar()
    return int(total) if total is not None else 0


async def project_usage_summary(db: AsyncSession, project_id: int) -> dict:
    """Return a structured usage summary for API responses.

    Includes both current-month and all-time totals plus a storage estimate.
    """
    current = await monthly_totals(db, project_id)
    total = await alltime_totals(db, project_id)
    storage = await latest_storage_bytes(db, project_id)

    now = datetime.now(UTC)
    return {
        "project_id": project_id,
        "current_month": {
            "year": now.year,
            "month": now.month,
            "totals": current,
        },
        "alltime": total,
        "storage_bytes_total": storage,
    }

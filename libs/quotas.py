"""Project-level quota enforcement.

Provides :func:`check_quota` which raises an ``HTTPException`` (403 or 429)
when a project would exceed a configured limit.

All quota fields on :class:`~libs.models.Project` are nullable; ``None``
means unlimited so existing projects without explicit limits are unaffected.
"""

from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from libs.models import Project, UsageEventType
from libs.usage import latest_storage_bytes, monthly_totals


async def check_quota(
    db: AsyncSession,
    project: Project,
    *,
    videos_upload: bool = False,
    video_seconds: int = 0,
) -> None:
    """Enforce project-level quotas before allowing a new operation.

    Args:
        db: Active async session (read-only; no writes performed here).
        project: The project whose quotas are checked.
        videos_upload: Set to ``True`` when a new video is being uploaded /
            initiated.  Checked against ``max_videos_per_month``.
        video_seconds: Non-zero when a video is being (re-)processed.
            Checked against ``max_video_seconds_per_month``.

    Raises:
        HTTPException 403: if the project is suspended.
        HTTPException 429: if any quota would be exceeded.
    """
    if project.is_suspended:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "quota_exceeded",
                "reason": "project_suspended",
                "message": "This project has been suspended and cannot submit new work.",
            },
        )

    # Only fetch month totals when at least one monthly quota is set.
    needs_monthly = (
        (videos_upload and project.max_videos_per_month is not None)
        or (video_seconds > 0 and project.max_video_seconds_per_month is not None)
    )

    if needs_monthly:
        totals = await monthly_totals(db, project.id)
    else:
        totals = {}

    if videos_upload and project.max_videos_per_month is not None:
        current = totals.get(UsageEventType.videos_uploaded, 0)
        if current >= project.max_videos_per_month:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "quota_exceeded",
                    "reason": "max_videos_per_month",
                    "limit": project.max_videos_per_month,
                    "current": current,
                    "message": (
                        f"Project has reached its monthly video upload limit "
                        f"({project.max_videos_per_month})."
                    ),
                },
            )

    if video_seconds > 0 and project.max_video_seconds_per_month is not None:
        current = totals.get(UsageEventType.video_seconds_processed, 0)
        projected = current + video_seconds
        if projected > project.max_video_seconds_per_month:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "quota_exceeded",
                    "reason": "max_video_seconds_per_month",
                    "limit": project.max_video_seconds_per_month,
                    "current": current,
                    "message": (
                        f"Project would exceed its monthly video-seconds limit "
                        f"({project.max_video_seconds_per_month}s)."
                    ),
                },
            )

    if project.max_storage_bytes is not None:
        storage_used = await latest_storage_bytes(db, project.id)
        if storage_used >= project.max_storage_bytes:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "quota_exceeded",
                    "reason": "max_storage_bytes",
                    "limit": project.max_storage_bytes,
                    "current": storage_used,
                    "message": (
                        f"Project has reached its storage limit "
                        f"({project.max_storage_bytes} bytes)."
                    ),
                },
            )

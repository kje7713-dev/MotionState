"""Job status routes."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from libs.db import get_db
from libs.models import Job

router = APIRouter()


@router.get("/{job_id}")
async def get_job(
    job_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return the current state of a processing job."""
    job = await db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "id": job.id,
        "video_id": job.video_id,
        "type": job.type,
        "status": job.status,
        "error": job.error,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }

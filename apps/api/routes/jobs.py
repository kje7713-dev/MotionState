"""Job status routes."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from libs.auth import get_current_project
from libs.db import get_db
from libs.models import Job, Project, Video

router = APIRouter()


@router.get("/{job_id}")
async def get_job(
    job_id: int,
    db: AsyncSession = Depends(get_db),
    current_project: Project = Depends(get_current_project),
) -> dict:
    """Return the current state of a processing job."""
    job = await db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # Scope job access to the authenticated project via video ownership.
    video = await db.get(Video, job.video_id)
    if video is None or video.project_id != current_project.id:
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

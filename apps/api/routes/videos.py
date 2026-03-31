"""Video upload and retrieval routes."""

import uuid
from pathlib import Path

import aiofiles
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.config import settings
from libs.db import get_db
from libs.models import Artifact, Job, JobStatus, JobType, Video, VideoStatus
from libs.queue import enqueue

router = APIRouter()


@router.post("", status_code=201)
async def upload_video(
    file: UploadFile,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Accept an uploaded video, persist it, and enqueue a processing job.

    Returns:
        ``{"video_id": int, "job_id": int}``
    """
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Use a UUID filename to avoid collisions while preserving the extension.
    suffix = Path(file.filename or "upload").suffix or ".mp4"
    dest_filename = f"{uuid.uuid4().hex}{suffix}"
    dest_path = upload_dir / dest_filename

    async with aiofiles.open(dest_path, "wb") as fh:
        await fh.write(await file.read())

    # Persist the Video row.
    video = Video(
        original_filename=file.filename or dest_filename,
        status=VideoStatus.pending,
        source_path=str(dest_path),
    )
    db.add(video)
    await db.flush()  # populate video.id before creating the Job

    # Persist the Job row.
    job = Job(
        video_id=video.id,
        type=JobType.process_video,
        status=JobStatus.queued,
    )
    db.add(job)
    await db.flush()

    # Enqueue the job in Redis.
    await enqueue(job.id, JobType.process_video.value, {"video_id": video.id})

    return {"video_id": video.id, "job_id": job.id}


@router.get("/{video_id}")
async def get_video(
    video_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return the current state of a video record."""
    video = await db.get(Video, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")

    return {
        "id": video.id,
        "original_filename": video.original_filename,
        "status": video.status,
        "source_path": video.source_path,
        "normalized_path": video.normalized_path,
        "duration_seconds": video.duration_seconds,
        "fps": video.fps,
        "width": video.width,
        "height": video.height,
        "created_at": video.created_at,
        "updated_at": video.updated_at,
    }


@router.get("/{video_id}/artifacts")
async def list_artifacts(
    video_id: int,
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    """Return all artifact records for a video.

    Returns a list of artifact objects, each with ``id``, ``type``,
    ``path``, ``metadata_json``, and ``created_at``.
    """
    video = await db.get(Video, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")

    result = await db.execute(select(Artifact).where(Artifact.video_id == video_id))
    artifacts = result.scalars().all()

    return [
        {
            "id": a.id,
            "video_id": a.video_id,
            "type": a.type,
            "path": a.path,
            "metadata_json": a.metadata_json,
            "created_at": a.created_at,
        }
        for a in artifacts
    ]

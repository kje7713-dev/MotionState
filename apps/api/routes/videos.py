"""Video upload and retrieval routes."""

import json
import logging
import uuid
from pathlib import Path

import aiofiles
from fastapi import APIRouter, Body, Depends, HTTPException, Query, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.artifacts import get_latest_artifact, read_artifact_json
from libs.auth import get_current_project
from libs.config import settings
from libs.db import get_db
from libs.events import RunEventType
from libs.models import (
    Artifact,
    Job,
    JobStatus,
    JobType,
    ProcessingRun,
    Project,
    RunStatus,
    TriggerType,
    UsageEventType,
    Video,
    VideoStatus,
)
from libs.queue import enqueue
from libs.quotas import check_quota
from libs.schemas import (
    DetectionsArtifact,
    FeaturesArtifact,
    PosesArtifact,
    SegmentsArtifact,
    StateArtifact,
    TracksArtifact,
)
from libs.storage import get_storage, source_video_key
from libs.usage import emit as emit_usage
from libs.webhooks import enqueue_run_event

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("", status_code=201)
async def upload_video(
    file: UploadFile,
    db: AsyncSession = Depends(get_db),
    current_project: Project = Depends(get_current_project),
) -> dict:
    """Accept an uploaded video, persist it, and enqueue a processing job.

    This is the simple multipart-upload path.  It works for both local
    development and hosted storage-backed deployments.  When the configured
    storage backend is S3/R2, the uploaded bytes are stored through the
    storage abstraction so that the worker (running in a separate container)
    can retrieve the file from shared object storage.

    For large files or direct-to-object-storage flows, use
    ``POST /videos/upload-init`` instead.

    Returns:
        ``{"video_id": int, "job_id": int, "processing_run_id": int}``
    """
    # Quota check before accepting the upload.
    await check_quota(db, current_project, videos_upload=True)

    # Read the file contents once so we can either write locally or upload to
    # object storage depending on the configured backend.
    contents = await file.read()
    suffix = Path(file.filename or "upload").suffix or ".mp4"
    original_filename = file.filename or f"upload{suffix}"

    # Persist the Video row first so we have a real video.id available for
    # generating the canonical storage key (mirrors the upload-init flow).
    video = Video(
        project_id=current_project.id,
        original_filename=original_filename,
        status=VideoStatus.pending,
        source_path=None,  # set below
    )
    db.add(video)
    await db.flush()  # populate video.id before creating the ProcessingRun

    if settings.storage_backend == "s3":
        # Store through the configured object-storage backend so the worker
        # (running in a separate container) can retrieve the file.
        storage = get_storage()
        key = source_video_key(video.id, ext=suffix)
        await storage.save(contents, key)
        video.source_path = key
    else:
        # Local backend: write to disk as before.
        upload_dir = Path(settings.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        dest_filename = f"{uuid.uuid4().hex}{suffix}"
        dest_path = upload_dir / dest_filename
        async with aiofiles.open(dest_path, "wb") as fh:
            await fh.write(contents)
        video.source_path = str(dest_path)

    # Create a ProcessingRun for lineage tracking.
    run = ProcessingRun(
        video_id=video.id,
        status=RunStatus.pending,
        trigger_type=TriggerType.initial,
        detector_backend=settings.detector_backend,
        tracker_backend=settings.tracker_backend,
        pose_backend=settings.pose_backend,
        storage_backend=settings.storage_backend,
    )
    db.add(run)
    await db.flush()  # populate run.id before creating the Job

    # Persist the Job row.
    job = Job(
        video_id=video.id,
        processing_run_id=run.id,
        type=JobType.process_video,
        status=JobStatus.queued,
    )
    db.add(job)
    await db.flush()

    # Record upload event.
    await emit_usage(
        db,
        project_id=current_project.id,
        event_type=UsageEventType.videos_uploaded,
        quantity=1,
        processing_run_id=run.id,
    )

    # Commit all rows so the worker can find them immediately after dequeue.
    await db.commit()

    # Enqueue the job in Redis.
    await enqueue(job.id, JobType.process_video.value, {"video_id": video.id})

    # Emit created event for downstream webhook subscribers (non-blocking).
    try:
        await enqueue_run_event(
            db,
            RunEventType.created,
            project_id=current_project.id,
            video_id=video.id,
            processing_run_id=run.id,
            status=RunStatus.pending,
        )
    except Exception:
        logger.exception("Failed to enqueue created event for run %s", run.id)

    return {"video_id": video.id, "job_id": job.id, "processing_run_id": run.id}


@router.post("/upload-init", status_code=201)
async def upload_init(
    filename: str = Body(..., embed=True),
    db: AsyncSession = Depends(get_db),
    current_project: Project = Depends(get_current_project),
) -> dict:
    """Prepare a direct-to-storage upload and return a pre-signed upload URL.

    This is the production-oriented upload path.  Instead of streaming the
    video through the API server, the client uploads the file directly to
    the configured object storage backend (S3 / R2) using the returned URL.

    Steps performed by this endpoint:

    1. Create a ``pending`` :class:`~libs.models.Video` row.
    2. Generate a canonical storage key for the source video.
    3. For the S3 backend: generate a pre-signed PUT URL the client can use
       to upload directly to object storage.
    4. Return the ``video_id``, ``upload_url``, and ``storage_key`` to the
       client.

    After the client has finished uploading, it should trigger processing
    via the regular job queue (a separate endpoint or operation).

    Args:
        filename: Original filename of the video being uploaded.  Used to
            derive the file extension for the canonical storage key.

    Returns:
        ``{"video_id": int, "upload_url": str | None, "storage_key": str}``

        ``upload_url`` is ``None`` when the storage backend does not support
        pre-signed uploads (i.e. the local backend).  In that case fall back
        to ``POST /videos`` for direct upload through the API server.
    """
    suffix = Path(filename).suffix or ".mp4"
    storage = get_storage()

    # Quota check before creating the video row.
    await check_quota(db, current_project, videos_upload=True)

    # Create a pending video row first so we have a real video_id for the key.
    video = Video(
        project_id=current_project.id,
        original_filename=filename,
        status=VideoStatus.pending,
        source_path=None,  # will be set after upload completes
    )
    db.add(video)
    await db.flush()  # populate video.id

    key = source_video_key(video.id, ext=suffix)

    # For S3 backend: generate a pre-signed upload URL.
    # For local backend: returns None (client should use POST /videos instead).
    upload_url = await storage.generate_upload_url(
        key, expires_in=settings.signed_url_expiration_seconds
    )

    # Store the canonical key as the source path so the worker can locate
    # the uploaded file via the storage backend.
    video.source_path = key

    # Record upload init event.
    await emit_usage(
        db,
        project_id=current_project.id,
        event_type=UsageEventType.videos_uploaded,
        quantity=1,
    )

    await db.commit()

    return {
        "video_id": video.id,
        "upload_url": upload_url,
        "storage_key": key,
    }


@router.get("/{video_id}")
async def get_video(
    video_id: int,
    db: AsyncSession = Depends(get_db),
    current_project: Project = Depends(get_current_project),
) -> dict:
    """Return the current state of a video record."""
    video = await db.get(Video, video_id)
    if video is None or video.project_id != current_project.id:
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
    current_project: Project = Depends(get_current_project),
) -> list[dict]:
    """Return all artifact records for a video.

    Returns a list of artifact objects, each with ``id``, ``type``,
    ``path``, ``metadata_json``, and ``created_at``.
    """
    video = await db.get(Video, video_id)
    if video is None or video.project_id != current_project.id:
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


@router.get("/{video_id}/timeline")
async def get_timeline(
    video_id: int,
    run_id: int | None = Query(None, description="Return artifacts from this specific run"),
    db: AsyncSession = Depends(get_db),
    current_project: Project = Depends(get_current_project),
) -> dict:
    """Return the parsed timeline manifest for a video.

    The manifest ties all pipeline artifacts together with a per-segment
    timeline that references clip paths and related artifact files.

    If *run_id* is omitted the latest successful run is used.

    Returns:
        The parsed ``timeline_manifest.json`` contents.

    Raises:
        HTTPException 404: if the video, the manifest artifact row, or the
            manifest file on disk / object key cannot be found.
    """
    video = await db.get(Video, video_id)
    if video is None or video.project_id != current_project.id:
        raise HTTPException(status_code=404, detail="Video not found")

    artifact = await get_latest_artifact(db, video_id, "timeline_manifest", run_id=run_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="Timeline manifest not found")

    if settings.storage_backend == "s3":
        storage = get_storage()
        try:
            data = await storage.load(artifact.path)
        except Exception as exc:
            # Distinguish object-not-found from other storage errors.
            http_response = getattr(exc, "response", None)
            if http_response is not None:
                code = http_response.get("Error", {}).get("Code", "")
                if code in ("404", "NoSuchKey"):
                    raise HTTPException(
                        status_code=404, detail="Timeline manifest object not found"
                    ) from exc
            raise HTTPException(
                status_code=500, detail="Storage backend error while reading timeline manifest"
            ) from exc
        try:
            return json.loads(data)
        except ValueError as exc:
            raise HTTPException(
                status_code=500, detail="Timeline manifest is not valid JSON"
            ) from exc

    # Local backend: read directly from disk.
    manifest_file = Path(artifact.path)
    if not manifest_file.exists():
        raise HTTPException(status_code=404, detail="Timeline manifest file not found")

    async with aiofiles.open(manifest_file) as fh:
        content = await fh.read()
    return json.loads(content)


# ---------------------------------------------------------------------------
# Artifact-type read helpers
# ---------------------------------------------------------------------------


async def _get_artifact_content(
    db: AsyncSession,
    video_id: int,
    artifact_type: str,
    not_found_detail: str,
    current_project: Project,
    run_id: int | None = None,
) -> dict:
    """Shared implementation for single-artifact read endpoints.

    Raises ``HTTPException`` 404 for missing video, cross-project access,
    missing artifact row, path outside ``settings.artifacts_dir`` (local
    backend), or missing file / key.
    """
    video = await db.get(Video, video_id)
    if video is None or video.project_id != current_project.id:
        raise HTTPException(status_code=404, detail="Video not found")

    artifact = await get_latest_artifact(db, video_id, artifact_type, run_id=run_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail=not_found_detail)

    try:
        return await read_artifact_json(artifact.path)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Artifact path is invalid") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Artifact file not found") from exc


@router.get("/{video_id}/state", response_model=StateArtifact)
async def get_state(
    video_id: int,
    run_id: int | None = Query(None, description="Return artifacts from this specific run"),
    db: AsyncSession = Depends(get_db),
    current_project: Project = Depends(get_current_project),
) -> StateArtifact:
    """Return the latest ``state.json`` artifact for a video.

    If *run_id* is omitted the latest successful run is used.

    Raises:
        HTTPException 404: if the video, artifact row, or file is missing.
    """
    data = await _get_artifact_content(
        db, video_id, "state", "State artifact not found", current_project, run_id=run_id
    )
    return StateArtifact(**data)


@router.get("/{video_id}/detections", response_model=DetectionsArtifact)
async def get_detections(
    video_id: int,
    run_id: int | None = Query(None, description="Return artifacts from this specific run"),
    db: AsyncSession = Depends(get_db),
    current_project: Project = Depends(get_current_project),
) -> DetectionsArtifact:
    """Return the latest ``detections.json`` artifact for a video.

    If *run_id* is omitted the latest successful run is used.

    Raises:
        HTTPException 404: if the video, artifact row, or file is missing.
    """
    data = await _get_artifact_content(
        db, video_id, "detections", "Detections artifact not found", current_project, run_id=run_id
    )
    return DetectionsArtifact(**data)


@router.get("/{video_id}/tracks", response_model=TracksArtifact)
async def get_tracks(
    video_id: int,
    run_id: int | None = Query(None, description="Return artifacts from this specific run"),
    db: AsyncSession = Depends(get_db),
    current_project: Project = Depends(get_current_project),
) -> TracksArtifact:
    """Return the latest ``tracks.json`` artifact for a video.

    If *run_id* is omitted the latest successful run is used.

    Raises:
        HTTPException 404: if the video, artifact row, or file is missing.
    """
    data = await _get_artifact_content(
        db, video_id, "tracks", "Tracks artifact not found", current_project, run_id=run_id
    )
    return TracksArtifact(**data)


@router.get("/{video_id}/poses", response_model=PosesArtifact)
async def get_poses(
    video_id: int,
    run_id: int | None = Query(None, description="Return artifacts from this specific run"),
    db: AsyncSession = Depends(get_db),
    current_project: Project = Depends(get_current_project),
) -> PosesArtifact:
    """Return the latest ``poses.json`` artifact for a video.

    If *run_id* is omitted the latest successful run is used.

    Raises:
        HTTPException 404: if the video, artifact row, or file is missing.
    """
    data = await _get_artifact_content(
        db, video_id, "poses", "Poses artifact not found", current_project, run_id=run_id
    )
    return PosesArtifact(**data)


@router.get("/{video_id}/features", response_model=FeaturesArtifact)
async def get_features(
    video_id: int,
    run_id: int | None = Query(None, description="Return artifacts from this specific run"),
    db: AsyncSession = Depends(get_db),
    current_project: Project = Depends(get_current_project),
) -> FeaturesArtifact:
    """Return the latest ``features.json`` artifact for a video.

    If *run_id* is omitted the latest successful run is used.

    Raises:
        HTTPException 404: if the video, artifact row, or file is missing.
    """
    data = await _get_artifact_content(
        db, video_id, "features", "Features artifact not found", current_project, run_id=run_id
    )
    return FeaturesArtifact(**data)


@router.get("/{video_id}/segments", response_model=SegmentsArtifact)
async def get_segments(
    video_id: int,
    run_id: int | None = Query(None, description="Return artifacts from this specific run"),
    db: AsyncSession = Depends(get_db),
    current_project: Project = Depends(get_current_project),
) -> SegmentsArtifact:
    """Return the latest ``segments.json`` artifact for a video.

    If *run_id* is omitted the latest successful run is used.

    Raises:
        HTTPException 404: if the video, artifact row, or file is missing.
    """
    data = await _get_artifact_content(
        db, video_id, "segments", "Segments artifact not found", current_project, run_id=run_id
    )
    return SegmentsArtifact(**data)


# ---------------------------------------------------------------------------
# Processing-run endpoints
# ---------------------------------------------------------------------------


@router.get("/{video_id}/runs")
async def list_runs(
    video_id: int,
    db: AsyncSession = Depends(get_db),
    current_project: Project = Depends(get_current_project),
) -> list[dict]:
    """Return all processing runs for a video, newest first.

    Each entry contains:
    - ``id``
    - ``status``
    - ``trigger_type``
    - ``pipeline_version``
    - ``created_at``
    - ``completed_at``
    - ``error``

    Raises:
        HTTPException 404: if the video does not exist.
    """
    video = await db.get(Video, video_id)
    if video is None or video.project_id != current_project.id:
        raise HTTPException(status_code=404, detail="Video not found")

    result = await db.execute(
        select(ProcessingRun)
        .where(ProcessingRun.video_id == video_id)
        .order_by(ProcessingRun.id.desc())
    )
    runs = result.scalars().all()

    return [
        {
            "id": r.id,
            "status": r.status,
            "trigger_type": r.trigger_type,
            "pipeline_version": r.pipeline_version,
            "created_at": r.created_at,
            "completed_at": r.completed_at,
            "error": r.error,
        }
        for r in runs
    ]


@router.post("/{video_id}/reprocess", status_code=201)
async def reprocess_video(
    video_id: int,
    db: AsyncSession = Depends(get_db),
    current_project: Project = Depends(get_current_project),
) -> dict:
    """Create a new processing run and enqueue a reprocessing job for a video.

    This endpoint is idempotent with respect to lineage: it always creates a
    **new** :class:`~libs.models.ProcessingRun` row so that artifacts from
    previous runs are never overwritten.

    Returns:
        ``{"video_id": int, "processing_run_id": int, "job_id": int}``

    Raises:
        HTTPException 404: if the video does not exist.
    """
    video = await db.get(Video, video_id)
    if video is None or video.project_id != current_project.id:
        raise HTTPException(status_code=404, detail="Video not found")

    # Quota check before allowing reprocess.
    await check_quota(db, current_project)

    # Create a new run for lineage tracking.
    run = ProcessingRun(
        video_id=video_id,
        status=RunStatus.pending,
        trigger_type=TriggerType.reprocess,
        detector_backend=settings.detector_backend,
        tracker_backend=settings.tracker_backend,
        pose_backend=settings.pose_backend,
        storage_backend=settings.storage_backend,
    )
    db.add(run)
    await db.flush()  # populate run.id

    # Create a new job linked to this run.
    job = Job(
        video_id=video_id,
        processing_run_id=run.id,
        type=JobType.process_video,
        status=JobStatus.queued,
    )
    db.add(job)
    await db.flush()  # populate job.id

    # Commit all rows so the worker can find them immediately after dequeue.
    await db.commit()

    await enqueue(job.id, JobType.process_video.value, {"video_id": video_id})

    # Emit created event for downstream webhook subscribers (non-blocking).
    try:
        await enqueue_run_event(
            db,
            RunEventType.created,
            project_id=current_project.id,
            video_id=video_id,
            processing_run_id=run.id,
            status=RunStatus.pending,
        )
    except Exception:
        logger.exception("Failed to enqueue created event for run %s", run.id)

    return {"video_id": video_id, "processing_run_id": run.id, "job_id": job.id}


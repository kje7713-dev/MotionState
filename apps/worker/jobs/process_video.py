"""process_video job handler.

Steps:
1. Mark job as running.
2. Normalize video with FFmpeg.
3. Extract metadata via ffprobe.
4. Write placeholder state artifact.
5. Create an Artifact row.
6. Mark job + video as done.
"""

import json
import logging
from pathlib import Path

from libs.config import settings
from libs.db import AsyncSessionLocal
from libs.models import Artifact, Job, JobStatus, Video, VideoStatus
from libs.pipeline.run_pipeline import run_pipeline
from libs.video.ffmpeg import normalize_video, probe_video

logger = logging.getLogger(__name__)


async def handle_process_video(message: dict) -> None:
    """Process a single ``process_video`` job message."""
    job_id: int = message["job_id"]
    video_id: int = message["payload"]["video_id"]

    async with AsyncSessionLocal() as db:
        job = await db.get(Job, job_id)
        video = await db.get(Video, video_id)

        if job is None or video is None:
            logger.error("Job %s or Video %s not found; skipping.", job_id, video_id)
            return

        # --- Mark as running ---
        job.status = JobStatus.running
        video.status = VideoStatus.processing
        await db.commit()

        try:
            source = video.source_path
            if not source or not Path(source).exists():
                raise FileNotFoundError(f"Source video not found: {source}")

            # --- Normalize ---
            norm_dir = Path(settings.normalized_dir)
            norm_dir.mkdir(parents=True, exist_ok=True)
            norm_path = norm_dir / f"{video_id}_normalized.mp4"
            normalize_video(source, str(norm_path))
            logger.info("Normalized video written to %s", norm_path)

            # --- Probe metadata ---
            meta = probe_video(str(norm_path))
            logger.info("Probe result for video %s: %s", video_id, meta)

            # --- Placeholder state artifact ---
            artifact_dir = Path(settings.artifacts_dir) / str(video_id)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = artifact_dir / "state.json"

            state = run_pipeline(video_id)
            artifact_path.write_text(json.dumps(state, indent=2))
            logger.info("State artifact written to %s", artifact_path)

            # --- Persist artifact row ---
            artifact = Artifact(
                video_id=video_id,
                type="state",
                path=str(artifact_path),
                metadata_json={"version": 1},
            )
            db.add(artifact)

            # --- Update video ---
            video.normalized_path = str(norm_path)
            video.duration_seconds = meta["duration_seconds"]
            video.fps = meta["fps"]
            video.width = meta["width"]
            video.height = meta["height"]
            video.status = VideoStatus.ready

            # --- Update job ---
            job.status = JobStatus.done

            await db.commit()
            logger.info("Job %s completed successfully.", job_id)

        except Exception as exc:
            logger.exception("Error processing job %s", job_id)
            job.status = JobStatus.error
            job.error = str(exc)
            video.status = VideoStatus.error
            await db.commit()
            raise

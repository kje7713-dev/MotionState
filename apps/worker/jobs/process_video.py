"""process_video job handler.

Steps:
1. Mark job as running.
2. Normalize video with FFmpeg.
3. Extract metadata via ffprobe.
4. Extract frames at configured sample rate.
5. Run CV pipeline (detector + tracker + pose estimator + stubs) to produce
   detections, tracks, and pose estimates.
6. Write state.json, detections.json, tracks.json, and poses.json artifacts.
7. Create Artifact rows for all four files.
8. Mark job + video as done.
"""

import json
import logging
from pathlib import Path

from libs.config import settings
from libs.db import AsyncSessionLocal
from libs.models import Artifact, Job, JobStatus, Video, VideoStatus
from libs.pipeline.run_pipeline import run_pipeline
from libs.video.ffmpeg import normalize_video, probe_video
from libs.video.frames import extract_frames

logger = logging.getLogger(__name__)


def _build_detector():
    """Return a real YoloDetector when enabled by config, else StubDetector."""
    if settings.detector_backend == "yolo":
        try:
            from libs.pipeline.detector_yolo import YoloDetector

            return YoloDetector(model_name=settings.detector_model)
        except ImportError:
            logger.warning(
                "YoloDetector requested but 'ultralytics' is not installed; "
                "falling back to StubDetector."
            )
    from libs.pipeline.detector import StubDetector

    return StubDetector()


def _build_tracker():
    """Return an IOUTracker when enabled by config, else StubTracker."""
    if settings.tracker_backend == "iou":
        from libs.pipeline.tracker_bytetrack import IOUTracker

        return IOUTracker(
            iou_threshold=settings.tracker_iou_threshold,
            max_age=settings.tracker_max_age,
        )
    from libs.pipeline.tracker import StubTracker

    return StubTracker()


def _build_pose_estimator():
    """Return a MediaPipePoseEstimator when enabled by config, else StubPoseEstimator."""
    if settings.pose_backend == "mediapipe":
        try:
            from libs.pipeline.pose_mediapipe import MediaPipePoseEstimator

            return MediaPipePoseEstimator(min_confidence=settings.pose_min_confidence)
        except ImportError:
            logger.warning(
                "MediaPipePoseEstimator requested but 'mediapipe' is not installed; "
                "falling back to StubPoseEstimator."
            )
    from libs.pipeline.pose import StubPoseEstimator

    return StubPoseEstimator()


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

            # --- Extract frames ---
            artifact_dir = Path(settings.artifacts_dir) / str(video_id)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            frames_dir = artifact_dir / "frames"
            frames = extract_frames(
                norm_path,
                frames_dir,
                sample_fps=settings.frame_sample_fps,
            )
            logger.info("Extracted %d frames for video %s", len(frames), video_id)

            # --- Run CV pipeline ---
            detector = _build_detector()
            tracker = _build_tracker()
            pose_estimator = _build_pose_estimator()
            state, detections, tracks, poses = run_pipeline(
                video_id,
                frames=frames,
                detector=detector,
                tracker=tracker,
                pose_estimator=pose_estimator,
                sample_fps=settings.frame_sample_fps,
            )

            # --- Write state artifact ---
            state_path = artifact_dir / "state.json"
            state_path.write_text(json.dumps(state, indent=2))
            logger.info("State artifact written to %s", state_path)

            # --- Write detections artifact ---
            detections_path = artifact_dir / "detections.json"
            detections_path.write_text(json.dumps(detections, indent=2))
            logger.info("Detections artifact written to %s", detections_path)

            # --- Write tracks artifact ---
            tracks_path = artifact_dir / "tracks.json"
            tracks_path.write_text(json.dumps(tracks, indent=2))
            logger.info("Tracks artifact written to %s", tracks_path)

            # --- Write poses artifact ---
            poses_path = artifact_dir / "poses.json"
            poses_path.write_text(json.dumps(poses, indent=2))
            logger.info("Poses artifact written to %s", poses_path)

            # --- Persist artifact rows ---
            db.add(
                Artifact(
                    video_id=video_id,
                    type="state",
                    path=str(state_path),
                    metadata_json={"version": state["version"]},
                )
            )
            db.add(
                Artifact(
                    video_id=video_id,
                    type="detections",
                    path=str(detections_path),
                    metadata_json={
                        "version": detections["version"],
                        "sample_fps": settings.frame_sample_fps,
                        "frame_count": len(frames),
                    },
                )
            )
            db.add(
                Artifact(
                    video_id=video_id,
                    type="tracks",
                    path=str(tracks_path),
                    metadata_json={
                        "version": tracks["version"],
                        "track_count": tracks["track_count"],
                    },
                )
            )
            db.add(
                Artifact(
                    video_id=video_id,
                    type="poses",
                    path=str(poses_path),
                    metadata_json={
                        "version": poses["version"],
                        "pose_count": poses["pose_count"],
                    },
                )
            )

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

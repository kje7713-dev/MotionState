"""process_video job handler.

Steps:
1. Mark job as running.
2. Normalize video with FFmpeg.
3. Extract metadata via ffprobe.
4. Extract frames at configured sample rate.
5. Run CV pipeline (detector + tracker + pose estimator + feature deriver + segmenter) to produce
   detections, tracks, pose estimates, motion features, and temporal segments.
6. Generate MP4 clips for each segment.
7. Write timeline_manifest.json tying all artifacts together.
8. Write state.json, detections.json, tracks.json, poses.json, features.json, segments.json,
   clip files, and timeline_manifest.json artifacts through the configured storage backend.
9. Create Artifact rows for all files, linked to the ProcessingRun.
10. Mark job + video + ProcessingRun as done.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from libs.config import settings
from libs.db import AsyncSessionLocal
from libs.events import RunEventType
from libs.models import (
    Artifact,
    Job,
    JobStatus,
    ProcessingRun,
    RunStatus,
    UsageEventType,
    Video,
    VideoStatus,
)
from libs.pipeline.run_pipeline import run_pipeline
from libs.storage import artifact_key, get_storage, normalized_video_key
from libs.usage import emit as emit_usage
from libs.video.clips import generate_clips
from libs.video.ffmpeg import normalize_video, probe_video
from libs.video.frames import extract_frames
from libs.webhooks import enqueue_run_event

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


async def _save_json(storage, key: str, data: dict) -> tuple[str, int]:
    """Serialize *data* to JSON bytes and persist via *storage*.

    Returns:
        A ``(canonical_path, byte_count)`` tuple.  The byte count can be used
        to record a ``storage_bytes_written`` usage event.
    """
    payload = json.dumps(data, indent=2).encode()
    saved = await storage.save(payload, key)
    return saved, len(payload)


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

        # Resolve the ProcessingRun for this job (may be None for legacy jobs).
        run: ProcessingRun | None = None
        if job.processing_run_id is not None:
            run = await db.get(ProcessingRun, job.processing_run_id)

        # --- Mark as running ---
        job.status = JobStatus.running
        video.status = VideoStatus.processing
        if run is not None:
            run.status = RunStatus.running
            run.started_at = datetime.now(UTC)
        await db.commit()

        # Emit running event (non-blocking; errors are swallowed).
        if run is not None and video.project_id is not None:
            try:
                await enqueue_run_event(
                    db,
                    RunEventType.running,
                    project_id=video.project_id,
                    video_id=video_id,
                    processing_run_id=run.id,
                    status=RunStatus.running,
                )
            except Exception:
                logger.exception(
                    "Failed to enqueue running event for run %s", run.id if run else None
                )

        try:
            source = video.source_path
            if not source or not Path(source).exists():
                raise FileNotFoundError(f"Source video not found: {source}")

            storage = get_storage(settings)

            # --- Normalize ---
            norm_dir = Path(settings.normalized_dir)
            norm_dir.mkdir(parents=True, exist_ok=True)
            norm_path = norm_dir / f"{video_id}_normalized.mp4"
            normalize_video(source, str(norm_path))
            logger.info("Normalized video written to %s", norm_path)

            # --- Probe metadata ---
            meta = probe_video(str(norm_path))
            logger.info("Probe result for video %s: %s", video_id, meta)

            # --- Extract frames (always local; frames are transient) ---
            # Frames are intermediate and kept on local disk for pipeline speed.
            # Only the final JSON artifacts and clips are persisted via storage.
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
            state, detections, tracks, poses, features, segments = run_pipeline(
                video_id,
                frames=frames,
                detector=detector,
                tracker=tracker,
                pose_estimator=pose_estimator,
                sample_fps=settings.frame_sample_fps,
            )

            # --- Write detections artifact ---
            det_key = artifact_key(video_id, "detections.json")
            det_saved, det_bytes = await _save_json(storage, det_key, detections)
            logger.info("Detections artifact written to %s", det_saved)

            # --- Write tracks artifact ---
            trk_key = artifact_key(video_id, "tracks.json")
            trk_saved, trk_bytes = await _save_json(storage, trk_key, tracks)
            logger.info("Tracks artifact written to %s", trk_saved)

            # --- Write poses artifact ---
            poses_key = artifact_key(video_id, "poses.json")
            poses_saved, poses_bytes = await _save_json(storage, poses_key, poses)
            logger.info("Poses artifact written to %s", poses_saved)

            # --- Write features artifact ---
            feat_key = artifact_key(video_id, "features.json")
            feat_saved, feat_bytes = await _save_json(storage, feat_key, features)
            logger.info("Features artifact written to %s", feat_saved)

            # --- Write segments artifact ---
            seg_key = artifact_key(video_id, "segments.json")
            seg_saved, seg_bytes = await _save_json(storage, seg_key, segments)
            logger.info("Segments artifact written to %s", seg_saved)

            # --- Generate clips (local temp dir; then upload via storage) ---
            clips_dir = artifact_dir / "clips"
            clips_info = generate_clips(norm_path, segments["segments"], clips_dir)
            logger.info("Generated %d clips for video %s", len(clips_info), video_id)

            # Upload each clip through the storage backend and record the
            # canonical path returned by the backend.
            clip_saved_paths: list[str] = []
            clip_total_bytes: int = 0
            for clip in clips_info:
                clip_local = Path(clip["path"])
                clip_data = clip_local.read_bytes()
                clip_rel_key = artifact_key(
                    video_id, f"clips/{clip_local.name}"
                )
                clip_saved = await storage.save(clip_data, clip_rel_key)
                clip_saved_paths.append(clip_saved)
                clip_total_bytes += len(clip_data)

            # --- Build timeline manifest ---
            # Use the canonical paths returned by the storage backend so that
            # the manifest references the correct locations in all backends.
            total_clip_duration_ms = sum(c["end_ms"] - c["start_ms"] for c in clips_info)
            manifest_key = artifact_key(video_id, "timeline_manifest.json")
            confidence_by_index = {
                i: s.get("confidence", 1.0) for i, s in enumerate(segments["segments"])
            }
            manifest = {
                "video_id": str(video_id),
                "version": 1,
                "duration_seconds": meta["duration_seconds"],
                "artifacts": {
                    "state": artifact_key(video_id, "state.json"),
                    "detections": det_saved,
                    "tracks": trk_saved,
                    "poses": poses_saved,
                    "features": feat_saved,
                    "segments": seg_saved,
                },
                "timeline": [
                    {
                        "segment_index": seg["segment_index"],
                        "start_ms": seg["start_ms"],
                        "end_ms": seg["end_ms"],
                        "label": seg["label"],
                        "confidence": confidence_by_index.get(seg["segment_index"], 1.0),
                        "clip_path": clip_saved_paths[i],
                        "related_artifacts": {
                            "segments": seg_saved,
                            "features": feat_saved,
                        },
                    }
                    for i, seg in enumerate(clips_info)
                ],
            }
            manifest_saved, manifest_bytes = await _save_json(storage, manifest_key, manifest)
            logger.info("Timeline manifest written to %s", manifest_saved)

            # --- Update state with clip summary and manifest path ---
            state["clip_summary"] = {
                "clip_count": len(clips_info),
                "total_clip_duration_ms": total_clip_duration_ms,
            }
            state["manifest_path"] = manifest_saved
            state_key = artifact_key(video_id, "state.json")
            state_saved, state_bytes = await _save_json(storage, state_key, state)
            logger.info("State artifact written to %s", state_saved)

            # Helper to build Artifact kwargs with optional run linkage.
            run_id = run.id if run is not None else None

            # --- Persist artifact rows ---
            db.add(
                Artifact(
                    video_id=video_id,
                    processing_run_id=run_id,
                    type="state",
                    path=state_saved,
                    metadata_json={"version": state["version"]},
                )
            )
            db.add(
                Artifact(
                    video_id=video_id,
                    processing_run_id=run_id,
                    type="detections",
                    path=det_saved,
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
                    processing_run_id=run_id,
                    type="tracks",
                    path=trk_saved,
                    metadata_json={
                        "version": tracks["version"],
                        "track_count": tracks["track_count"],
                    },
                )
            )
            db.add(
                Artifact(
                    video_id=video_id,
                    processing_run_id=run_id,
                    type="poses",
                    path=poses_saved,
                    metadata_json={
                        "version": poses["version"],
                        "pose_count": poses["pose_count"],
                    },
                )
            )
            db.add(
                Artifact(
                    video_id=video_id,
                    processing_run_id=run_id,
                    type="features",
                    path=feat_saved,
                    metadata_json={
                        "version": features["version"],
                        "feature_count": features["feature_count"],
                    },
                )
            )
            db.add(
                Artifact(
                    video_id=video_id,
                    processing_run_id=run_id,
                    type="segments",
                    path=seg_saved,
                    metadata_json={
                        "version": segments["version"],
                        "segment_count": segments["segment_count"],
                    },
                )
            )
            for i, clip in enumerate(clips_info):
                db.add(
                    Artifact(
                        video_id=video_id,
                        processing_run_id=run_id,
                        type="segment_clip",
                        path=clip_saved_paths[i],
                        metadata_json={
                            "segment_index": clip["segment_index"],
                            "label": clip["label"],
                            "start_ms": clip["start_ms"],
                            "end_ms": clip["end_ms"],
                        },
                    )
                )
            db.add(
                Artifact(
                    video_id=video_id,
                    processing_run_id=run_id,
                    type="timeline_manifest",
                    path=manifest_saved,
                    metadata_json={"version": manifest["version"]},
                )
            )

            # --- Update video ---
            video.normalized_path = normalized_video_key(video_id)
            video.duration_seconds = meta["duration_seconds"]
            video.fps = meta["fps"]
            video.width = meta["width"]
            video.height = meta["height"]
            video.status = VideoStatus.ready

            # --- Update job ---
            job.status = JobStatus.done

            # --- Update ProcessingRun ---
            if run is not None:
                run.status = RunStatus.completed
                run.completed_at = datetime.now(UTC)
                run.pipeline_version = str(state.get("version", ""))

            # --- Emit usage events (if project is set) ---
            if video.project_id is not None:
                total_json_bytes = (
                    det_bytes + trk_bytes + poses_bytes
                    + feat_bytes + seg_bytes + manifest_bytes + state_bytes
                )
                total_storage_bytes = total_json_bytes + clip_total_bytes

                await emit_usage(
                    db,
                    project_id=video.project_id,
                    event_type=UsageEventType.video_seconds_processed,
                    quantity=max(1, int(meta["duration_seconds"])),
                    processing_run_id=run_id,
                    metadata={"video_id": video_id},
                )
                await emit_usage(
                    db,
                    project_id=video.project_id,
                    event_type=UsageEventType.frames_extracted,
                    quantity=len(frames),
                    processing_run_id=run_id,
                    metadata={"video_id": video_id},
                )
                if clips_info:
                    await emit_usage(
                        db,
                        project_id=video.project_id,
                        event_type=UsageEventType.clips_generated,
                        quantity=len(clips_info),
                        processing_run_id=run_id,
                        metadata={"video_id": video_id},
                    )
                await emit_usage(
                    db,
                    project_id=video.project_id,
                    event_type=UsageEventType.storage_bytes_written,
                    quantity=total_storage_bytes,
                    processing_run_id=run_id,
                    metadata={
                        "video_id": video_id,
                        "json_bytes": total_json_bytes,
                        "clip_bytes": clip_total_bytes,
                    },
                )

            await db.commit()
            logger.info("Job %s completed successfully.", job_id)

            # Emit completed event (non-blocking; errors are swallowed).
            if run is not None and video.project_id is not None:
                artifact_types = [
                    "state", "detections", "tracks", "poses",
                    "features", "segments", "timeline_manifest",
                ]
                if clips_info:
                    artifact_types.append("segment_clip")
                try:
                    await enqueue_run_event(
                        db,
                        RunEventType.completed,
                        project_id=video.project_id,
                        video_id=video_id,
                        processing_run_id=run.id,
                        status=RunStatus.completed,
                        artifact_types=artifact_types,
                    )
                except Exception:
                    logger.exception(
                        "Failed to enqueue completed event for run %s", run.id
                    )

        except Exception as exc:
            logger.exception("Error processing job %s", job_id)
            job.status = JobStatus.error
            job.error = str(exc)
            video.status = VideoStatus.error
            if run is not None:
                run.status = RunStatus.error
                run.error = str(exc)
                run.completed_at = datetime.now(UTC)
            await db.commit()

            # Emit failed event (non-blocking; errors are swallowed).
            if run is not None and video.project_id is not None:
                try:
                    await enqueue_run_event(
                        db,
                        RunEventType.failed,
                        project_id=video.project_id,
                        video_id=video_id,
                        processing_run_id=run.id,
                        status=RunStatus.error,
                        error=str(exc),
                    )
                except Exception:
                    logger.exception(
                        "Failed to enqueue failed event for run %s", run.id if run else None
                    )
            raise





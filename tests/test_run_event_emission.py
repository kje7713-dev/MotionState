"""Tests for run lifecycle event emission.

Covers:
- worker emits processing_run.running when it starts
- worker emits processing_run.completed with artifact_types on success
- worker emits processing_run.failed with error on failure
- API emits processing_run.created when a run is created
- completed run sends artifact-aware payload
- failed run sends error payload
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from libs.events import RunEventType
from libs.models import ProcessingRun, RunStatus

# ---------------------------------------------------------------------------
# Helpers shared with test_processing_runs.py
# ---------------------------------------------------------------------------


def _make_fake_state(video_id: str = "1") -> dict:
    return {
        "video_id": video_id,
        "version": 7,
        "segments": [],
        "tracks": [],
        "features": [],
        "detections_summary": {"frame_count": 0, "frames_with_people": 0, "total_detections": 0},
        "tracking_summary": {
            "track_count": 0,
            "tracked_frame_count": 0,
            "average_detections_per_frame": 0.0,
        },
        "pose_summary": {
            "pose_count": 0,
            "posed_track_count": 0,
            "average_keypoints_per_pose": 0.0,
        },
        "feature_summary": {"feature_count": 0, "featured_track_count": 0, "feature_names": []},
        "segmentation_summary": {
            "segment_count": 0,
            "segment_labels": [],
            "total_segment_duration_ms": 0,
        },
        "clip_summary": {"clip_count": 0, "total_clip_duration_ms": 0},
        "manifest_path": "",
        "notes": "stub",
    }


def _make_fake_detections() -> dict:
    return {"video_id": "1", "version": 1, "sample_fps": 2.0, "frames": []}


def _make_fake_tracks() -> dict:
    return {"video_id": "1", "version": 1, "track_count": 0, "tracks": []}


def _make_fake_poses() -> dict:
    return {"video_id": "1", "version": 1, "pose_count": 0, "poses": []}


def _make_fake_features() -> dict:
    return {"video_id": "1", "version": 1, "feature_count": 0, "features": []}


def _make_fake_segments() -> dict:
    return {"video_id": "1", "version": 1, "segment_count": 0, "segments": []}


def _make_mock_db(tmp_path, *, video_id: int = 1, job_id: int = 1, run_id: int = 10,
                  project_id: int = 5):
    """Return a mock async DB session with a fake ProcessingRun attached."""
    fake_video = MagicMock()
    fake_video.id = video_id
    fake_video.project_id = project_id
    fake_video.source_path = str(tmp_path / "input.mp4")
    fake_video.status = None

    (tmp_path / "input.mp4").write_bytes(b"fake")

    fake_run = MagicMock(spec=ProcessingRun)
    fake_run.id = run_id
    fake_run.status = RunStatus.pending
    fake_run.error = None
    fake_run.started_at = None
    fake_run.completed_at = None
    fake_run.pipeline_version = None

    fake_job = MagicMock()
    fake_job.id = job_id
    fake_job.status = None
    fake_job.error = None
    fake_job.processing_run_id = run_id

    def _get_side_effect(model, pk):
        name = model.__name__
        if name == "Job":
            return fake_job
        if name == "ProcessingRun":
            return fake_run
        return fake_video

    mock_db = AsyncMock()
    mock_db.get = AsyncMock(side_effect=_get_side_effect)
    mock_db.add = MagicMock()
    mock_db.commit = AsyncMock()
    mock_db.__aenter__ = AsyncMock(return_value=mock_db)
    mock_db.__aexit__ = AsyncMock(return_value=False)

    return mock_db, fake_video, fake_job, fake_run


def _patch_settings(mock_settings, tmp_path):
    mock_settings.normalized_dir = str(tmp_path / "normalized")
    mock_settings.artifacts_dir = str(tmp_path / "artifacts")
    mock_settings.frame_sample_fps = 2.0
    mock_settings.detector_backend = "stub"
    mock_settings.detector_model = "yolov8n.pt"
    mock_settings.tracker_backend = "stub"
    mock_settings.tracker_iou_threshold = 0.3
    mock_settings.tracker_max_age = 30
    mock_settings.pose_backend = "stub"
    mock_settings.pose_min_confidence = 0.3
    mock_settings.storage_backend = "local"


# ---------------------------------------------------------------------------
# Worker event emission
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_emits_running_event(tmp_path):
    """Worker calls enqueue_run_event with running event type."""
    mock_db, fake_video, _, fake_run = _make_mock_db(tmp_path, project_id=5)
    emitted: list = []

    async def _capture_event(db, event_type, **kwargs):
        emitted.append((event_type, kwargs))

    with (
        patch("apps.worker.jobs.process_video.AsyncSessionLocal", return_value=mock_db),
        patch("apps.worker.jobs.process_video.normalize_video"),
        patch(
            "apps.worker.jobs.process_video.probe_video",
            return_value={"duration_seconds": 5.0, "fps": 30.0, "width": 1280, "height": 720},
        ),
        patch("apps.worker.jobs.process_video.extract_frames", return_value=[]),
        patch("apps.worker.jobs.process_video.generate_clips", return_value=[]),
        patch(
            "apps.worker.jobs.process_video.run_pipeline",
            return_value=(
                _make_fake_state(),
                _make_fake_detections(),
                _make_fake_tracks(),
                _make_fake_poses(),
                _make_fake_features(),
                _make_fake_segments(),
            ),
        ),
        patch("apps.worker.jobs.process_video.settings") as mock_settings,
        patch("apps.worker.jobs.process_video.enqueue_run_event", side_effect=_capture_event),
    ):
        _patch_settings(mock_settings, tmp_path)
        from apps.worker.jobs.process_video import handle_process_video

        await handle_process_video({"job_id": 1, "payload": {"video_id": 1}})

    event_types = [e[0] for e in emitted]
    assert RunEventType.running in event_types


@pytest.mark.asyncio
async def test_worker_emits_completed_event(tmp_path):
    """Worker calls enqueue_run_event with completed event type on success."""
    mock_db, fake_video, _, fake_run = _make_mock_db(tmp_path, project_id=5)
    emitted: list = []

    async def _capture_event(db, event_type, **kwargs):
        emitted.append((event_type, kwargs))

    with (
        patch("apps.worker.jobs.process_video.AsyncSessionLocal", return_value=mock_db),
        patch("apps.worker.jobs.process_video.normalize_video"),
        patch(
            "apps.worker.jobs.process_video.probe_video",
            return_value={"duration_seconds": 5.0, "fps": 30.0, "width": 1280, "height": 720},
        ),
        patch("apps.worker.jobs.process_video.extract_frames", return_value=[]),
        patch("apps.worker.jobs.process_video.generate_clips", return_value=[]),
        patch(
            "apps.worker.jobs.process_video.run_pipeline",
            return_value=(
                _make_fake_state(),
                _make_fake_detections(),
                _make_fake_tracks(),
                _make_fake_poses(),
                _make_fake_features(),
                _make_fake_segments(),
            ),
        ),
        patch("apps.worker.jobs.process_video.settings") as mock_settings,
        patch("apps.worker.jobs.process_video.enqueue_run_event", side_effect=_capture_event),
    ):
        _patch_settings(mock_settings, tmp_path)
        from apps.worker.jobs.process_video import handle_process_video

        await handle_process_video({"job_id": 1, "payload": {"video_id": 1}})

    event_types = [e[0] for e in emitted]
    assert RunEventType.completed in event_types


@pytest.mark.asyncio
async def test_worker_completed_event_includes_artifact_types(tmp_path):
    """Completed event payload includes artifact_types."""
    mock_db, fake_video, _, fake_run = _make_mock_db(tmp_path, project_id=5)
    emitted: list = []

    async def _capture_event(db, event_type, **kwargs):
        emitted.append((event_type, kwargs))

    with (
        patch("apps.worker.jobs.process_video.AsyncSessionLocal", return_value=mock_db),
        patch("apps.worker.jobs.process_video.normalize_video"),
        patch(
            "apps.worker.jobs.process_video.probe_video",
            return_value={"duration_seconds": 5.0, "fps": 30.0, "width": 1280, "height": 720},
        ),
        patch("apps.worker.jobs.process_video.extract_frames", return_value=[]),
        patch("apps.worker.jobs.process_video.generate_clips", return_value=[]),
        patch(
            "apps.worker.jobs.process_video.run_pipeline",
            return_value=(
                _make_fake_state(),
                _make_fake_detections(),
                _make_fake_tracks(),
                _make_fake_poses(),
                _make_fake_features(),
                _make_fake_segments(),
            ),
        ),
        patch("apps.worker.jobs.process_video.settings") as mock_settings,
        patch("apps.worker.jobs.process_video.enqueue_run_event", side_effect=_capture_event),
    ):
        _patch_settings(mock_settings, tmp_path)
        from apps.worker.jobs.process_video import handle_process_video

        await handle_process_video({"job_id": 1, "payload": {"video_id": 1}})

    completed_events = [(et, kw) for et, kw in emitted if et == RunEventType.completed]
    assert len(completed_events) == 1
    _, kwargs = completed_events[0]
    assert "artifact_types" in kwargs
    assert "state" in kwargs["artifact_types"]
    assert "detections" in kwargs["artifact_types"]


@pytest.mark.asyncio
async def test_worker_emits_failed_event_with_error(tmp_path):
    """Worker emits failed event with the error string on pipeline failure."""
    mock_db, fake_video, _, fake_run = _make_mock_db(tmp_path, project_id=5)
    emitted: list = []

    async def _capture_event(db, event_type, **kwargs):
        emitted.append((event_type, kwargs))

    with (
        patch("apps.worker.jobs.process_video.AsyncSessionLocal", return_value=mock_db),
        patch("apps.worker.jobs.process_video.normalize_video"),
        patch(
            "apps.worker.jobs.process_video.probe_video",
            return_value={"duration_seconds": 5.0, "fps": 30.0, "width": 1280, "height": 720},
        ),
        patch("apps.worker.jobs.process_video.extract_frames", return_value=[]),
        patch("apps.worker.jobs.process_video.generate_clips", return_value=[]),
        patch(
            "apps.worker.jobs.process_video.run_pipeline",
            side_effect=RuntimeError("cv pipeline exploded"),
        ),
        patch("apps.worker.jobs.process_video.settings") as mock_settings,
        patch("apps.worker.jobs.process_video.enqueue_run_event", side_effect=_capture_event),
    ):
        _patch_settings(mock_settings, tmp_path)
        from apps.worker.jobs.process_video import handle_process_video

        with pytest.raises(RuntimeError):
            await handle_process_video({"job_id": 1, "payload": {"video_id": 1}})

    event_types = [e[0] for e in emitted]
    assert RunEventType.failed in event_types

    failed_events = [(et, kw) for et, kw in emitted if et == RunEventType.failed]
    assert len(failed_events) == 1
    _, kwargs = failed_events[0]
    assert "cv pipeline exploded" in kwargs.get("error", "")


@pytest.mark.asyncio
async def test_worker_failed_event_not_emit_completed(tmp_path):
    """Worker does NOT emit completed event when pipeline fails."""
    mock_db, fake_video, _, fake_run = _make_mock_db(tmp_path, project_id=5)
    emitted: list = []

    async def _capture_event(db, event_type, **kwargs):
        emitted.append((event_type, kwargs))

    with (
        patch("apps.worker.jobs.process_video.AsyncSessionLocal", return_value=mock_db),
        patch("apps.worker.jobs.process_video.normalize_video"),
        patch(
            "apps.worker.jobs.process_video.probe_video",
            return_value={"duration_seconds": 5.0, "fps": 30.0, "width": 1280, "height": 720},
        ),
        patch("apps.worker.jobs.process_video.extract_frames", return_value=[]),
        patch("apps.worker.jobs.process_video.generate_clips", return_value=[]),
        patch(
            "apps.worker.jobs.process_video.run_pipeline",
            side_effect=RuntimeError("boom"),
        ),
        patch("apps.worker.jobs.process_video.settings") as mock_settings,
        patch("apps.worker.jobs.process_video.enqueue_run_event", side_effect=_capture_event),
    ):
        _patch_settings(mock_settings, tmp_path)
        from apps.worker.jobs.process_video import handle_process_video

        with pytest.raises(RuntimeError):
            await handle_process_video({"job_id": 1, "payload": {"video_id": 1}})

    event_types = [e[0] for e in emitted]
    assert RunEventType.completed not in event_types


@pytest.mark.asyncio
async def test_worker_running_event_has_correct_project_id(tmp_path):
    """Running event includes the video's project_id."""
    mock_db, fake_video, _, fake_run = _make_mock_db(tmp_path, project_id=99)
    emitted: list = []

    async def _capture_event(db, event_type, **kwargs):
        emitted.append((event_type, kwargs))

    with (
        patch("apps.worker.jobs.process_video.AsyncSessionLocal", return_value=mock_db),
        patch("apps.worker.jobs.process_video.normalize_video"),
        patch(
            "apps.worker.jobs.process_video.probe_video",
            return_value={"duration_seconds": 5.0, "fps": 30.0, "width": 1280, "height": 720},
        ),
        patch("apps.worker.jobs.process_video.extract_frames", return_value=[]),
        patch("apps.worker.jobs.process_video.generate_clips", return_value=[]),
        patch(
            "apps.worker.jobs.process_video.run_pipeline",
            return_value=(
                _make_fake_state(),
                _make_fake_detections(),
                _make_fake_tracks(),
                _make_fake_poses(),
                _make_fake_features(),
                _make_fake_segments(),
            ),
        ),
        patch("apps.worker.jobs.process_video.settings") as mock_settings,
        patch("apps.worker.jobs.process_video.enqueue_run_event", side_effect=_capture_event),
    ):
        _patch_settings(mock_settings, tmp_path)
        from apps.worker.jobs.process_video import handle_process_video

        await handle_process_video({"job_id": 1, "payload": {"video_id": 1}})

    running_events = [(et, kw) for et, kw in emitted if et == RunEventType.running]
    assert running_events[0][1]["project_id"] == 99

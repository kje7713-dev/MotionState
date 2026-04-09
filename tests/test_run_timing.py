"""Tests for run timing fields (started_at, completed_at) populated by the worker."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from libs.models import ProcessingRun, RunStatus

# Minimal valid StreamInfo to satisfy the pre-normalization probe in the worker.
_VALID_SRC_INFO = {
    "has_video": True,
    "has_audio": True,
    "video_codec": "h264",
    "audio_codec": "aac",
    "container_format": "mov,mp4,m4a,3gp,3g2,mj2",
}


# ---------------------------------------------------------------------------
# Shared test helpers (mirrors test_processing_runs.py)
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


def _make_fake_detections(video_id: str = "1") -> dict:
    return {"video_id": video_id, "version": 1, "sample_fps": 2.0, "frames": []}


def _make_fake_tracks(video_id: str = "1") -> dict:
    return {"video_id": video_id, "version": 1, "track_count": 0, "tracks": []}


def _make_fake_poses(video_id: str = "1") -> dict:
    return {"video_id": video_id, "version": 1, "pose_count": 0, "poses": []}


def _make_fake_features(video_id: str = "1") -> dict:
    return {"video_id": video_id, "version": 1, "feature_count": 0, "features": []}


def _make_fake_segments(video_id: str = "1") -> dict:
    return {"video_id": video_id, "version": 1, "segment_count": 0, "segments": []}


def _make_mock_db(tmp_path, *, video_id: int = 1, job_id: int = 1, run_id: int = 10):
    """Return a mock async DB session with a fake ProcessingRun."""
    fake_video = MagicMock()
    fake_video.id = video_id
    fake_video.source_path = str(tmp_path / "input.mp4")
    fake_video.status = None
    fake_video.project_id = None

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
# Timing tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_started_at_set_on_start(tmp_path):
    """Worker sets ProcessingRun.started_at when the run begins."""
    mock_db, _, _, fake_run = _make_mock_db(tmp_path)

    with (
        patch("apps.worker.jobs.process_video.AsyncSessionLocal", return_value=mock_db),
        patch("apps.worker.jobs.process_video.normalize_video"),
        patch("apps.worker.jobs.process_video.probe_media_streams", return_value=_VALID_SRC_INFO),
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
    ):
        _patch_settings(mock_settings, tmp_path)
        from apps.worker.jobs.process_video import handle_process_video

        await handle_process_video({"job_id": 1, "payload": {"video_id": 1}})

    assert fake_run.started_at is not None


@pytest.mark.asyncio
async def test_run_completed_at_set_on_success(tmp_path):
    """Worker sets ProcessingRun.completed_at when the run completes successfully."""
    mock_db, _, _, fake_run = _make_mock_db(tmp_path)

    with (
        patch("apps.worker.jobs.process_video.AsyncSessionLocal", return_value=mock_db),
        patch("apps.worker.jobs.process_video.normalize_video"),
        patch("apps.worker.jobs.process_video.probe_media_streams", return_value=_VALID_SRC_INFO),
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
    ):
        _patch_settings(mock_settings, tmp_path)
        from apps.worker.jobs.process_video import handle_process_video

        await handle_process_video({"job_id": 1, "payload": {"video_id": 1}})

    assert fake_run.completed_at is not None
    assert fake_run.status == RunStatus.completed


@pytest.mark.asyncio
async def test_run_completed_at_set_on_failure(tmp_path):
    """Worker sets ProcessingRun.completed_at even when the run fails."""
    mock_db, _, _, fake_run = _make_mock_db(tmp_path)

    with (
        patch("apps.worker.jobs.process_video.AsyncSessionLocal", return_value=mock_db),
        patch("apps.worker.jobs.process_video.normalize_video"),
        patch("apps.worker.jobs.process_video.probe_media_streams", return_value=_VALID_SRC_INFO),
        patch(
            "apps.worker.jobs.process_video.probe_video",
            return_value={"duration_seconds": 5.0, "fps": 30.0, "width": 1280, "height": 720},
        ),
        patch("apps.worker.jobs.process_video.extract_frames", return_value=[]),
        patch("apps.worker.jobs.process_video.generate_clips", return_value=[]),
        patch(
            "apps.worker.jobs.process_video.run_pipeline",
            side_effect=RuntimeError("pipeline blew up"),
        ),
        patch("apps.worker.jobs.process_video.settings") as mock_settings,
    ):
        _patch_settings(mock_settings, tmp_path)
        from apps.worker.jobs.process_video import handle_process_video

        with pytest.raises(RuntimeError):
            await handle_process_video({"job_id": 1, "payload": {"video_id": 1}})

    assert fake_run.completed_at is not None
    assert fake_run.status == RunStatus.error


@pytest.mark.asyncio
async def test_run_error_string_captured_on_failure(tmp_path):
    """Worker records the exception string in ProcessingRun.error on failure."""
    mock_db, _, _, fake_run = _make_mock_db(tmp_path)

    with (
        patch("apps.worker.jobs.process_video.AsyncSessionLocal", return_value=mock_db),
        patch("apps.worker.jobs.process_video.normalize_video"),
        patch("apps.worker.jobs.process_video.probe_media_streams", return_value=_VALID_SRC_INFO),
        patch(
            "apps.worker.jobs.process_video.probe_video",
            return_value={"duration_seconds": 5.0, "fps": 30.0, "width": 1280, "height": 720},
        ),
        patch("apps.worker.jobs.process_video.extract_frames", return_value=[]),
        patch("apps.worker.jobs.process_video.generate_clips", return_value=[]),
        patch(
            "apps.worker.jobs.process_video.run_pipeline",
            side_effect=RuntimeError("stage: detect — out of memory"),
        ),
        patch("apps.worker.jobs.process_video.settings") as mock_settings,
    ):
        _patch_settings(mock_settings, tmp_path)
        from apps.worker.jobs.process_video import handle_process_video

        with pytest.raises(RuntimeError):
            await handle_process_video({"job_id": 1, "payload": {"video_id": 1}})

    assert fake_run.error is not None
    assert "out of memory" in fake_run.error

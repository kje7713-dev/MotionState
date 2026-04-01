"""Focused tests for apps/worker/jobs/process_video.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from libs.models import Artifact, JobStatus, VideoStatus


def _make_fake_state(video_id: str = "1") -> dict:
    return {
        "video_id": video_id,
        "version": 2,
        "segments": [],
        "tracks": [],
        "features": [],
        "detections_summary": {"frame_count": 2, "frames_with_people": 1, "total_detections": 1},
        "notes": "first real CV stage: frame extraction and person detection",
    }


def _make_fake_detections(video_id: str = "1") -> dict:
    return {
        "video_id": video_id,
        "version": 1,
        "sample_fps": 2.0,
        "frames": [],
    }


def _make_mock_db(tmp_path, *, video_id: int = 1, job_id: int = 1):
    """Return a mock async DB session and the fake video/job objects."""
    fake_video = MagicMock()
    fake_video.id = video_id
    fake_video.source_path = str(tmp_path / "input.mp4")
    fake_video.status = None
    fake_video.normalized_path = None

    (tmp_path / "input.mp4").write_bytes(b"fake")

    fake_job = MagicMock()
    fake_job.id = job_id
    fake_job.status = None
    fake_job.error = None

    added_objects: list = []

    mock_db = AsyncMock()
    mock_db.get = AsyncMock(
        side_effect=lambda model, pk: fake_job if model.__name__ == "Job" else fake_video
    )
    mock_db.add = MagicMock(side_effect=lambda obj: added_objects.append(obj))
    mock_db.commit = AsyncMock()
    mock_db.__aenter__ = AsyncMock(return_value=mock_db)
    mock_db.__aexit__ = AsyncMock(return_value=False)

    return mock_db, fake_video, fake_job, added_objects


def _patch_settings(mock_settings, tmp_path):
    mock_settings.normalized_dir = str(tmp_path / "normalized")
    mock_settings.artifacts_dir = str(tmp_path / "artifacts")
    mock_settings.frame_sample_fps = 2.0
    mock_settings.detector_backend = "stub"
    mock_settings.detector_model = "yolov8n.pt"


# ---------------------------------------------------------------------------
# Artifact file creation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_state_json_is_written(tmp_path):
    """handle_process_video writes state.json to the artifacts directory."""
    mock_db, _, _, _ = _make_mock_db(tmp_path)
    fake_state = _make_fake_state()
    fake_detections = _make_fake_detections()

    with (
        patch("apps.worker.jobs.process_video.AsyncSessionLocal", return_value=mock_db),
        patch("apps.worker.jobs.process_video.normalize_video"),
        patch(
            "apps.worker.jobs.process_video.probe_video",
            return_value={"duration_seconds": 5.0, "fps": 30.0, "width": 1280, "height": 720},
        ),
        patch("apps.worker.jobs.process_video.extract_frames", return_value=[]),
        patch(
            "apps.worker.jobs.process_video.run_pipeline",
            return_value=(fake_state, fake_detections),
        ),
        patch("apps.worker.jobs.process_video.settings") as mock_settings,
    ):
        _patch_settings(mock_settings, tmp_path)
        from apps.worker.jobs.process_video import handle_process_video

        await handle_process_video({"job_id": 1, "payload": {"video_id": 1}})

    state_path = tmp_path / "artifacts" / "1" / "state.json"
    assert state_path.exists(), "state.json was not written"


@pytest.mark.asyncio
async def test_detections_json_is_written(tmp_path):
    """handle_process_video writes detections.json to the artifacts directory."""
    mock_db, _, _, _ = _make_mock_db(tmp_path)
    fake_state = _make_fake_state()
    fake_detections = _make_fake_detections()

    with (
        patch("apps.worker.jobs.process_video.AsyncSessionLocal", return_value=mock_db),
        patch("apps.worker.jobs.process_video.normalize_video"),
        patch(
            "apps.worker.jobs.process_video.probe_video",
            return_value={"duration_seconds": 5.0, "fps": 30.0, "width": 1280, "height": 720},
        ),
        patch("apps.worker.jobs.process_video.extract_frames", return_value=[]),
        patch(
            "apps.worker.jobs.process_video.run_pipeline",
            return_value=(fake_state, fake_detections),
        ),
        patch("apps.worker.jobs.process_video.settings") as mock_settings,
    ):
        _patch_settings(mock_settings, tmp_path)
        from apps.worker.jobs.process_video import handle_process_video

        await handle_process_video({"job_id": 1, "payload": {"video_id": 1}})

    det_path = tmp_path / "artifacts" / "1" / "detections.json"
    assert det_path.exists(), "detections.json was not written"


# ---------------------------------------------------------------------------
# Artifact rows persisted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_artifact_rows_are_added(tmp_path):
    """Both state and detections Artifact rows are persisted."""
    mock_db, _, _, added_objects = _make_mock_db(tmp_path)
    fake_state = _make_fake_state()
    fake_detections = _make_fake_detections()

    with (
        patch("apps.worker.jobs.process_video.AsyncSessionLocal", return_value=mock_db),
        patch("apps.worker.jobs.process_video.normalize_video"),
        patch(
            "apps.worker.jobs.process_video.probe_video",
            return_value={"duration_seconds": 5.0, "fps": 30.0, "width": 1280, "height": 720},
        ),
        patch("apps.worker.jobs.process_video.extract_frames", return_value=[]),
        patch(
            "apps.worker.jobs.process_video.run_pipeline",
            return_value=(fake_state, fake_detections),
        ),
        patch("apps.worker.jobs.process_video.settings") as mock_settings,
    ):
        _patch_settings(mock_settings, tmp_path)
        from apps.worker.jobs.process_video import handle_process_video

        await handle_process_video({"job_id": 1, "payload": {"video_id": 1}})

    artifact_rows = [o for o in added_objects if isinstance(o, Artifact)]
    assert len(artifact_rows) == 2


@pytest.mark.asyncio
async def test_artifact_types_are_state_and_detections(tmp_path):
    """The two persisted Artifact rows have types 'state' and 'detections'."""
    mock_db, _, _, added_objects = _make_mock_db(tmp_path)

    with (
        patch("apps.worker.jobs.process_video.AsyncSessionLocal", return_value=mock_db),
        patch("apps.worker.jobs.process_video.normalize_video"),
        patch(
            "apps.worker.jobs.process_video.probe_video",
            return_value={"duration_seconds": 5.0, "fps": 30.0, "width": 1280, "height": 720},
        ),
        patch("apps.worker.jobs.process_video.extract_frames", return_value=[]),
        patch(
            "apps.worker.jobs.process_video.run_pipeline",
            return_value=(_make_fake_state(), _make_fake_detections()),
        ),
        patch("apps.worker.jobs.process_video.settings") as mock_settings,
    ):
        _patch_settings(mock_settings, tmp_path)
        from apps.worker.jobs.process_video import handle_process_video

        await handle_process_video({"job_id": 1, "payload": {"video_id": 1}})

    artifact_types = {o.type for o in added_objects if isinstance(o, Artifact)}
    assert artifact_types == {"state", "detections"}


# ---------------------------------------------------------------------------
# Status transitions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_job_status_transitions_to_done_on_success(tmp_path):
    mock_db, _, fake_job, _ = _make_mock_db(tmp_path)

    with (
        patch("apps.worker.jobs.process_video.AsyncSessionLocal", return_value=mock_db),
        patch("apps.worker.jobs.process_video.normalize_video"),
        patch(
            "apps.worker.jobs.process_video.probe_video",
            return_value={"duration_seconds": 5.0, "fps": 30.0, "width": 1280, "height": 720},
        ),
        patch("apps.worker.jobs.process_video.extract_frames", return_value=[]),
        patch(
            "apps.worker.jobs.process_video.run_pipeline",
            return_value=(_make_fake_state(), _make_fake_detections()),
        ),
        patch("apps.worker.jobs.process_video.settings") as mock_settings,
    ):
        _patch_settings(mock_settings, tmp_path)
        from apps.worker.jobs.process_video import handle_process_video

        await handle_process_video({"job_id": 1, "payload": {"video_id": 1}})

    assert fake_job.status == JobStatus.done


@pytest.mark.asyncio
async def test_video_status_transitions_to_ready_on_success(tmp_path):
    mock_db, fake_video, _, _ = _make_mock_db(tmp_path)

    with (
        patch("apps.worker.jobs.process_video.AsyncSessionLocal", return_value=mock_db),
        patch("apps.worker.jobs.process_video.normalize_video"),
        patch(
            "apps.worker.jobs.process_video.probe_video",
            return_value={"duration_seconds": 5.0, "fps": 30.0, "width": 1280, "height": 720},
        ),
        patch("apps.worker.jobs.process_video.extract_frames", return_value=[]),
        patch(
            "apps.worker.jobs.process_video.run_pipeline",
            return_value=(_make_fake_state(), _make_fake_detections()),
        ),
        patch("apps.worker.jobs.process_video.settings") as mock_settings,
    ):
        _patch_settings(mock_settings, tmp_path)
        from apps.worker.jobs.process_video import handle_process_video

        await handle_process_video({"job_id": 1, "payload": {"video_id": 1}})

    assert fake_video.status == VideoStatus.ready


@pytest.mark.asyncio
async def test_job_status_transitions_to_error_on_failure(tmp_path):
    """When the pipeline raises, job status must be set to error."""
    mock_db, _, fake_job, _ = _make_mock_db(tmp_path)

    with (
        patch("apps.worker.jobs.process_video.AsyncSessionLocal", return_value=mock_db),
        patch("apps.worker.jobs.process_video.normalize_video"),
        patch(
            "apps.worker.jobs.process_video.probe_video",
            return_value={"duration_seconds": 5.0, "fps": 30.0, "width": 1280, "height": 720},
        ),
        patch("apps.worker.jobs.process_video.extract_frames", return_value=[]),
        patch(
            "apps.worker.jobs.process_video.run_pipeline",
            side_effect=RuntimeError("boom"),
        ),
        patch("apps.worker.jobs.process_video.settings") as mock_settings,
    ):
        _patch_settings(mock_settings, tmp_path)
        from apps.worker.jobs.process_video import handle_process_video

        with pytest.raises(RuntimeError, match="boom"):
            await handle_process_video({"job_id": 1, "payload": {"video_id": 1}})

    assert fake_job.status == JobStatus.error


@pytest.mark.asyncio
async def test_video_status_transitions_to_error_on_failure(tmp_path):
    """When the pipeline raises, video status must be set to error."""
    mock_db, fake_video, _, _ = _make_mock_db(tmp_path)

    with (
        patch("apps.worker.jobs.process_video.AsyncSessionLocal", return_value=mock_db),
        patch("apps.worker.jobs.process_video.normalize_video"),
        patch(
            "apps.worker.jobs.process_video.probe_video",
            return_value={"duration_seconds": 5.0, "fps": 30.0, "width": 1280, "height": 720},
        ),
        patch("apps.worker.jobs.process_video.extract_frames", return_value=[]),
        patch(
            "apps.worker.jobs.process_video.run_pipeline",
            side_effect=RuntimeError("boom"),
        ),
        patch("apps.worker.jobs.process_video.settings") as mock_settings,
    ):
        _patch_settings(mock_settings, tmp_path)
        from apps.worker.jobs.process_video import handle_process_video

        with pytest.raises(RuntimeError):
            await handle_process_video({"job_id": 1, "payload": {"video_id": 1}})

    assert fake_video.status == VideoStatus.error

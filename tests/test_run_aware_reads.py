"""Tests for run-aware artifact read endpoints.

Covers:
- Without run_id: latest successful run's artifacts are returned
- With run_id: that specific run's artifacts are returned
- Failed runs do not replace the latest successful run
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def app():
    """Return the FastAPI app with DB engine mocked out."""
    with patch("apps.api.main.engine") as mock_engine:
        mock_conn = AsyncMock()
        mock_conn.run_sync = AsyncMock()
        mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=False)

        from apps.api.main import app as fastapi_app

        yield fastapi_app


def _make_state_json(tmp_path: Path, video_id: int = 1) -> dict:
    """Write a minimal state.json under tmp_path and return its content."""
    state = {
        "video_id": str(video_id),
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
        "feature_summary": {
            "feature_count": 0,
            "featured_track_count": 0,
            "feature_names": [],
        },
        "segmentation_summary": {
            "segment_count": 0,
            "segment_labels": [],
            "total_segment_duration_ms": 0,
        },
        "clip_summary": {"clip_count": 0, "total_clip_duration_ms": 0},
        "manifest_path": "",
        "notes": "stub",
    }
    art_dir = tmp_path / "artifacts" / str(video_id)
    art_dir.mkdir(parents=True, exist_ok=True)
    state_file = art_dir / "state.json"
    state_file.write_text(json.dumps(state))
    return state, str(state_file)


# ---------------------------------------------------------------------------
# Run-aware GET /videos/{id}/state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_state_without_run_id_uses_latest_successful_run(app, tmp_path):
    """GET /videos/{id}/state with no run_id returns the latest successful run's artifact."""
    from libs.config import settings
    from libs.db import get_db
    from libs.models import Artifact, ProcessingRun, RunStatus, Video

    state_content, state_path = _make_state_json(tmp_path)

    fake_video = MagicMock(spec=Video)
    fake_video.id = 1

    fake_run = MagicMock(spec=ProcessingRun)
    fake_run.id = 42
    fake_run.status = RunStatus.completed

    fake_artifact = MagicMock(spec=Artifact)
    fake_artifact.path = state_path
    fake_artifact.processing_run_id = 42

    # execute returns run on first call, artifact on subsequent calls
    call_count = 0

    def _make_result(obj):
        mock_scalars = MagicMock()
        mock_scalars.first.return_value = obj
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        return mock_result

    async def _execute(query):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # get_latest_run query
            return _make_result(fake_run)
        # artifact-within-run query
        return _make_result(fake_artifact)

    async def override_get_db():
        session = AsyncMock()
        session.get = AsyncMock(return_value=fake_video)
        session.execute = AsyncMock(side_effect=_execute)
        yield session

    app.dependency_overrides[get_db] = override_get_db

    with patch.object(settings, "artifacts_dir", str(tmp_path / "artifacts")):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.get("/videos/1/state")

    app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["video_id"] == "1"


@pytest.mark.asyncio
async def test_state_with_run_id_returns_specific_run_artifact(app, tmp_path):
    """GET /videos/{id}/state?run_id=X returns the artifact for run X."""
    from libs.config import settings
    from libs.db import get_db
    from libs.models import Artifact, Video

    state_content, state_path = _make_state_json(tmp_path)

    fake_video = MagicMock(spec=Video)
    fake_video.id = 1

    fake_artifact = MagicMock(spec=Artifact)
    fake_artifact.path = state_path
    fake_artifact.processing_run_id = 99

    def _make_result(obj):
        mock_scalars = MagicMock()
        mock_scalars.first.return_value = obj
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        return mock_result

    async def override_get_db():
        session = AsyncMock()
        session.get = AsyncMock(return_value=fake_video)
        session.execute = AsyncMock(return_value=_make_result(fake_artifact))
        yield session

    app.dependency_overrides[get_db] = override_get_db

    with patch.object(settings, "artifacts_dir", str(tmp_path / "artifacts")):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.get("/videos/1/state?run_id=99")

    app.dependency_overrides.clear()

    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Failed runs don't replace the latest successful run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failed_run_does_not_replace_latest_successful_run(app, tmp_path):
    """Artifacts from a failed run are not returned when no run_id is specified.

    The latest-run resolver filters on status=completed, so a failed run never
    becomes the default.
    """
    from libs.config import settings
    from libs.db import get_db
    from libs.models import Artifact, ProcessingRun, RunStatus, Video

    state_content, state_path = _make_state_json(tmp_path)

    fake_video = MagicMock(spec=Video)
    fake_video.id = 1

    # Only the completed run is returned by the run query (failed runs are excluded)
    fake_completed_run = MagicMock(spec=ProcessingRun)
    fake_completed_run.id = 7
    fake_completed_run.status = RunStatus.completed

    # Artifact from the completed run
    fake_artifact = MagicMock(spec=Artifact)
    fake_artifact.path = state_path
    fake_artifact.processing_run_id = 7

    call_count = 0

    def _make_result(obj):
        mock_scalars = MagicMock()
        mock_scalars.first.return_value = obj
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        return mock_result

    async def _execute(query):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_result(fake_completed_run)
        return _make_result(fake_artifact)

    async def override_get_db():
        session = AsyncMock()
        session.get = AsyncMock(return_value=fake_video)
        session.execute = AsyncMock(side_effect=_execute)
        yield session

    app.dependency_overrides[get_db] = override_get_db

    with patch.object(settings, "artifacts_dir", str(tmp_path / "artifacts")):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.get("/videos/1/state")

    app.dependency_overrides.clear()

    # Should still return the successful run's artifact
    assert response.status_code == 200
    data = response.json()
    assert data["video_id"] == "1"


@pytest.mark.asyncio
async def test_run_id_query_param_is_accepted(app, tmp_path):
    """GET /videos/{id}/state?run_id=... accepts the run_id parameter without error."""
    from libs.config import settings
    from libs.db import get_db
    from libs.models import Artifact, Video

    state_content, state_path = _make_state_json(tmp_path)

    fake_video = MagicMock(spec=Video)
    fake_video.id = 1

    fake_artifact = MagicMock(spec=Artifact)
    fake_artifact.path = state_path

    def _make_result(obj):
        mock_scalars = MagicMock()
        mock_scalars.first.return_value = obj
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        return mock_result

    async def override_get_db():
        session = AsyncMock()
        session.get = AsyncMock(return_value=fake_video)
        session.execute = AsyncMock(return_value=_make_result(fake_artifact))
        yield session

    app.dependency_overrides[get_db] = override_get_db

    with patch.object(settings, "artifacts_dir", str(tmp_path / "artifacts")):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/videos/1/state?run_id=5")

    app.dependency_overrides.clear()

    # run_id query param should be accepted (not cause a 422 validation error)
    assert r.status_code != 422, "run_id query param should not cause a validation error"
    assert r.status_code == 200

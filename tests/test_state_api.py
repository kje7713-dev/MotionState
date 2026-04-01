"""Tests for GET /videos/{video_id}/state."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from libs.config import settings

# ---------------------------------------------------------------------------
# App fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def app():
    with patch("apps.api.main.engine") as mock_engine:
        mock_conn = AsyncMock()
        mock_conn.run_sync = AsyncMock()
        mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=False)

        from apps.api.main import app as fastapi_app

        yield fastapi_app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_artifact(artifact_id: int, video_id: int, path: str):
    from libs.models import Artifact

    a = MagicMock(spec=Artifact)
    a.id = artifact_id
    a.video_id = video_id
    a.type = "state"
    a.path = path
    a.metadata_json = {"version": 7}
    return a


def _make_state_payload(video_id: str = "1") -> dict:
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
        "feature_summary": {
            "feature_count": 0,
            "featured_track_count": 0,
            "feature_names": [],
        },
        "segmentation_summary": {
            "segment_count": 0,
            "segment_labels": [],
            "total_segment_duration_ms": 0.0,
        },
        "clip_summary": {"clip_count": 0, "total_clip_duration_ms": 0},
        "manifest_path": "",
        "notes": "test",
    }


def _db_override(video=None, artifact=None):
    mock_scalars = MagicMock()
    mock_scalars.first.return_value = artifact
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars

    async def _get_db():
        session = AsyncMock()
        session.get = AsyncMock(return_value=video)
        session.execute = AsyncMock(return_value=mock_result)
        yield session

    return _get_db


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_state_returns_200(app, tmp_path):
    from libs.db import get_db
    from libs.models import Video

    fake_video = MagicMock(spec=Video)
    fake_video.id = 1

    payload = _make_state_payload("1")
    artifact_file = tmp_path / "state.json"
    artifact_file.write_text(json.dumps(payload))

    artifact = _make_artifact(1, 1, str(artifact_file))
    app.dependency_overrides[get_db] = _db_override(video=fake_video, artifact=artifact)

    with patch.object(settings, "artifacts_dir", str(tmp_path)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/videos/1/state")

    app.dependency_overrides.clear()
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_state_response_includes_expected_fields(app, tmp_path):
    from libs.db import get_db
    from libs.models import Video

    fake_video = MagicMock(spec=Video)
    fake_video.id = 1

    payload = _make_state_payload("1")
    artifact_file = tmp_path / "state.json"
    artifact_file.write_text(json.dumps(payload))

    artifact = _make_artifact(1, 1, str(artifact_file))
    app.dependency_overrides[get_db] = _db_override(video=fake_video, artifact=artifact)

    with patch.object(settings, "artifacts_dir", str(tmp_path)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/videos/1/state")

    app.dependency_overrides.clear()

    data = response.json()
    assert data["video_id"] == "1"
    assert data["version"] == 7
    assert "detections_summary" in data
    assert "tracking_summary" in data
    assert "segmentation_summary" in data
    assert "clip_summary" in data


# ---------------------------------------------------------------------------
# 404 cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_state_returns_404_for_missing_video(app):
    from libs.db import get_db

    app.dependency_overrides[get_db] = _db_override(video=None, artifact=None)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/videos/9999/state")

    app.dependency_overrides.clear()
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_state_returns_404_when_artifact_row_missing(app):
    from libs.db import get_db
    from libs.models import Video

    fake_video = MagicMock(spec=Video)
    fake_video.id = 1

    app.dependency_overrides[get_db] = _db_override(video=fake_video, artifact=None)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/videos/1/state")

    app.dependency_overrides.clear()
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_state_returns_404_when_file_missing(app, tmp_path):
    from libs.db import get_db
    from libs.models import Video

    fake_video = MagicMock(spec=Video)
    fake_video.id = 1

    missing = str(tmp_path / "nonexistent.json")
    artifact = _make_artifact(1, 1, missing)
    app.dependency_overrides[get_db] = _db_override(video=fake_video, artifact=artifact)

    with patch.object(settings, "artifacts_dir", str(tmp_path)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/videos/1/state")

    app.dependency_overrides.clear()
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_state_returns_404_for_path_outside_artifacts_dir(app, tmp_path):
    from libs.db import get_db
    from libs.models import Video

    fake_video = MagicMock(spec=Video)
    fake_video.id = 1

    # File exists but is NOT inside artifacts_dir
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_file = outside_dir / "state.json"
    outside_file.write_text("{}")

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    artifact = _make_artifact(1, 1, str(outside_file))
    app.dependency_overrides[get_db] = _db_override(video=fake_video, artifact=artifact)

    with patch.object(settings, "artifacts_dir", str(artifacts_dir)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/videos/1/state")

    app.dependency_overrides.clear()
    assert response.status_code == 404

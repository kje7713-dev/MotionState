"""Tests for GET /videos/{video_id}/features."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from libs.config import settings


@pytest.fixture
def app():
    with patch("apps.api.main.engine") as mock_engine:
        mock_conn = AsyncMock()
        mock_conn.run_sync = AsyncMock()
        mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=False)

        from apps.api.main import app as fastapi_app
        from libs.auth import get_current_project
        from tests.conftest import make_auth_override

        fastapi_app.dependency_overrides[get_current_project] = make_auth_override()
        yield fastapi_app


def _make_artifact(artifact_id: int, video_id: int, path: str):
    from libs.models import Artifact

    a = MagicMock(spec=Artifact)
    a.id = artifact_id
    a.video_id = video_id
    a.type = "features"
    a.path = path
    a.metadata_json = {"version": 1}
    return a


def _make_features_payload(video_id: str = "1") -> dict:
    return {
        "video_id": video_id,
        "version": 1,
        "feature_count": 2,
        "features": [
            {"track_id": 1, "name": "torso_angle", "start_ms": 0.0, "end_ms": 0.0, "value": 5.0},
            {
                "track_id": 1,
                "name": "centroid_velocity",
                "start_ms": 0.0,
                "end_ms": 500.0,
                "value": 3.2,
            },
        ],
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


@pytest.mark.asyncio
async def test_features_returns_200(app, tmp_path):
    from libs.db import get_db
    from libs.models import Video

    fake_video = MagicMock(spec=Video)
    fake_video.id = 1
    fake_video.project_id = 1

    payload = _make_features_payload("1")
    artifact_file = tmp_path / "features.json"
    artifact_file.write_text(json.dumps(payload))

    artifact = _make_artifact(1, 1, str(artifact_file))
    app.dependency_overrides[get_db] = _db_override(video=fake_video, artifact=artifact)

    with patch.object(settings, "artifacts_dir", str(tmp_path)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/videos/1/features")

    app.dependency_overrides.clear()
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_features_response_includes_expected_fields(app, tmp_path):
    from libs.db import get_db
    from libs.models import Video

    fake_video = MagicMock(spec=Video)
    fake_video.id = 1
    fake_video.project_id = 1

    payload = _make_features_payload("1")
    artifact_file = tmp_path / "features.json"
    artifact_file.write_text(json.dumps(payload))

    artifact = _make_artifact(1, 1, str(artifact_file))
    app.dependency_overrides[get_db] = _db_override(video=fake_video, artifact=artifact)

    with patch.object(settings, "artifacts_dir", str(tmp_path)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/videos/1/features")

    app.dependency_overrides.clear()
    data = response.json()
    assert data["video_id"] == "1"
    assert data["version"] == 1
    assert data["feature_count"] == 2
    assert "features" in data


@pytest.mark.asyncio
async def test_features_returns_404_for_missing_video(app):
    from libs.db import get_db

    app.dependency_overrides[get_db] = _db_override(video=None)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/videos/9999/features")

    app.dependency_overrides.clear()
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_features_returns_404_when_artifact_missing(app):
    from libs.db import get_db
    from libs.models import Video

    fake_video = MagicMock(spec=Video)
    fake_video.id = 1
    fake_video.project_id = 1

    app.dependency_overrides[get_db] = _db_override(video=fake_video, artifact=None)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/videos/1/features")

    app.dependency_overrides.clear()
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_features_returns_404_when_file_missing(app, tmp_path):
    from libs.db import get_db
    from libs.models import Video

    fake_video = MagicMock(spec=Video)
    fake_video.id = 1
    fake_video.project_id = 1

    artifact = _make_artifact(1, 1, str(tmp_path / "nonexistent.json"))
    app.dependency_overrides[get_db] = _db_override(video=fake_video, artifact=artifact)

    with patch.object(settings, "artifacts_dir", str(tmp_path)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/videos/1/features")

    app.dependency_overrides.clear()
    assert response.status_code == 404

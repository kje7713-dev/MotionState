"""Tests for GET /videos/{video_id}/timeline."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

# ---------------------------------------------------------------------------
# App fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def app():
    """Return the FastAPI app with DB creation disabled."""
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


def _make_manifest(video_id: str = "1", tmp_path: Path | None = None) -> dict:
    base = f"data/artifacts/{video_id}"
    return {
        "video_id": video_id,
        "version": 1,
        "duration_seconds": 5.0,
        "artifacts": {
            "state": f"{base}/state.json",
            "detections": f"{base}/detections.json",
            "tracks": f"{base}/tracks.json",
            "poses": f"{base}/poses.json",
            "features": f"{base}/features.json",
            "segments": f"{base}/segments.json",
        },
        "timeline": [
            {
                "segment_index": 0,
                "start_ms": 0.0,
                "end_ms": 2500.0,
                "label": "low_motion",
                "confidence": 0.88,
                "clip_path": f"{base}/clips/segment_000_low_motion.mp4",
                "related_artifacts": {
                    "segments": f"{base}/segments.json",
                    "features": f"{base}/features.json",
                },
            }
        ],
    }


def _make_manifest_artifact(artifact_id: int, video_id: int, manifest_path: str):
    """Return a MagicMock that looks like a timeline_manifest Artifact row."""
    from libs.models import Artifact

    a = MagicMock(spec=Artifact)
    a.id = artifact_id
    a.video_id = video_id
    a.type = "timeline_manifest"
    a.path = manifest_path
    a.metadata_json = {"version": 1}
    a.created_at = datetime(2024, 1, 1, tzinfo=UTC)
    return a


def _db_override_with_manifest(video=None, manifest_artifact=None):
    """Return a get_db override that returns the manifest artifact."""
    mock_scalars = MagicMock()
    mock_scalars.first.return_value = manifest_artifact
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
async def test_timeline_returns_200(app, tmp_path):
    """GET /videos/{id}/timeline returns 200 when the manifest exists."""
    from libs.db import get_db
    from libs.models import Video

    fake_video = MagicMock(spec=Video)
    fake_video.id = 1

    manifest_data = _make_manifest("1")
    manifest_file = tmp_path / "timeline_manifest.json"
    manifest_file.write_text(json.dumps(manifest_data))

    artifact = _make_manifest_artifact(1, 1, str(manifest_file))
    app.dependency_overrides[get_db] = _db_override_with_manifest(
        video=fake_video, manifest_artifact=artifact
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/videos/1/timeline")

    app.dependency_overrides.clear()

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_timeline_response_includes_video_id(app, tmp_path):
    """The timeline response body includes the video_id field."""
    from libs.db import get_db
    from libs.models import Video

    fake_video = MagicMock(spec=Video)
    fake_video.id = 1

    manifest_data = _make_manifest("1")
    manifest_file = tmp_path / "timeline_manifest.json"
    manifest_file.write_text(json.dumps(manifest_data))

    artifact = _make_manifest_artifact(1, 1, str(manifest_file))
    app.dependency_overrides[get_db] = _db_override_with_manifest(
        video=fake_video, manifest_artifact=artifact
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/videos/1/timeline")

    app.dependency_overrides.clear()

    assert response.json()["video_id"] == "1"


@pytest.mark.asyncio
async def test_timeline_response_includes_timeline_list(app, tmp_path):
    """The timeline response body contains a 'timeline' list."""
    from libs.db import get_db
    from libs.models import Video

    fake_video = MagicMock(spec=Video)
    fake_video.id = 1

    manifest_data = _make_manifest("1")
    manifest_file = tmp_path / "timeline_manifest.json"
    manifest_file.write_text(json.dumps(manifest_data))

    artifact = _make_manifest_artifact(1, 1, str(manifest_file))
    app.dependency_overrides[get_db] = _db_override_with_manifest(
        video=fake_video, manifest_artifact=artifact
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/videos/1/timeline")

    app.dependency_overrides.clear()

    data = response.json()
    assert "timeline" in data
    assert isinstance(data["timeline"], list)


@pytest.mark.asyncio
async def test_timeline_entry_structure(app, tmp_path):
    """Each timeline entry contains the expected fields."""
    from libs.db import get_db
    from libs.models import Video

    fake_video = MagicMock(spec=Video)
    fake_video.id = 1

    manifest_data = _make_manifest("1")
    manifest_file = tmp_path / "timeline_manifest.json"
    manifest_file.write_text(json.dumps(manifest_data))

    artifact = _make_manifest_artifact(1, 1, str(manifest_file))
    app.dependency_overrides[get_db] = _db_override_with_manifest(
        video=fake_video, manifest_artifact=artifact
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/videos/1/timeline")

    app.dependency_overrides.clear()

    entry = response.json()["timeline"][0]
    assert "segment_index" in entry
    assert "start_ms" in entry
    assert "end_ms" in entry
    assert "label" in entry
    assert "confidence" in entry
    assert "clip_path" in entry
    assert "related_artifacts" in entry


@pytest.mark.asyncio
async def test_timeline_artifacts_section_present(app, tmp_path):
    """The manifest artifacts dict is present in the response."""
    from libs.db import get_db
    from libs.models import Video

    fake_video = MagicMock(spec=Video)
    fake_video.id = 1

    manifest_data = _make_manifest("1")
    manifest_file = tmp_path / "timeline_manifest.json"
    manifest_file.write_text(json.dumps(manifest_data))

    artifact = _make_manifest_artifact(1, 1, str(manifest_file))
    app.dependency_overrides[get_db] = _db_override_with_manifest(
        video=fake_video, manifest_artifact=artifact
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/videos/1/timeline")

    app.dependency_overrides.clear()

    data = response.json()
    assert "artifacts" in data
    assert "state" in data["artifacts"]


# ---------------------------------------------------------------------------
# 404 cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timeline_returns_404_when_video_not_found(app):
    """GET /videos/{id}/timeline returns 404 when the video does not exist."""
    from libs.db import get_db

    app.dependency_overrides[get_db] = _db_override_with_manifest(
        video=None, manifest_artifact=None
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/videos/9999/timeline")

    app.dependency_overrides.clear()

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_timeline_returns_404_when_manifest_artifact_not_found(app):
    """GET /videos/{id}/timeline returns 404 when no timeline_manifest artifact row exists."""
    from libs.db import get_db
    from libs.models import Video

    fake_video = MagicMock(spec=Video)
    fake_video.id = 1

    app.dependency_overrides[get_db] = _db_override_with_manifest(
        video=fake_video, manifest_artifact=None
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/videos/1/timeline")

    app.dependency_overrides.clear()

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_timeline_returns_404_when_manifest_file_missing(app, tmp_path):
    """GET /videos/{id}/timeline returns 404 when the manifest file does not exist on disk."""
    from libs.db import get_db
    from libs.models import Video

    fake_video = MagicMock(spec=Video)
    fake_video.id = 1

    missing_path = str(tmp_path / "nonexistent" / "timeline_manifest.json")
    artifact = _make_manifest_artifact(1, 1, missing_path)
    app.dependency_overrides[get_db] = _db_override_with_manifest(
        video=fake_video, manifest_artifact=artifact
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/videos/1/timeline")

    app.dependency_overrides.clear()

    assert response.status_code == 404

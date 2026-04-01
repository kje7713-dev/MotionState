"""Focused tests for GET /videos/{video_id}/artifacts."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


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


def _make_artifact(
    artifact_id: int,
    video_id: int,
    artifact_type: str,
    path: str,
    metadata: dict | None = None,
):
    """Build a MagicMock that looks like an Artifact ORM row."""
    from libs.models import Artifact

    a = MagicMock(spec=Artifact)
    a.id = artifact_id
    a.video_id = video_id
    a.type = artifact_type
    a.path = path
    a.metadata_json = metadata or {}
    a.created_at = datetime(2024, 1, 1, tzinfo=UTC)
    return a


def _db_override(video=None, artifacts=None):
    """Return an async generator that yields a mock DB session."""
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = artifacts or []
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
async def test_returns_200_with_artifacts(app):
    from libs.db import get_db
    from libs.models import Video

    fake_video = MagicMock(spec=Video)
    fake_video.id = 1

    state_artifact = _make_artifact(1, 1, "state", "/data/1/state.json", {"version": 2})
    det_artifact = _make_artifact(
        2, 1, "detections", "/data/1/detections.json", {"version": 1, "sample_fps": 2.0}
    )

    app.dependency_overrides[get_db] = _db_override(
        video=fake_video, artifacts=[state_artifact, det_artifact]
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/videos/1/artifacts")

    app.dependency_overrides.clear()

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_returns_both_artifact_types(app):
    from libs.db import get_db
    from libs.models import Video

    fake_video = MagicMock(spec=Video)
    fake_video.id = 1

    state_artifact = _make_artifact(1, 1, "state", "/data/1/state.json")
    det_artifact = _make_artifact(2, 1, "detections", "/data/1/detections.json")

    app.dependency_overrides[get_db] = _db_override(
        video=fake_video, artifacts=[state_artifact, det_artifact]
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/videos/1/artifacts")

    app.dependency_overrides.clear()

    data = response.json()
    types = {item["type"] for item in data}
    assert types == {"state", "detections"}


@pytest.mark.asyncio
async def test_response_includes_type_path_metadata(app):
    from libs.db import get_db
    from libs.models import Video

    fake_video = MagicMock(spec=Video)
    fake_video.id = 2

    meta = {"version": 2, "sample_fps": 2.0}
    artifact = _make_artifact(10, 2, "state", "/data/2/state.json", meta)

    app.dependency_overrides[get_db] = _db_override(video=fake_video, artifacts=[artifact])

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/videos/2/artifacts")

    app.dependency_overrides.clear()

    item = response.json()[0]
    assert item["type"] == "state"
    assert item["path"] == "/data/2/state.json"
    assert item["metadata_json"] == meta


@pytest.mark.asyncio
async def test_returns_empty_list_when_no_artifacts(app):
    from libs.db import get_db
    from libs.models import Video

    fake_video = MagicMock(spec=Video)
    fake_video.id = 3

    app.dependency_overrides[get_db] = _db_override(video=fake_video, artifacts=[])

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/videos/3/artifacts")

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == []


# ---------------------------------------------------------------------------
# 404 cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_returns_404_for_unknown_video(app):
    from libs.db import get_db

    app.dependency_overrides[get_db] = _db_override(video=None)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/videos/9999/artifacts")

    app.dependency_overrides.clear()

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_404_body_contains_detail(app):
    from libs.db import get_db

    app.dependency_overrides[get_db] = _db_override(video=None)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/videos/0/artifacts")

    app.dependency_overrides.clear()

    assert "detail" in response.json()

"""Tests for GET /videos/{video_id}/runs endpoint."""

from datetime import UTC
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
        from libs.auth import get_current_project
        from tests.conftest import make_auth_override

        fastapi_app.dependency_overrides[get_current_project] = make_auth_override()
        yield fastapi_app


def _make_fake_run(
    run_id: int,
    video_id: int = 1,
    status: str = "completed",
    trigger_type: str = "initial",
    pipeline_version: str | None = "7",
    created_at=None,
    completed_at=None,
    error: str | None = None,
):
    from datetime import datetime

    from libs.models import ProcessingRun

    run = MagicMock(spec=ProcessingRun)
    run.id = run_id
    run.video_id = video_id
    run.status = status
    run.trigger_type = trigger_type
    run.pipeline_version = pipeline_version
    run.created_at = created_at or datetime(2024, 1, 1, tzinfo=UTC)
    run.completed_at = completed_at or datetime(2024, 1, 1, 1, tzinfo=UTC)
    run.error = error
    return run


# ---------------------------------------------------------------------------
# GET /videos/{id}/runs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_runs_returns_200_for_existing_video(app):
    """GET /videos/{id}/runs returns 200 when the video exists."""
    from libs.db import get_db
    from libs.models import Video

    fake_video = MagicMock(spec=Video)
    fake_video.id = 1
    fake_video.project_id = 1

    fake_runs = [_make_fake_run(3), _make_fake_run(2), _make_fake_run(1)]

    mock_scalars = MagicMock()
    mock_scalars.all.return_value = fake_runs
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars

    async def override_get_db():
        session = AsyncMock()
        session.get = AsyncMock(return_value=fake_video)
        session.execute = AsyncMock(return_value=mock_result)
        yield session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        response = await c.get("/videos/1/runs")

    app.dependency_overrides.clear()

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_list_runs_returns_404_for_missing_video(app):
    """GET /videos/{id}/runs returns 404 when the video doesn't exist."""
    from libs.db import get_db

    async def override_get_db():
        session = AsyncMock()
        session.get = AsyncMock(return_value=None)
        yield session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        response = await c.get("/videos/9999/runs")

    app.dependency_overrides.clear()

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_list_runs_returns_correct_fields(app):
    """GET /videos/{id}/runs returns the expected fields for each run."""
    from libs.db import get_db
    from libs.models import Video

    fake_video = MagicMock(spec=Video)
    fake_video.id = 1
    fake_video.project_id = 1

    fake_runs = [_make_fake_run(5, trigger_type="reprocess"), _make_fake_run(1)]

    mock_scalars = MagicMock()
    mock_scalars.all.return_value = fake_runs
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars

    async def override_get_db():
        session = AsyncMock()
        session.get = AsyncMock(return_value=fake_video)
        session.execute = AsyncMock(return_value=mock_result)
        yield session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        response = await c.get("/videos/1/runs")

    app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2

    expected_keys = {"id", "status", "trigger_type", "pipeline_version", "created_at",
                     "completed_at", "error"}
    for run in data:
        assert expected_keys == set(run.keys()), f"Unexpected keys in run: {set(run.keys())}"


@pytest.mark.asyncio
async def test_list_runs_returns_newest_first(app):
    """GET /videos/{id}/runs returns runs ordered newest-first (highest id first)."""
    from libs.db import get_db
    from libs.models import Video

    fake_video = MagicMock(spec=Video)
    fake_video.id = 1
    fake_video.project_id = 1

    # Simulate the DB returning runs newest-first (as the route requests)
    fake_runs = [_make_fake_run(10), _make_fake_run(5), _make_fake_run(1)]

    mock_scalars = MagicMock()
    mock_scalars.all.return_value = fake_runs
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars

    async def override_get_db():
        session = AsyncMock()
        session.get = AsyncMock(return_value=fake_video)
        session.execute = AsyncMock(return_value=mock_result)
        yield session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        response = await c.get("/videos/1/runs")

    app.dependency_overrides.clear()

    data = response.json()
    ids = [r["id"] for r in data]
    assert ids == sorted(ids, reverse=True), "Runs should be returned newest-first"


@pytest.mark.asyncio
async def test_list_runs_empty_for_video_with_no_runs(app):
    """GET /videos/{id}/runs returns an empty list when no runs exist."""
    from libs.db import get_db
    from libs.models import Video

    fake_video = MagicMock(spec=Video)
    fake_video.id = 1
    fake_video.project_id = 1

    mock_scalars = MagicMock()
    mock_scalars.all.return_value = []
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars

    async def override_get_db():
        session = AsyncMock()
        session.get = AsyncMock(return_value=fake_video)
        session.execute = AsyncMock(return_value=mock_result)
        yield session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        response = await c.get("/videos/1/runs")

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == []

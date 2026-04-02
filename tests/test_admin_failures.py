"""Tests for GET /admin/failures/recent and GET /admin/projects/usage/top."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from libs.models import ProcessingRun, RunStatus, TriggerType

# ---------------------------------------------------------------------------
# App fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def app():
    """Return the FastAPI app with DB engine mocked."""
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


def _make_failed_run(
    run_id: int,
    video_id: int,
    project_id: int,
    error: str = "pipeline exploded",
) -> tuple:
    run = MagicMock(spec=ProcessingRun)
    run.id = run_id
    run.video_id = video_id
    run.status = RunStatus.error
    run.trigger_type = TriggerType.initial
    run.started_at = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
    run.completed_at = datetime(2024, 1, 1, 10, 0, 2, tzinfo=UTC)
    run.error = error
    run.created_at = datetime(2024, 1, 1, 9, 59, 0, tzinfo=UTC)
    return run, project_id


def _db_override_rows(rows: list[tuple]):
    mock_result = MagicMock()
    mock_result.all.return_value = rows

    async def _get_db():
        session = AsyncMock()
        session.execute = AsyncMock(return_value=mock_result)
        yield session

    return _get_db


def _db_override_scalar(value):
    """Return a DB override where execute returns a scalar result."""
    mock_scalar_result = MagicMock()
    mock_scalar_result.all.return_value = value

    async def _get_db():
        session = AsyncMock()
        session.execute = AsyncMock(return_value=mock_scalar_result)
        yield session

    return _get_db


# ---------------------------------------------------------------------------
# Tests: recent failures endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recent_failures_requires_admin_token(app):
    """GET /admin/failures/recent returns 403 without admin token."""
    with patch("apps.api.routes.admin.settings") as mock_settings:
        mock_settings.admin_token = ""

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/admin/failures/recent")

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_recent_failures_returns_list(app):
    """GET /admin/failures/recent returns a list of failed runs."""
    from libs.db import get_db

    run1, pid1 = _make_failed_run(run_id=10, video_id=5, project_id=2, error="boom")
    run2, pid2 = _make_failed_run(run_id=8, video_id=4, project_id=1, error="oops")
    app.dependency_overrides[get_db] = _db_override_rows([(run1, pid1), (run2, pid2)])

    with patch("apps.api.routes.admin.settings") as mock_settings:
        mock_settings.admin_token = "secret"

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/admin/failures/recent",
                headers={"X-Admin-Token": "secret"},
            )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2


@pytest.mark.asyncio
async def test_recent_failures_include_required_identifiers(app):
    """Each failure entry includes run_id, video_id, project_id, and error."""
    from libs.db import get_db

    run, pid = _make_failed_run(
        run_id=99, video_id=50, project_id=7, error="normalize failed"
    )
    app.dependency_overrides[get_db] = _db_override_rows([(run, pid)])

    with patch("apps.api.routes.admin.settings") as mock_settings:
        mock_settings.admin_token = "secret"

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/admin/failures/recent",
                headers={"X-Admin-Token": "secret"},
            )

    app.dependency_overrides.clear()
    entry = response.json()[0]
    assert entry["run_id"] == 99
    assert entry["video_id"] == 50
    assert entry["project_id"] == 7
    assert entry["error"] == "normalize failed"
    assert "failed_at" in entry
    assert "created_at" in entry


@pytest.mark.asyncio
async def test_recent_failures_empty_when_no_failures(app):
    """GET /admin/failures/recent returns empty list when no failures exist."""
    from libs.db import get_db

    app.dependency_overrides[get_db] = _db_override_rows([])

    with patch("apps.api.routes.admin.settings") as mock_settings:
        mock_settings.admin_token = "secret"

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/admin/failures/recent",
                headers={"X-Admin-Token": "secret"},
            )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json() == []


# ---------------------------------------------------------------------------
# Tests: top projects usage endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_top_projects_usage_requires_admin_token(app):
    """GET /admin/projects/usage/top returns 403 without admin token."""
    with patch("apps.api.routes.admin.settings") as mock_settings:
        mock_settings.admin_token = ""

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/admin/projects/usage/top")

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_top_projects_usage_returns_ranked_projects(app):
    """GET /admin/projects/usage/top returns projects ordered by usage."""
    from libs.db import get_db

    # Simulate two rows: project 3 with 500 seconds, project 1 with 200 seconds
    row1 = MagicMock()
    row1.project_id = 3
    row1.total = 500
    row2 = MagicMock()
    row2.project_id = 1
    row2.total = 200

    mock_result = MagicMock()
    mock_result.all.return_value = [(3, 500), (1, 200)]

    async def _get_db():
        session = AsyncMock()
        session.execute = AsyncMock(return_value=mock_result)
        yield session

    app.dependency_overrides[get_db] = _get_db

    with patch("apps.api.routes.admin.settings") as mock_settings:
        mock_settings.admin_token = "secret"

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/admin/projects/usage/top",
                headers={"X-Admin-Token": "secret"},
            )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    data = response.json()
    assert "projects" in data
    assert "year" in data
    assert "month" in data
    assert "metric" in data
    assert len(data["projects"]) == 2
    assert data["projects"][0]["project_id"] == 3
    assert data["projects"][0]["total"] == 500


@pytest.mark.asyncio
async def test_top_projects_usage_empty_when_no_usage(app):
    """GET /admin/projects/usage/top returns empty projects list when no usage."""
    from libs.db import get_db

    mock_result = MagicMock()
    mock_result.all.return_value = []

    async def _get_db():
        session = AsyncMock()
        session.execute = AsyncMock(return_value=mock_result)
        yield session

    app.dependency_overrides[get_db] = _get_db

    with patch("apps.api.routes.admin.settings") as mock_settings:
        mock_settings.admin_token = "secret"

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/admin/projects/usage/top",
                headers={"X-Admin-Token": "secret"},
            )

    app.dependency_overrides.clear()
    data = response.json()
    assert data["projects"] == []

"""Tests for project usage and quota API endpoints.

Covers:
- GET /projects/{id}/usage returns 200 with correct shape
- GET /projects/{id}/usage returns 404 for missing project
- GET /projects/{id}/usage/current-month returns 200 with year/month/totals
- GET /projects/{id}/usage/current-month returns 404 for missing project
- GET /projects/{id}/quotas returns 200 with quota fields
- GET /projects/{id}/quotas returns 404 for missing project
- Endpoints are project-scoped (only return data for the requested project)
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

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


def _make_fake_project(
    project_id: int = 1,
    *,
    is_suspended: bool = False,
    max_videos_per_month: int | None = None,
    max_video_seconds_per_month: int | None = None,
    max_storage_bytes: int | None = None,
    max_api_reads_per_month: int | None = None,
):
    from libs.models import Project

    p = MagicMock(spec=Project)
    p.id = project_id
    p.name = "test-project"
    p.created_at = datetime(2024, 1, 1, tzinfo=UTC)
    p.updated_at = datetime(2024, 1, 1, tzinfo=UTC)
    p.is_suspended = is_suspended
    p.max_videos_per_month = max_videos_per_month
    p.max_video_seconds_per_month = max_video_seconds_per_month
    p.max_storage_bytes = max_storage_bytes
    p.max_api_reads_per_month = max_api_reads_per_month
    return p


def _db_with_project(
    project,
    *,
    monthly: dict | None = None,
    alltime: dict | None = None,
    storage: int = 0,
):
    """Return a get_db override that mocks project lookup and usage aggregation."""
    monthly = monthly or {}
    alltime = alltime or {}

    class _Row:
        def __init__(self, event_type, total):
            self.event_type = event_type
            self.total = total

    async def _get_db():
        session = AsyncMock()
        session.get = AsyncMock(return_value=project)
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.commit = AsyncMock()

        async def _execute(stmt):
            result = MagicMock()
            result.all.return_value = [_Row(k, v) for k, v in alltime.items()]
            result.scalar.return_value = storage
            return result

        session.execute = AsyncMock(side_effect=_execute)
        yield session

    return _get_db


# ---------------------------------------------------------------------------
# GET /projects/{id}/usage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_usage_returns_200(app):
    """GET /projects/{id}/usage returns 200."""
    from libs.db import get_db

    project = _make_fake_project(project_id=1)
    with patch("apps.api.routes.projects.project_usage_summary", new=AsyncMock(return_value={
        "project_id": 1,
        "current_month": {"year": 2024, "month": 1, "totals": {}},
        "alltime": {},
        "storage_bytes_total": 0,
    })):
        app.dependency_overrides[get_db] = _db_with_project(project)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.get("/projects/1/usage")

    app.dependency_overrides.clear()
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_get_usage_returns_expected_shape(app):
    """GET /projects/{id}/usage response includes project_id, current_month, alltime, storage."""
    from libs.db import get_db

    project = _make_fake_project(project_id=5)
    summary = {
        "project_id": 5,
        "current_month": {"year": 2024, "month": 3, "totals": {"videos_uploaded": 2}},
        "alltime": {"videos_uploaded": 20},
        "storage_bytes_total": 9999,
    }
    with patch(
        "apps.api.routes.projects.project_usage_summary",
        new=AsyncMock(return_value=summary),
    ):
        app.dependency_overrides[get_db] = _db_with_project(project)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.get("/projects/5/usage")

    app.dependency_overrides.clear()
    data = response.json()
    assert data["project_id"] == 5
    assert "current_month" in data
    assert "alltime" in data
    assert data["storage_bytes_total"] == 9999


@pytest.mark.asyncio
async def test_get_usage_returns_404_for_missing_project(app):
    """GET /projects/{id}/usage returns 404 when project does not exist."""
    from libs.db import get_db

    app.dependency_overrides[get_db] = _db_with_project(None)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        response = await c.get("/projects/9999/usage")

    app.dependency_overrides.clear()
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /projects/{id}/usage/current-month
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_usage_current_month_returns_200(app):
    """GET /projects/{id}/usage/current-month returns 200."""
    from libs.db import get_db

    project = _make_fake_project(project_id=1)
    with patch(
        "apps.api.routes.projects.monthly_totals",
        new=AsyncMock(return_value={"videos_uploaded": 1}),
    ):
        app.dependency_overrides[get_db] = _db_with_project(project)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.get("/projects/1/usage/current-month")

    app.dependency_overrides.clear()
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_get_usage_current_month_has_year_month_totals(app):
    """GET /projects/{id}/usage/current-month response includes year, month, totals."""
    from libs.db import get_db

    project = _make_fake_project(project_id=2)
    mock_totals = {"videos_uploaded": 3, "frames_extracted": 150}
    with patch("apps.api.routes.projects.monthly_totals", new=AsyncMock(return_value=mock_totals)):
        app.dependency_overrides[get_db] = _db_with_project(project)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.get("/projects/2/usage/current-month")

    app.dependency_overrides.clear()
    data = response.json()
    now = datetime.now(UTC)
    assert data["year"] == now.year
    assert data["month"] == now.month
    assert data["totals"]["videos_uploaded"] == 3
    assert data["totals"]["frames_extracted"] == 150


@pytest.mark.asyncio
async def test_get_usage_current_month_returns_404_for_missing_project(app):
    """GET /projects/{id}/usage/current-month returns 404 when project doesn't exist."""
    from libs.db import get_db

    app.dependency_overrides[get_db] = _db_with_project(None)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        response = await c.get("/projects/9999/usage/current-month")

    app.dependency_overrides.clear()
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /projects/{id}/quotas
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_quotas_returns_200(app):
    """GET /projects/{id}/quotas returns 200."""
    from libs.db import get_db

    project = _make_fake_project(project_id=1)
    app.dependency_overrides[get_db] = _db_with_project(project)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        response = await c.get("/projects/1/quotas")

    app.dependency_overrides.clear()
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_get_quotas_returns_null_for_unlimited_project(app):
    """GET /projects/{id}/quotas returns null quota fields when no limit is set."""
    from libs.db import get_db

    project = _make_fake_project(project_id=1)
    app.dependency_overrides[get_db] = _db_with_project(project)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        response = await c.get("/projects/1/quotas")

    app.dependency_overrides.clear()
    data = response.json()
    assert data["max_videos_per_month"] is None
    assert data["max_video_seconds_per_month"] is None
    assert data["max_storage_bytes"] is None
    assert data["max_api_reads_per_month"] is None
    assert data["is_suspended"] is False


@pytest.mark.asyncio
async def test_get_quotas_returns_configured_limits(app):
    """GET /projects/{id}/quotas returns the configured quota values."""
    from libs.db import get_db

    project = _make_fake_project(
        project_id=3,
        max_videos_per_month=50,
        max_storage_bytes=1_000_000,
        is_suspended=False,
    )
    app.dependency_overrides[get_db] = _db_with_project(project)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        response = await c.get("/projects/3/quotas")

    app.dependency_overrides.clear()
    data = response.json()
    assert data["max_videos_per_month"] == 50
    assert data["max_storage_bytes"] == 1_000_000


@pytest.mark.asyncio
async def test_get_quotas_returns_suspended_flag(app):
    """GET /projects/{id}/quotas reflects the is_suspended flag."""
    from libs.db import get_db

    project = _make_fake_project(project_id=4, is_suspended=True)
    app.dependency_overrides[get_db] = _db_with_project(project)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        response = await c.get("/projects/4/quotas")

    app.dependency_overrides.clear()
    data = response.json()
    assert data["is_suspended"] is True


@pytest.mark.asyncio
async def test_get_quotas_returns_404_for_missing_project(app):
    """GET /projects/{id}/quotas returns 404 when project does not exist."""
    from libs.db import get_db

    app.dependency_overrides[get_db] = _db_with_project(None)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        response = await c.get("/projects/9999/quotas")

    app.dependency_overrides.clear()
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Project-scoped checks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_usage_endpoint_is_project_scoped(app):
    """GET /projects/{id}/usage uses the project_id from the URL path."""
    from libs.db import get_db

    project_3 = _make_fake_project(project_id=3)
    summary_for_3 = {
        "project_id": 3,
        "current_month": {"year": 2024, "month": 1, "totals": {}},
        "alltime": {},
        "storage_bytes_total": 0,
    }

    with patch(
        "apps.api.routes.projects.project_usage_summary",
        new=AsyncMock(return_value=summary_for_3),
    ) as mock_summary:
        app.dependency_overrides[get_db] = _db_with_project(project_3)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.get("/projects/3/usage")

        # Verify the helper was called with project_id=3.
        mock_summary.assert_called_once()
        call_kwargs = mock_summary.call_args
        assert call_kwargs.args[1] == 3 or call_kwargs.kwargs.get("project_id") == 3

    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["project_id"] == 3

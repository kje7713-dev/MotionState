"""Tests for GET /admin/runs/recent endpoint."""

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


def _make_fake_run(
    run_id: int,
    video_id: int,
    project_id: int,
    status: RunStatus = RunStatus.completed,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
    error: str | None = None,
) -> tuple:
    run = MagicMock(spec=ProcessingRun)
    run.id = run_id
    run.video_id = video_id
    run.status = status
    run.trigger_type = TriggerType.initial
    run.started_at = started_at or datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
    run.completed_at = completed_at or datetime(2024, 1, 1, 10, 0, 5, tzinfo=UTC)
    run.error = error
    run.created_at = datetime(2024, 1, 1, 9, 59, 0, tzinfo=UTC)
    return run, project_id


def _db_override_for_runs(rows: list[tuple]):
    """Return a get_db override that returns the given run rows."""
    mock_result = MagicMock()
    mock_result.all.return_value = rows

    async def _get_db():
        session = AsyncMock()
        session.execute = AsyncMock(return_value=mock_result)
        yield session

    return _get_db


# ---------------------------------------------------------------------------
# Tests: recent runs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recent_runs_requires_admin_token(app):
    """GET /admin/runs/recent returns 403 without admin token."""
    with patch("apps.api.routes.admin.settings") as mock_settings:
        mock_settings.admin_token = ""

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/admin/runs/recent")

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_recent_runs_returns_list(app):
    """GET /admin/runs/recent returns a list of runs."""
    from libs.db import get_db

    run1, pid1 = _make_fake_run(run_id=2, video_id=10, project_id=1)
    run2, pid2 = _make_fake_run(run_id=1, video_id=9, project_id=1)
    app.dependency_overrides[get_db] = _db_override_for_runs([(run1, pid1), (run2, pid2)])

    with patch("apps.api.routes.admin.settings") as mock_settings:
        mock_settings.admin_token = "secret"

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/admin/runs/recent",
                headers={"X-Admin-Token": "secret"},
            )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2


@pytest.mark.asyncio
async def test_recent_runs_ordered_newest_first(app):
    """GET /admin/runs/recent returns runs ordered newest-first (id desc)."""
    from libs.db import get_db

    run1, pid1 = _make_fake_run(run_id=5, video_id=10, project_id=1)
    run2, pid2 = _make_fake_run(run_id=3, video_id=9, project_id=1)
    run3, pid3 = _make_fake_run(run_id=1, video_id=8, project_id=1)
    app.dependency_overrides[get_db] = _db_override_for_runs(
        [(run1, pid1), (run2, pid2), (run3, pid3)]
    )

    with patch("apps.api.routes.admin.settings") as mock_settings:
        mock_settings.admin_token = "secret"

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/admin/runs/recent",
                headers={"X-Admin-Token": "secret"},
            )

    app.dependency_overrides.clear()
    data = response.json()
    run_ids = [r["run_id"] for r in data]
    assert run_ids == [5, 3, 1]


@pytest.mark.asyncio
async def test_recent_runs_include_expected_fields(app):
    """Each run entry includes the required operational fields."""
    from libs.db import get_db

    started = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
    completed = datetime(2024, 1, 1, 10, 0, 5, tzinfo=UTC)
    run, pid = _make_fake_run(
        run_id=7, video_id=20, project_id=3, started_at=started, completed_at=completed
    )
    app.dependency_overrides[get_db] = _db_override_for_runs([(run, pid)])

    with patch("apps.api.routes.admin.settings") as mock_settings:
        mock_settings.admin_token = "secret"

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/admin/runs/recent",
                headers={"X-Admin-Token": "secret"},
            )

    app.dependency_overrides.clear()
    entry = response.json()[0]
    assert entry["run_id"] == 7
    assert entry["video_id"] == 20
    assert entry["project_id"] == 3
    assert "status" in entry
    assert "started_at" in entry
    assert "completed_at" in entry
    assert "duration_ms" in entry
    assert entry["duration_ms"] == 5000


@pytest.mark.asyncio
async def test_recent_runs_empty_when_no_runs(app):
    """GET /admin/runs/recent returns an empty list when there are no runs."""
    from libs.db import get_db

    app.dependency_overrides[get_db] = _db_override_for_runs([])

    with patch("apps.api.routes.admin.settings") as mock_settings:
        mock_settings.admin_token = "secret"

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/admin/runs/recent",
                headers={"X-Admin-Token": "secret"},
            )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json() == []

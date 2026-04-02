"""Tests for project-scoped resource access.

Covers:
- cross-project resource access returns 404
- protected routes correctly scope video lookup to current project
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from tests.conftest import make_auth_override

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
        from libs.auth import get_current_project

        fastapi_app.dependency_overrides[get_current_project] = make_auth_override(project_id=1)
        yield fastapi_app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_video(video_id: int, project_id: int):
    from libs.models import Video

    v = MagicMock(spec=Video)
    v.id = video_id
    v.project_id = project_id
    v.original_filename = "test.mp4"
    v.status = "ready"
    v.source_path = None
    v.normalized_path = None
    v.duration_seconds = None
    v.fps = None
    v.width = None
    v.height = None
    v.created_at = None
    v.updated_at = None
    return v


def _db_override_for_video(video=None):
    """Return a get_db override that returns a specific video from db.get."""

    async def _get_db():
        session = AsyncMock()
        session.get = AsyncMock(return_value=video)
        yield session

    return _get_db


# ---------------------------------------------------------------------------
# Cross-project access returns 404
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cross_project_video_access_returns_404(app):
    """GET /videos/{id} returns 404 when the video belongs to a different project.

    The auth override sets current_project.id = 1.
    The video belongs to project 2.
    Result should be 404 (not 403) to avoid leaking existence.
    """
    from libs.db import get_db

    # Video belongs to project 2, but the authenticated project is 1.
    video_from_other_project = _make_video(video_id=42, project_id=2)
    app.dependency_overrides[get_db] = _db_override_for_video(video=video_from_other_project)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/videos/42")

    app.dependency_overrides.clear()
    # Refresh auth override after clear
    from libs.auth import get_current_project
    app.dependency_overrides[get_current_project] = make_auth_override(project_id=1)

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_own_project_video_access_returns_200(app):
    """GET /videos/{id} returns 200 when the video belongs to the authenticated project."""
    from libs.db import get_db

    video_from_same_project = _make_video(video_id=7, project_id=1)
    app.dependency_overrides[get_db] = _db_override_for_video(video=video_from_same_project)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/videos/7")

    app.dependency_overrides.clear()

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_cross_project_reprocess_returns_404(app):
    """POST /videos/{id}/reprocess returns 404 for a video in another project."""
    from libs.db import get_db

    video_from_other_project = _make_video(video_id=99, project_id=2)
    app.dependency_overrides[get_db] = _db_override_for_video(video=video_from_other_project)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/videos/99/reprocess")

    app.dependency_overrides.clear()

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_cross_project_runs_returns_404(app):
    """GET /videos/{id}/runs returns 404 for a video in another project."""
    from libs.db import get_db

    video_from_other_project = _make_video(video_id=55, project_id=2)
    app.dependency_overrides[get_db] = _db_override_for_video(video=video_from_other_project)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/videos/55/runs")

    app.dependency_overrides.clear()

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_cross_project_artifacts_returns_404(app):
    """GET /videos/{id}/artifacts returns 404 for a video in another project."""
    from libs.db import get_db

    video_from_other_project = _make_video(video_id=11, project_id=2)
    app.dependency_overrides[get_db] = _db_override_for_video(video=video_from_other_project)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/videos/11/artifacts")

    app.dependency_overrides.clear()

    assert response.status_code == 404

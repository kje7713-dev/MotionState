"""Tests for quota enforcement (libs/quotas.py).

Covers:
- suspended project raises 403
- project over video upload limit raises 429
- project over video-seconds limit raises 429
- project over storage limit raises 429
- project with no quotas (None) is always allowed
- upload and reprocess endpoints reject suspended / over-quota projects
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_project(
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
    p.name = "test"
    p.is_suspended = is_suspended
    p.max_videos_per_month = max_videos_per_month
    p.max_video_seconds_per_month = max_video_seconds_per_month
    p.max_storage_bytes = max_storage_bytes
    p.max_api_reads_per_month = max_api_reads_per_month
    return p


async def _session_with_monthly(totals: dict):
    """Return a mock session where monthly_totals() returns *totals*."""
    session = AsyncMock()

    class _Row:
        def __init__(self, event_type, total):
            self.event_type = event_type
            self.total = total

    rows = [_Row(k, v) for k, v in totals.items()]
    mock_result = MagicMock()
    mock_result.all.return_value = rows
    mock_result.scalar.return_value = 0
    session.execute = AsyncMock(return_value=mock_result)
    return session


# ---------------------------------------------------------------------------
# check_quota() — suspended
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_quota_raises_403_for_suspended_project():
    """check_quota() raises HTTPException(403) when project is suspended."""
    from libs.quotas import check_quota

    project = _make_project(is_suspended=True)
    session = AsyncMock()

    with pytest.raises(HTTPException) as exc_info:
        await check_quota(session, project)

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail["reason"] == "project_suspended"


# ---------------------------------------------------------------------------
# check_quota() — video upload limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_quota_raises_429_when_video_limit_reached():
    """check_quota() raises 429 when video upload limit is reached."""
    from libs.quotas import check_quota

    project = _make_project(max_videos_per_month=5)
    session = await _session_with_monthly({"videos_uploaded": 5})

    with pytest.raises(HTTPException) as exc_info:
        await check_quota(session, project, videos_upload=True)

    assert exc_info.value.status_code == 429
    assert exc_info.value.detail["reason"] == "max_videos_per_month"


@pytest.mark.asyncio
async def test_check_quota_allows_when_video_limit_not_reached():
    """check_quota() does not raise when under the video upload limit."""
    from libs.quotas import check_quota

    project = _make_project(max_videos_per_month=10)
    session = await _session_with_monthly({"videos_uploaded": 3})

    # Should not raise.
    await check_quota(session, project, videos_upload=True)


@pytest.mark.asyncio
async def test_check_quota_allows_when_no_video_limit():
    """check_quota() does not raise when max_videos_per_month is None."""
    from libs.quotas import check_quota

    project = _make_project(max_videos_per_month=None)
    session = await _session_with_monthly({"videos_uploaded": 999})

    await check_quota(session, project, videos_upload=True)


# ---------------------------------------------------------------------------
# check_quota() — video seconds limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_quota_raises_429_when_video_seconds_exceeded():
    """check_quota() raises 429 when adding video_seconds would exceed the limit."""
    from libs.quotas import check_quota

    project = _make_project(max_video_seconds_per_month=100)
    session = await _session_with_monthly({"video_seconds_processed": 90})

    with pytest.raises(HTTPException) as exc_info:
        await check_quota(session, project, video_seconds=20)

    assert exc_info.value.status_code == 429
    assert exc_info.value.detail["reason"] == "max_video_seconds_per_month"


@pytest.mark.asyncio
async def test_check_quota_allows_when_video_seconds_within_limit():
    """check_quota() does not raise when video seconds are within the limit."""
    from libs.quotas import check_quota

    project = _make_project(max_video_seconds_per_month=200)
    session = await _session_with_monthly({"video_seconds_processed": 50})

    await check_quota(session, project, video_seconds=30)


# ---------------------------------------------------------------------------
# check_quota() — storage limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_quota_raises_429_when_storage_limit_reached():
    """check_quota() raises 429 when storage bytes limit is reached."""
    from libs.quotas import check_quota

    project = _make_project(max_storage_bytes=1024)
    session = AsyncMock()
    mock_result = MagicMock()
    mock_result.all.return_value = []
    mock_result.scalar.return_value = 1024  # already at limit
    session.execute = AsyncMock(return_value=mock_result)

    with pytest.raises(HTTPException) as exc_info:
        await check_quota(session, project)

    assert exc_info.value.status_code == 429
    assert exc_info.value.detail["reason"] == "max_storage_bytes"


@pytest.mark.asyncio
async def test_check_quota_allows_when_storage_below_limit():
    """check_quota() does not raise when storage is below the limit."""
    from libs.quotas import check_quota

    project = _make_project(max_storage_bytes=10_000)
    session = AsyncMock()
    mock_result = MagicMock()
    mock_result.all.return_value = []
    mock_result.scalar.return_value = 5000
    session.execute = AsyncMock(return_value=mock_result)

    await check_quota(session, project)


# ---------------------------------------------------------------------------
# Upload endpoint — quota rejection via HTTP
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


def _db_for_quota_test(project, *, monthly_videos: int = 0):
    """Return a get_db override for quota test scenarios."""

    class _Row:
        def __init__(self, event_type, total):
            self.event_type = event_type
            self.total = total
            self.scalar = lambda: total

    async def _get_db():
        session = AsyncMock()
        session.get = AsyncMock(return_value=project)
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.commit = AsyncMock()

        mock_result = MagicMock()
        mock_result.all.return_value = (
            [_Row("videos_uploaded", monthly_videos)] if monthly_videos else []
        )
        mock_result.scalar.return_value = 0
        session.execute = AsyncMock(return_value=mock_result)
        yield session

    return _get_db


@pytest.mark.asyncio
async def test_upload_init_returns_403_for_suspended_project(app):
    """POST /videos/upload-init returns 403 for a suspended project."""
    from libs.auth import get_current_project
    from libs.db import get_db
    from libs.storage.local import LocalStorage

    suspended_project = _make_project(project_id=1, is_suspended=True)

    async def _auth_override():
        return suspended_project

    app.dependency_overrides[get_current_project] = _auth_override
    app.dependency_overrides[get_db] = _db_for_quota_test(suspended_project)

    with patch("libs.storage.get_storage", return_value=LocalStorage("/tmp")):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post("/videos/upload-init", json={"filename": "test.mp4"})

    app.dependency_overrides.clear()
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_upload_init_returns_429_when_video_quota_exceeded(app):
    """POST /videos/upload-init returns 429 when the monthly video limit is reached."""
    from libs.auth import get_current_project
    from libs.db import get_db
    from libs.storage.local import LocalStorage

    quota_project = _make_project(project_id=1, max_videos_per_month=3)

    async def _auth_override():
        return quota_project

    app.dependency_overrides[get_current_project] = _auth_override
    app.dependency_overrides[get_db] = _db_for_quota_test(quota_project, monthly_videos=3)

    with patch("libs.storage.get_storage", return_value=LocalStorage("/tmp")):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post("/videos/upload-init", json={"filename": "test.mp4"})

    app.dependency_overrides.clear()
    assert response.status_code == 429


@pytest.mark.asyncio
async def test_reprocess_returns_403_for_suspended_project(app, tmp_path):
    """POST /videos/{id}/reprocess returns 403 for a suspended project."""
    from libs.auth import get_current_project
    from libs.db import get_db
    from libs.models import Video, VideoStatus

    suspended_project = _make_project(project_id=1, is_suspended=True)

    fake_video = MagicMock(spec=Video)
    fake_video.id = 1
    fake_video.project_id = 1
    fake_video.status = VideoStatus.ready

    async def _auth_override():
        return suspended_project

    async def _get_db():
        session = AsyncMock()

        async def _get(model, pk):
            from libs.models import Video
            if model is Video:
                return fake_video
            return None

        session.get = AsyncMock(side_effect=_get)
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.commit = AsyncMock()
        mock_result = MagicMock()
        mock_result.all.return_value = []
        mock_result.scalar.return_value = 0
        session.execute = AsyncMock(return_value=mock_result)
        yield session

    app.dependency_overrides[get_current_project] = _auth_override
    app.dependency_overrides[get_db] = _get_db

    with (
        patch("apps.api.routes.videos.enqueue", new=AsyncMock()),
        patch("apps.api.routes.videos.enqueue_run_event", new=AsyncMock()),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post("/videos/1/reprocess")

    app.dependency_overrides.clear()
    assert response.status_code == 403


# ---------------------------------------------------------------------------
# Suspended project cannot submit new work
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_suspended_project_cannot_upload(app):
    """Suspended project cannot call upload-init."""
    from libs.auth import get_current_project
    from libs.db import get_db
    from libs.storage.local import LocalStorage

    project = _make_project(project_id=7, is_suspended=True)

    async def _auth():
        return project

    app.dependency_overrides[get_current_project] = _auth
    app.dependency_overrides[get_db] = _db_for_quota_test(project)

    with patch("libs.storage.get_storage", return_value=LocalStorage("/tmp")):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post("/videos/upload-init", json={"filename": "vid.mp4"})

    app.dependency_overrides.clear()
    detail = response.json()["detail"]
    assert response.status_code == 403
    assert detail["reason"] == "project_suspended"

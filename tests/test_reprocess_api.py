"""Tests for the POST /videos/{video_id}/reprocess endpoint."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def app():
    """Return the FastAPI app with the DB engine mocked out."""
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


@pytest.mark.asyncio
async def test_reprocess_returns_201(app):
    """POST /videos/{id}/reprocess returns 201 with correct keys."""
    from libs.db import get_db
    from libs.models import Job, ProcessingRun, Video

    async def override_get_db():
        session = AsyncMock()

        fake_video = MagicMock(spec=Video)
        fake_video.id = 1
        fake_video.project_id = 1

        call_count = 0

        def _set_id(obj):
            nonlocal call_count
            call_count += 1
            if isinstance(obj, ProcessingRun):
                obj.id = 10
            elif isinstance(obj, Job):
                obj.id = 99

        session.get = AsyncMock(return_value=fake_video)
        session.add = MagicMock(side_effect=_set_id)
        session.flush = AsyncMock()
        session.commit = AsyncMock()
        yield session

    app.dependency_overrides[get_db] = override_get_db

    with patch("apps.api.routes.videos.enqueue", new_callable=AsyncMock):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post("/videos/1/reprocess")

    app.dependency_overrides.clear()

    assert response.status_code == 201
    data = response.json()
    assert "video_id" in data
    assert "processing_run_id" in data
    assert "job_id" in data
    assert data["video_id"] == 1


@pytest.mark.asyncio
async def test_reprocess_returns_404_for_missing_video(app):
    """POST /videos/{id}/reprocess returns 404 when the video doesn't exist."""
    from libs.db import get_db

    async def override_get_db():
        session = AsyncMock()
        session.get = AsyncMock(return_value=None)
        yield session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        response = await c.post("/videos/9999/reprocess")

    app.dependency_overrides.clear()

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_reprocess_commits_before_enqueue(app):
    """POST /videos/{id}/reprocess must commit the DB transaction before calling enqueue."""
    from libs.db import get_db
    from libs.models import Job, ProcessingRun, Video

    call_order: list[str] = []

    async def override_get_db():
        session = AsyncMock()
        fake_video = MagicMock(spec=Video)
        fake_video.id = 7
        fake_video.project_id = 1

        def _set_id(obj):
            if isinstance(obj, ProcessingRun):
                obj.id = 50
            elif isinstance(obj, Job):
                obj.id = 500

        async def _commit():
            call_order.append("commit")

        session.get = AsyncMock(return_value=fake_video)
        session.add = MagicMock(side_effect=_set_id)
        session.flush = AsyncMock()
        session.commit = AsyncMock(side_effect=_commit)
        yield session

    app.dependency_overrides[get_db] = override_get_db

    async def _fake_enqueue(*args, **kwargs):
        call_order.append("enqueue")

    with patch("apps.api.routes.videos.enqueue", side_effect=_fake_enqueue):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post("/videos/7/reprocess")

    app.dependency_overrides.clear()

    assert response.status_code == 201
    assert call_order.index("commit") < call_order.index("enqueue"), (
        "DB commit must happen before enqueue; got order: " + str(call_order)
    )


@pytest.mark.asyncio
async def test_reprocess_enqueues_a_new_job(app):
    """POST /videos/{id}/reprocess enqueues a new process_video job."""
    from libs.db import get_db
    from libs.models import Job, ProcessingRun, Video

    async def override_get_db():
        session = AsyncMock()
        fake_video = MagicMock(spec=Video)
        fake_video.id = 5
        fake_video.project_id = 1

        def _set_id(obj):
            if isinstance(obj, ProcessingRun):
                obj.id = 20
            elif isinstance(obj, Job):
                obj.id = 200

        session.get = AsyncMock(return_value=fake_video)
        session.add = MagicMock(side_effect=_set_id)
        session.flush = AsyncMock()
        session.commit = AsyncMock()
        yield session

    app.dependency_overrides[get_db] = override_get_db

    with patch("apps.api.routes.videos.enqueue", new_callable=AsyncMock) as mock_enqueue:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            await c.post("/videos/5/reprocess")

    app.dependency_overrides.clear()

    mock_enqueue.assert_called_once()
    call_args = mock_enqueue.call_args
    # Second positional arg is the job type
    assert call_args[0][1] == "process_video"
    # Payload must include the video_id
    assert call_args[0][2]["video_id"] == 5

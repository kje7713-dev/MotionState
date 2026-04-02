"""Tests for video upload and job creation."""

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def app():
    """Return the FastAPI app with external dependencies mocked."""
    with (
        patch("apps.api.main.engine") as mock_engine,
    ):
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
async def test_upload_video_creates_job(app, tmp_path):
    """POST /videos should return video_id and job_id."""
    # Build a fake video row and job row.
    fake_video = MagicMock()
    fake_video.id = 42

    fake_job = MagicMock()
    fake_job.id = 7

    mock_session = AsyncMock()
    mock_session.flush = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    # Simulate flush populating the id attributes.
    call_count = 0

    async def fake_flush():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # After first flush (video), make sure the video mock has an id
            pass

    mock_session.flush = fake_flush

    with (
        patch("apps.api.routes.videos.aiofiles.open", create=True) as mock_open,
        patch("apps.api.routes.videos.enqueue", new_callable=AsyncMock),
        patch("libs.db.AsyncSessionLocal", return_value=mock_session),
    ):
        # Mock the file write
        mock_file = AsyncMock()
        mock_open.return_value.__aenter__ = AsyncMock(return_value=mock_file)
        mock_open.return_value.__aexit__ = AsyncMock(return_value=False)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # Use a real DB session mock via dependency override.
            from libs.db import get_db

            async def override_get_db():
                from libs.models import Job, Video

                def _set_id(obj):
                    if isinstance(obj, Video):
                        obj.id = 42
                    elif isinstance(obj, Job):
                        obj.id = 7

                session = AsyncMock()
                session.add = MagicMock(side_effect=_set_id)
                session.flush = AsyncMock()
                session.commit = AsyncMock()

                yield session

            app.dependency_overrides[get_db] = override_get_db

            response = await client.post(
                "/videos",
                files={"file": ("test.mp4", io.BytesIO(b"fake video bytes"), "video/mp4")},
            )

    app.dependency_overrides.clear()

    assert response.status_code == 201
    data = response.json()
    assert "video_id" in data
    assert "job_id" in data


@pytest.mark.asyncio
async def test_get_video_not_found(app):
    """GET /videos/{id} returns 404 for a missing video."""
    from libs.db import get_db

    async def override_get_db():
        session = AsyncMock()
        session.get = AsyncMock(return_value=None)
        yield session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/videos/9999")

    app.dependency_overrides.clear()

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_job_not_found(app):
    """GET /jobs/{id} returns 404 for a missing job."""
    from libs.db import get_db

    async def override_get_db():
        session = AsyncMock()
        session.get = AsyncMock(return_value=None)
        yield session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/jobs/9999")

    app.dependency_overrides.clear()

    assert response.status_code == 404

"""Tests for video upload and job creation."""

import io
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

# ---------------------------------------------------------------------------
# Ensure boto3 is importable even when the [storage] extras are not installed.
# ---------------------------------------------------------------------------
if "boto3" not in sys.modules:
    sys.modules["boto3"] = MagicMock()


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


def _make_db_override(video_id: int = 42, job_id: int = 7):
    """Return a get_db override that assigns IDs on flush."""
    from libs.models import Job, Video

    added: list = []

    async def _get_db():
        def _set_id(obj):
            added.append(obj)
            if isinstance(obj, Video):
                obj.id = video_id
            elif isinstance(obj, Job):
                obj.id = job_id

        session = AsyncMock()
        session.add = MagicMock(side_effect=_set_id)
        session.flush = AsyncMock()
        session.commit = AsyncMock()
        yield session

    return _get_db, added


@pytest.mark.asyncio
async def test_upload_video_creates_job(app, tmp_path):
    """POST /videos should return video_id and job_id."""
    from libs.config import settings
    from libs.db import get_db

    db_override, _ = _make_db_override(video_id=42, job_id=7)
    app.dependency_overrides[get_db] = db_override

    with (
        patch.object(settings, "storage_backend", "local"),
        patch.object(settings, "upload_dir", str(tmp_path / "uploads")),
        patch("apps.api.routes.videos.enqueue", new_callable=AsyncMock),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
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
async def test_upload_video_local_backend_writes_to_disk(app, tmp_path):
    """POST /videos with local backend writes the file to the upload directory."""
    from libs.config import settings
    from libs.db import get_db

    db_override, added_objs = _make_db_override(video_id=10)
    app.dependency_overrides[get_db] = db_override

    upload_dir = tmp_path / "uploads"

    with (
        patch.object(settings, "storage_backend", "local"),
        patch.object(settings, "upload_dir", str(upload_dir)),
        patch.object(settings, "artifacts_dir", str(tmp_path / "artifacts")),
        patch("apps.api.routes.videos.enqueue", new_callable=AsyncMock),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/videos",
                files={"file": ("myvideo.mp4", io.BytesIO(b"local video bytes"), "video/mp4")},
            )

    app.dependency_overrides.clear()

    assert response.status_code == 201

    from libs.models import Video

    video_obj = next((o for o in added_objs if isinstance(o, Video)), None)
    assert video_obj is not None
    # source_path should be a local filesystem path inside the upload dir.
    assert video_obj.source_path is not None
    assert video_obj.source_path.startswith(str(upload_dir))
    # The uploaded file should exist on disk.
    assert (tmp_path / "uploads").exists()


@pytest.mark.asyncio
async def test_upload_video_s3_backend_stores_through_storage(app, tmp_path):
    """POST /videos with s3 backend stores bytes through the storage abstraction."""
    from libs.config import settings
    from libs.db import get_db
    from libs.storage.s3 import S3Storage

    db_override, added_objs = _make_db_override(video_id=5)
    app.dependency_overrides[get_db] = db_override

    saved_calls: list = []

    async def fake_save(self, data: bytes, key: str) -> str:
        saved_calls.append((key, data))
        return key

    with (
        patch.object(settings, "storage_backend", "s3"),
        patch.object(settings, "s3_bucket", "my-bucket"),
        patch.object(settings, "s3_region", "us-east-1"),
        patch.object(settings, "s3_endpoint_url", ""),
        patch.object(settings, "s3_access_key_id", "key"),
        patch.object(settings, "s3_secret_access_key", "secret"),
        patch.object(S3Storage, "save", new=fake_save),
        patch("boto3.client", return_value=MagicMock()),
        patch("apps.api.routes.videos.enqueue", new_callable=AsyncMock),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/videos",
                files={"file": ("game.mp4", io.BytesIO(b"s3 video bytes"), "video/mp4")},
            )

    app.dependency_overrides.clear()

    assert response.status_code == 201

    # Exactly one save call should have been made.
    assert len(saved_calls) == 1
    saved_key, saved_data = saved_calls[0]
    # Key follows canonical pattern: videos/{video_id}/source{ext}
    assert saved_key == "videos/5/source.mp4"
    assert saved_data == b"s3 video bytes"


@pytest.mark.asyncio
async def test_upload_video_s3_backend_sets_source_path_to_storage_key(app, tmp_path):
    """POST /videos with s3 backend sets video.source_path to the storage key."""
    from libs.config import settings
    from libs.db import get_db
    from libs.storage.s3 import S3Storage

    db_override, added_objs = _make_db_override(video_id=3)
    app.dependency_overrides[get_db] = db_override

    async def fake_save(self, data: bytes, key: str) -> str:
        return key

    with (
        patch.object(settings, "storage_backend", "s3"),
        patch.object(settings, "s3_bucket", "my-bucket"),
        patch.object(settings, "s3_region", "us-east-1"),
        patch.object(settings, "s3_endpoint_url", ""),
        patch.object(settings, "s3_access_key_id", "key"),
        patch.object(settings, "s3_secret_access_key", "secret"),
        patch.object(S3Storage, "save", new=fake_save),
        patch("boto3.client", return_value=MagicMock()),
        patch("apps.api.routes.videos.enqueue", new_callable=AsyncMock),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/videos",
                files={"file": ("match.mp4", io.BytesIO(b"bytes"), "video/mp4")},
            )

    app.dependency_overrides.clear()

    assert response.status_code == 201

    from libs.models import Video

    video_obj = next((o for o in added_objs if isinstance(o, Video)), None)
    assert video_obj is not None
    assert video_obj.source_path == "videos/3/source.mp4"


@pytest.mark.asyncio
async def test_upload_video_response_includes_processing_run_id(app, tmp_path):
    """POST /videos response includes processing_run_id."""
    from libs.config import settings
    from libs.db import get_db

    db_override, _ = _make_db_override(video_id=1, job_id=2)
    app.dependency_overrides[get_db] = db_override

    with (
        patch.object(settings, "storage_backend", "local"),
        patch.object(settings, "upload_dir", str(tmp_path / "uploads")),
        patch("apps.api.routes.videos.enqueue", new_callable=AsyncMock),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/videos",
                files={"file": ("clip.mp4", io.BytesIO(b"bytes"), "video/mp4")},
            )

    app.dependency_overrides.clear()

    assert response.status_code == 201
    data = response.json()
    assert "processing_run_id" in data


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

"""Tests for the POST /videos/upload-init endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

# ---------------------------------------------------------------------------
# App fixture (matches pattern used in other API tests)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# DB override helpers
# ---------------------------------------------------------------------------


def _make_db_override(video_id: int = 1):
    """Return a get_db override that yields a mock session."""
    from libs.models import Video

    created_video: list = []

    async def _get_db():
        session = AsyncMock()

        async def _flush():
            # Simulate auto-increment by assigning an id after flush.
            for obj in created_video:
                if not hasattr(obj, "_id_assigned"):
                    obj.id = video_id
                    obj._id_assigned = True

        def _add(obj):
            if isinstance(obj, Video):
                created_video.append(obj)

        session.add = MagicMock(side_effect=_add)
        session.flush = AsyncMock(side_effect=_flush)
        session.commit = AsyncMock()
        yield session

    return _get_db, created_video


# ---------------------------------------------------------------------------
# Tests: upload-init with local backend
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upload_init_returns_video_id(app, tmp_path):
    """POST /videos/upload-init returns a non-null video_id."""
    from libs.config import settings
    from libs.db import get_db

    db_override, _ = _make_db_override(video_id=7)
    app.dependency_overrides[get_db] = db_override

    with (
        patch.object(settings, "storage_backend", "local"),
        patch.object(settings, "artifacts_dir", str(tmp_path)),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post("/videos/upload-init", json={"filename": "myvideo.mp4"})

    app.dependency_overrides.clear()

    assert response.status_code == 201
    data = response.json()
    assert "video_id" in data
    assert data["video_id"] == 7


@pytest.mark.asyncio
async def test_upload_init_returns_storage_key(app, tmp_path):
    """POST /videos/upload-init returns a canonical storage_key."""
    from libs.config import settings
    from libs.db import get_db

    db_override, _ = _make_db_override(video_id=3)
    app.dependency_overrides[get_db] = db_override

    with (
        patch.object(settings, "storage_backend", "local"),
        patch.object(settings, "artifacts_dir", str(tmp_path)),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post("/videos/upload-init", json={"filename": "clip.mp4"})

    app.dependency_overrides.clear()

    assert response.status_code == 201
    data = response.json()
    assert "storage_key" in data
    assert data["storage_key"] == "videos/3/source.mp4"


@pytest.mark.asyncio
async def test_upload_init_upload_url_is_none_for_local(app, tmp_path):
    """POST /videos/upload-init returns upload_url=null for the local backend."""
    from libs.config import settings
    from libs.db import get_db

    db_override, _ = _make_db_override(video_id=1)
    app.dependency_overrides[get_db] = db_override

    with (
        patch.object(settings, "storage_backend", "local"),
        patch.object(settings, "artifacts_dir", str(tmp_path)),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post("/videos/upload-init", json={"filename": "test.mp4"})

    app.dependency_overrides.clear()

    assert response.status_code == 201
    assert response.json()["upload_url"] is None


@pytest.mark.asyncio
async def test_upload_init_upload_url_present_for_s3(app, tmp_path):
    """POST /videos/upload-init returns a signed upload_url for the S3 backend."""
    from libs.config import settings
    from libs.db import get_db
    from libs.storage.s3 import S3Storage

    db_override, _ = _make_db_override(video_id=5)
    app.dependency_overrides[get_db] = db_override

    fake_url = "https://bucket.s3.example.com/videos/5/source.mp4?sig=abc"

    with (
        patch.object(settings, "storage_backend", "s3"),
        patch.object(settings, "s3_bucket", "my-bucket"),
        patch.object(settings, "s3_region", "us-east-1"),
        patch.object(settings, "s3_endpoint_url", ""),
        patch.object(settings, "s3_access_key_id", "key"),
        patch.object(settings, "s3_secret_access_key", "secret"),
        patch.object(settings, "signed_url_expiration_seconds", 3600),
        # Patch the S3Storage.generate_upload_url to avoid real boto3 calls.
        patch.object(S3Storage, "generate_upload_url", new=AsyncMock(return_value=fake_url)),
        patch("boto3.client", return_value=MagicMock()),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post("/videos/upload-init", json={"filename": "game.mp4"})

    app.dependency_overrides.clear()

    assert response.status_code == 201
    data = response.json()
    assert data["upload_url"] == fake_url


@pytest.mark.asyncio
async def test_upload_init_preserves_extension_in_key(app, tmp_path):
    """POST /videos/upload-init uses the original file extension in the storage key."""
    from libs.config import settings
    from libs.db import get_db

    db_override, _ = _make_db_override(video_id=9)
    app.dependency_overrides[get_db] = db_override

    with (
        patch.object(settings, "storage_backend", "local"),
        patch.object(settings, "artifacts_dir", str(tmp_path)),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post("/videos/upload-init", json={"filename": "match.mov"})

    app.dependency_overrides.clear()

    assert response.status_code == 201
    key = response.json()["storage_key"]
    assert key == "videos/9/source.mov"


@pytest.mark.asyncio
async def test_upload_init_response_has_all_fields(app, tmp_path):
    """POST /videos/upload-init response contains video_id, upload_url, storage_key."""
    from libs.config import settings
    from libs.db import get_db

    db_override, _ = _make_db_override(video_id=2)
    app.dependency_overrides[get_db] = db_override

    with (
        patch.object(settings, "storage_backend", "local"),
        patch.object(settings, "artifacts_dir", str(tmp_path)),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post("/videos/upload-init", json={"filename": "test.mp4"})

    app.dependency_overrides.clear()

    assert response.status_code == 201
    data = response.json()
    assert set(data.keys()) >= {"video_id", "upload_url", "storage_key"}

"""Tests for API key authentication.

Covers:
- valid API key can access project resources
- missing key returns 401
- invalid key returns 401
- inactive key returns 401
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from libs.auth import generate_api_key, hash_api_key

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


def _make_fake_api_key(project_id: int = 1, is_active: bool = True, key: str | None = None):
    """Return a MagicMock ApiKey row."""
    from libs.models import ApiKey

    raw_key = key or generate_api_key()[0]
    key_hash = hash_api_key(raw_key)

    ak = MagicMock(spec=ApiKey)
    ak.id = 1
    ak.project_id = project_id
    ak.is_active = is_active
    ak.key_hash = key_hash
    ak.last_used_at = None
    return ak, raw_key


def _make_fake_project(project_id: int = 1):
    from libs.models import Project

    p = MagicMock(spec=Project)
    p.id = project_id
    p.name = "test-project"
    return p


def _make_fake_video(video_id: int = 1, project_id: int = 1):
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


def _db_override_with_auth(api_key_row=None, project_row=None, video_row=None):
    """Return a get_db override that yields a mock session with configurable lookups."""

    def _make_execute_result(obj):
        mock_scalars = MagicMock()
        mock_scalars.first.return_value = obj
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        return mock_result

    async def _get_db():
        session = AsyncMock()

        def _get_side_effect(model, pk):
            model_name = model.__name__
            if model_name == "Project":
                return project_row
            if model_name == "Video":
                return video_row
            return None

        session.get = AsyncMock(side_effect=_get_side_effect)
        session.execute = AsyncMock(return_value=_make_execute_result(api_key_row))
        yield session

    return _get_db


# ---------------------------------------------------------------------------
# Missing key → 401
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_api_key_returns_401(app):
    """GET /videos/{id} without any API key returns 401."""
    from libs.db import get_db

    app.dependency_overrides[get_db] = _db_override_with_auth(
        api_key_row=None, project_row=None, video_row=None
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/videos/1")

    app.dependency_overrides.clear()

    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Invalid key → 401
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_api_key_returns_401(app):
    """GET /videos/{id} with a key that doesn't exist in DB returns 401."""
    from libs.db import get_db

    # No matching key in DB
    app.dependency_overrides[get_db] = _db_override_with_auth(
        api_key_row=None, project_row=None, video_row=None
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/videos/1", headers={"X-API-Key": "ms_live_doesnotexist"})

    app.dependency_overrides.clear()

    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Inactive key → 401
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inactive_api_key_returns_401(app):
    """GET /videos/{id} with an inactive key returns 401."""
    from libs.db import get_db

    api_key, raw_key = _make_fake_api_key(is_active=False)
    project = _make_fake_project()

    app.dependency_overrides[get_db] = _db_override_with_auth(
        api_key_row=api_key, project_row=project, video_row=None
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/videos/1", headers={"X-API-Key": raw_key})

    app.dependency_overrides.clear()

    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Valid key → can access project resource
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_valid_api_key_can_access_video(app):
    """GET /videos/{id} with a valid API key for the owning project returns 200."""
    from libs.db import get_db

    api_key, raw_key = _make_fake_api_key(project_id=1)
    project = _make_fake_project(project_id=1)
    video = _make_fake_video(video_id=1, project_id=1)

    app.dependency_overrides[get_db] = _db_override_with_auth(
        api_key_row=api_key, project_row=project, video_row=video
    )

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/videos/1", headers={"X-API-Key": raw_key})

    app.dependency_overrides.clear()

    assert response.status_code == 200


# ---------------------------------------------------------------------------
# generate_api_key / hash_api_key unit tests
# ---------------------------------------------------------------------------


def test_generate_api_key_format():
    """Generated key starts with ms_live_ prefix."""
    raw_key, prefix, key_hash = generate_api_key()
    assert raw_key.startswith("ms_live_")
    assert len(key_hash) == 64  # SHA-256 hex digest


def test_hash_api_key_is_deterministic():
    """The same raw key always produces the same hash."""
    raw_key = "ms_live_testkey123"
    assert hash_api_key(raw_key) == hash_api_key(raw_key)


def test_different_keys_have_different_hashes():
    """Two different raw keys produce different hashes."""
    raw1, _, _ = generate_api_key()
    raw2, _, _ = generate_api_key()
    assert hash_api_key(raw1) != hash_api_key(raw2)

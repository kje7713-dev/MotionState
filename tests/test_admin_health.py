"""Tests for GET /admin/health/summary endpoint."""

from __future__ import annotations

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
# Helper: minimal DB override that passes SELECT 1
# ---------------------------------------------------------------------------


def _db_override_ok():
    async def _get_db():
        session = AsyncMock()
        session.execute = AsyncMock(return_value=MagicMock())
        yield session

    return _get_db


def _db_override_fail():
    async def _get_db():
        session = AsyncMock()
        session.execute = AsyncMock(side_effect=Exception("db down"))
        yield session

    return _get_db


# ---------------------------------------------------------------------------
# Tests: admin token enforcement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_summary_requires_admin_token(app):
    """GET /admin/health/summary returns 403 when admin_token is not configured."""
    from libs.db import get_db

    app.dependency_overrides[get_db] = _db_override_ok()

    with patch("apps.api.routes.admin.settings") as mock_settings:
        mock_settings.admin_token = ""
        mock_settings.redis_url = "redis://localhost:6379/0"
        mock_settings.storage_backend = "local"
        mock_settings.artifacts_dir = "/tmp/artifacts"

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/admin/health/summary")

    app.dependency_overrides.clear()
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_health_summary_rejects_wrong_token(app):
    """GET /admin/health/summary returns 403 for a wrong token."""
    from libs.db import get_db

    app.dependency_overrides[get_db] = _db_override_ok()

    with patch("apps.api.routes.admin.settings") as mock_settings:
        mock_settings.admin_token = "correct-token"
        mock_settings.redis_url = "redis://localhost:6379/0"
        mock_settings.storage_backend = "local"
        mock_settings.artifacts_dir = "/tmp/artifacts"

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/admin/health/summary",
                headers={"X-Admin-Token": "wrong-token"},
            )

    app.dependency_overrides.clear()
    assert response.status_code == 403


# ---------------------------------------------------------------------------
# Tests: health summary structure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_summary_returns_200_with_valid_token(app):
    """GET /admin/health/summary returns 200 when the correct token is supplied."""
    from libs.db import get_db

    app.dependency_overrides[get_db] = _db_override_ok()

    with (
        patch("apps.api.routes.admin.settings") as mock_settings,
        patch("apps.api.routes.admin.get_storage") as mock_get_storage,
    ):
        mock_settings.admin_token = "secret"
        mock_settings.redis_url = "redis://localhost:6379/0"
        mock_settings.storage_backend = "local"
        mock_settings.artifacts_dir = "/tmp/artifacts"

        mock_storage = AsyncMock()
        mock_storage.check_reachable = AsyncMock()
        mock_get_storage.return_value = mock_storage

        # Mock redis ping
        mock_redis_client = AsyncMock()
        mock_redis_client.ping = AsyncMock()
        mock_redis_client.aclose = AsyncMock()

        with patch("apps.api.routes.admin.aioredis") as mock_aioredis:
            mock_aioredis.from_url.return_value = mock_redis_client

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get(
                    "/admin/health/summary",
                    headers={"X-Admin-Token": "secret"},
                )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "checks" in data
    assert "checked_at" in data


@pytest.mark.asyncio
async def test_health_summary_includes_db_check(app):
    """Health summary includes a 'db' check key."""
    from libs.db import get_db

    app.dependency_overrides[get_db] = _db_override_ok()

    with (
        patch("apps.api.routes.admin.settings") as mock_settings,
        patch("apps.api.routes.admin.get_storage") as mock_get_storage,
        patch("apps.api.routes.admin.aioredis") as mock_aioredis,
    ):
        mock_settings.admin_token = "secret"
        mock_settings.redis_url = "redis://localhost:6379/0"
        mock_settings.storage_backend = "local"
        mock_settings.artifacts_dir = "/tmp/artifacts"

        mock_storage = AsyncMock()
        mock_storage.check_reachable = AsyncMock()
        mock_get_storage.return_value = mock_storage

        mock_redis_client = AsyncMock()
        mock_redis_client.ping = AsyncMock()
        mock_redis_client.aclose = AsyncMock()
        mock_aioredis.from_url.return_value = mock_redis_client

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/admin/health/summary",
                headers={"X-Admin-Token": "secret"},
            )

    app.dependency_overrides.clear()
    data = response.json()
    assert "db" in data["checks"]
    assert "redis" in data["checks"]
    assert "storage" in data["checks"]
    assert "app" in data["checks"]


@pytest.mark.asyncio
async def test_health_summary_db_failure_reflected(app):
    """Health summary reflects 'degraded' status when DB is unreachable."""
    from libs.db import get_db

    app.dependency_overrides[get_db] = _db_override_fail()

    with (
        patch("apps.api.routes.admin.settings") as mock_settings,
        patch("apps.api.routes.admin.get_storage") as mock_get_storage,
        patch("apps.api.routes.admin.aioredis") as mock_aioredis,
    ):
        mock_settings.admin_token = "secret"
        mock_settings.redis_url = "redis://localhost:6379/0"
        mock_settings.storage_backend = "local"
        mock_settings.artifacts_dir = "/tmp/artifacts"

        mock_storage = AsyncMock()
        mock_storage.check_reachable = AsyncMock()
        mock_get_storage.return_value = mock_storage

        mock_redis_client = AsyncMock()
        mock_redis_client.ping = AsyncMock()
        mock_redis_client.aclose = AsyncMock()
        mock_aioredis.from_url.return_value = mock_redis_client

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/admin/health/summary",
                headers={"X-Admin-Token": "secret"},
            )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["checks"]["db"].startswith("error:")

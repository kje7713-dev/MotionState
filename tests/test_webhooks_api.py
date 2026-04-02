"""Tests for the webhook management API.

Covers:
- POST /projects/{id}/webhooks creates a webhook and returns secret
- GET /projects/{id}/webhooks lists webhooks (without secret)
- PATCH /projects/{id}/webhooks/{id} updates a webhook
- DELETE /projects/{id}/webhooks/{id} deletes a webhook
- 404 for missing project or webhook
- Cross-project webhook access blocked
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


def _make_fake_project(project_id: int = 1, name: str = "Test Project"):
    from libs.models import Project

    p = MagicMock(spec=Project)
    p.id = project_id
    p.name = name
    p.created_at = datetime(2024, 1, 1, tzinfo=UTC)
    p.updated_at = datetime(2024, 1, 1, tzinfo=UTC)
    return p


def _make_fake_webhook(webhook_id: int = 1, project_id: int = 1, url: str = "https://example.com/hook"):
    from libs.models import WebhookEndpoint

    w = MagicMock(spec=WebhookEndpoint)
    w.id = webhook_id
    w.project_id = project_id
    w.url = url
    w.secret = "fake-secret"
    w.is_active = True
    w.event_types_json = None
    w.created_at = datetime(2024, 1, 1, tzinfo=UTC)
    w.updated_at = datetime(2024, 1, 1, tzinfo=UTC)
    w.last_success_at = None
    w.last_failure_at = None
    return w


def _db_override(project=None, webhook=None, webhooks=None):
    """Return a get_db override for webhook management tests."""
    created_objects: list = []

    mock_scalars = MagicMock()
    mock_scalars.all.return_value = webhooks or []
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars

    async def _get_db():
        session = AsyncMock()

        def _set_id(obj):
            from libs.models import WebhookEndpoint

            created_objects.append(obj)
            if isinstance(obj, WebhookEndpoint):
                obj.id = 1
                obj.created_at = datetime(2024, 1, 1, tzinfo=UTC)
                obj.updated_at = datetime(2024, 1, 1, tzinfo=UTC)

        def _get_side(model, pk):
            from libs.models import Project, WebhookEndpoint

            if issubclass(model, Project):
                return project
            if issubclass(model, WebhookEndpoint):
                return webhook
            return None

        session.get = AsyncMock(side_effect=_get_side)
        session.add = MagicMock(side_effect=_set_id)
        session.delete = AsyncMock()
        session.flush = AsyncMock()
        session.commit = AsyncMock()
        session.execute = AsyncMock(return_value=mock_result)
        yield session

    return _get_db, created_objects


# ---------------------------------------------------------------------------
# POST /projects/{id}/webhooks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_webhook_returns_201(app):
    """POST /projects/{id}/webhooks returns 201."""
    from libs.db import get_db

    fake_project = _make_fake_project(project_id=1)
    db_override, _ = _db_override(project=fake_project)
    app.dependency_overrides[get_db] = db_override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/projects/1/webhooks",
            json={"url": "https://example.com/hook"},
        )

    app.dependency_overrides.clear()
    assert response.status_code == 201


@pytest.mark.asyncio
async def test_create_webhook_returns_secret(app):
    """POST /projects/{id}/webhooks response includes the signing secret."""
    from libs.db import get_db

    fake_project = _make_fake_project(project_id=1)
    db_override, _ = _db_override(project=fake_project)
    app.dependency_overrides[get_db] = db_override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/projects/1/webhooks",
            json={"url": "https://example.com/hook"},
        )

    app.dependency_overrides.clear()
    data = response.json()
    assert "secret" in data
    assert len(data["secret"]) == 64  # 32 bytes as hex


@pytest.mark.asyncio
async def test_create_webhook_stores_url(app):
    """POST /projects/{id}/webhooks stores the provided URL."""
    from libs.db import get_db

    fake_project = _make_fake_project(project_id=1)
    db_override, created = _db_override(project=fake_project)
    app.dependency_overrides[get_db] = db_override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await client.post(
            "/projects/1/webhooks",
            json={"url": "https://hooks.example.com/v1/receive"},
        )

    app.dependency_overrides.clear()
    from libs.models import WebhookEndpoint

    webhook_rows = [o for o in created if isinstance(o, WebhookEndpoint)]
    assert len(webhook_rows) == 1
    assert webhook_rows[0].url == "https://hooks.example.com/v1/receive"


@pytest.mark.asyncio
async def test_create_webhook_with_event_types(app):
    """POST /projects/{id}/webhooks stores event_types filter."""
    from libs.db import get_db

    fake_project = _make_fake_project(project_id=1)
    db_override, created = _db_override(project=fake_project)
    app.dependency_overrides[get_db] = db_override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/projects/1/webhooks",
            json={
                "url": "https://example.com/hook",
                "event_types": ["processing_run.completed", "processing_run.failed"],
            },
        )

    app.dependency_overrides.clear()
    assert response.status_code == 201
    data = response.json()
    assert data["event_types"] == ["processing_run.completed", "processing_run.failed"]


@pytest.mark.asyncio
async def test_create_webhook_returns_404_for_missing_project(app):
    """POST /projects/{id}/webhooks returns 404 when project does not exist."""
    from libs.db import get_db

    db_override, _ = _db_override(project=None)
    app.dependency_overrides[get_db] = db_override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/projects/9999/webhooks",
            json={"url": "https://example.com/hook"},
        )

    app.dependency_overrides.clear()
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /projects/{id}/webhooks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_webhooks_returns_200(app):
    """GET /projects/{id}/webhooks returns 200 with a list."""
    from libs.db import get_db

    fake_project = _make_fake_project(project_id=1)
    w1 = _make_fake_webhook(webhook_id=1, project_id=1)
    w2 = _make_fake_webhook(webhook_id=2, project_id=1, url="https://other.example.com/hook")
    db_override, _ = _db_override(project=fake_project, webhooks=[w1, w2])
    app.dependency_overrides[get_db] = db_override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/projects/1/webhooks")

    app.dependency_overrides.clear()
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2


@pytest.mark.asyncio
async def test_list_webhooks_does_not_include_secret(app):
    """GET /projects/{id}/webhooks does NOT include the signing secret."""
    from libs.db import get_db

    fake_project = _make_fake_project(project_id=1)
    w = _make_fake_webhook(webhook_id=1, project_id=1)
    db_override, _ = _db_override(project=fake_project, webhooks=[w])
    app.dependency_overrides[get_db] = db_override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/projects/1/webhooks")

    app.dependency_overrides.clear()
    data = response.json()
    assert len(data) == 1
    assert "secret" not in data[0]


@pytest.mark.asyncio
async def test_list_webhooks_returns_404_for_missing_project(app):
    """GET /projects/{id}/webhooks returns 404 when project does not exist."""
    from libs.db import get_db

    db_override, _ = _db_override(project=None)
    app.dependency_overrides[get_db] = db_override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/projects/9999/webhooks")

    app.dependency_overrides.clear()
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# PATCH /projects/{id}/webhooks/{webhook_id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_webhook_returns_200(app):
    """PATCH /projects/{id}/webhooks/{webhook_id} returns 200."""
    from libs.db import get_db

    fake_project = _make_fake_project(project_id=1)
    fake_webhook = _make_fake_webhook(webhook_id=5, project_id=1)
    db_override, _ = _db_override(project=fake_project, webhook=fake_webhook)
    app.dependency_overrides[get_db] = db_override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.patch(
            "/projects/1/webhooks/5",
            json={"is_active": False},
        )

    app.dependency_overrides.clear()
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_update_webhook_sets_is_active(app):
    """PATCH /projects/{id}/webhooks/{webhook_id} updates is_active."""
    from libs.db import get_db

    fake_project = _make_fake_project(project_id=1)
    fake_webhook = _make_fake_webhook(webhook_id=5, project_id=1)
    db_override, _ = _db_override(project=fake_project, webhook=fake_webhook)
    app.dependency_overrides[get_db] = db_override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await client.patch("/projects/1/webhooks/5", json={"is_active": False})

    app.dependency_overrides.clear()
    assert fake_webhook.is_active is False


@pytest.mark.asyncio
async def test_update_webhook_cross_project_returns_404(app):
    """PATCH returns 404 when the webhook belongs to a different project."""
    from libs.db import get_db

    fake_project = _make_fake_project(project_id=1)
    # Webhook belongs to project 2, not project 1.
    fake_webhook = _make_fake_webhook(webhook_id=5, project_id=2)
    db_override, _ = _db_override(project=fake_project, webhook=fake_webhook)
    app.dependency_overrides[get_db] = db_override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.patch(
            "/projects/1/webhooks/5",
            json={"is_active": False},
        )

    app.dependency_overrides.clear()
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /projects/{id}/webhooks/{webhook_id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_webhook_returns_204(app):
    """DELETE /projects/{id}/webhooks/{webhook_id} returns 204."""
    from libs.db import get_db

    fake_project = _make_fake_project(project_id=1)
    fake_webhook = _make_fake_webhook(webhook_id=3, project_id=1)
    db_override, _ = _db_override(project=fake_project, webhook=fake_webhook)
    app.dependency_overrides[get_db] = db_override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.delete("/projects/1/webhooks/3")

    app.dependency_overrides.clear()
    assert response.status_code == 204


@pytest.mark.asyncio
async def test_delete_webhook_cross_project_returns_404(app):
    """DELETE returns 404 when the webhook belongs to a different project."""
    from libs.db import get_db

    fake_project = _make_fake_project(project_id=1)
    fake_webhook = _make_fake_webhook(webhook_id=3, project_id=2)
    db_override, _ = _db_override(project=fake_project, webhook=fake_webhook)
    app.dependency_overrides[get_db] = db_override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.delete("/projects/1/webhooks/3")

    app.dependency_overrides.clear()
    assert response.status_code == 404

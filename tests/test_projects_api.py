"""Tests for the project management API.

Covers:
- POST /projects creates a project
- GET /projects/{id} returns project metadata
- POST /projects/{id}/api-keys returns raw key once
- GET /projects/{id}/api-keys lists keys without raw key
- 404 for missing project
"""

from __future__ import annotations

from datetime import UTC, datetime
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


def _make_fake_project(project_id: int = 1, name: str = "My Project"):
    from libs.models import Project

    p = MagicMock(spec=Project)
    p.id = project_id
    p.name = name
    p.created_at = datetime(2024, 1, 1, tzinfo=UTC)
    p.updated_at = datetime(2024, 1, 1, tzinfo=UTC)
    return p


def _make_fake_api_key_row(key_id: int = 1, project_id: int = 1, name: str = "prod"):
    from libs.models import ApiKey

    ak = MagicMock(spec=ApiKey)
    ak.id = key_id
    ak.project_id = project_id
    ak.name = name
    ak.key_prefix = "ms_live_abcd"
    raw_key, prefix, key_hash = generate_api_key()
    ak.key_hash = key_hash
    ak.is_active = True
    ak.created_at = datetime(2024, 1, 1, tzinfo=UTC)
    ak.last_used_at = None
    return ak, raw_key


def _db_override_for_projects(project=None, api_keys=None):
    """Return a get_db override for project management tests."""
    created_objects: list = []

    mock_scalars = MagicMock()
    mock_scalars.all.return_value = api_keys or []
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars

    async def _get_db():
        session = AsyncMock()

        def _set_id(obj):
            from libs.models import ApiKey, Project

            created_objects.append(obj)
            if isinstance(obj, Project):
                obj.id = 10
                obj.created_at = datetime(2024, 1, 1, tzinfo=UTC)
                obj.updated_at = datetime(2024, 1, 1, tzinfo=UTC)
            elif isinstance(obj, ApiKey):
                obj.id = 1
                obj.created_at = datetime(2024, 1, 1, tzinfo=UTC)

        session.get = AsyncMock(return_value=project)
        session.add = MagicMock(side_effect=_set_id)
        session.flush = AsyncMock()
        session.commit = AsyncMock()
        session.execute = AsyncMock(return_value=mock_result)
        yield session

    return _get_db, created_objects


# ---------------------------------------------------------------------------
# POST /projects
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_project_returns_201(app):
    """POST /projects returns 201 and project data."""
    from libs.db import get_db

    db_override, _ = _db_override_for_projects(project=None)
    app.dependency_overrides[get_db] = db_override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/projects?name=TestProject")

    app.dependency_overrides.clear()

    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["id"] == 10
    assert "name" in data


@pytest.mark.asyncio
async def test_create_project_sets_name(app):
    """POST /projects stores the given name."""
    from libs.db import get_db

    db_override, created = _db_override_for_projects(project=None)
    app.dependency_overrides[get_db] = db_override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await client.post("/projects?name=MyApp")

    app.dependency_overrides.clear()

    from libs.models import Project

    project_rows = [o for o in created if isinstance(o, Project)]
    assert len(project_rows) == 1
    assert project_rows[0].name == "MyApp"


# ---------------------------------------------------------------------------
# GET /projects/{id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_project_returns_200(app):
    """GET /projects/{id} returns 200 with project metadata."""
    from libs.db import get_db

    fake_project = _make_fake_project(project_id=5, name="Cool Project")
    db_override, _ = _db_override_for_projects(project=fake_project)
    app.dependency_overrides[get_db] = db_override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/projects/5")

    app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == 5
    assert data["name"] == "Cool Project"
    assert "created_at" in data


@pytest.mark.asyncio
async def test_get_project_returns_404_for_missing(app):
    """GET /projects/{id} returns 404 when the project doesn't exist."""
    from libs.db import get_db

    db_override, _ = _db_override_for_projects(project=None)
    app.dependency_overrides[get_db] = db_override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/projects/9999")

    app.dependency_overrides.clear()

    assert response.status_code == 404


# ---------------------------------------------------------------------------
# POST /projects/{id}/api-keys
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_api_key_returns_201(app):
    """POST /projects/{id}/api-keys returns 201 with raw key."""
    from libs.db import get_db

    fake_project = _make_fake_project(project_id=1)
    db_override, _ = _db_override_for_projects(project=fake_project)
    app.dependency_overrides[get_db] = db_override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/projects/1/api-keys?name=production")

    app.dependency_overrides.clear()

    assert response.status_code == 201


@pytest.mark.asyncio
async def test_create_api_key_returns_raw_key(app):
    """POST /projects/{id}/api-keys response includes the raw key."""
    from libs.db import get_db

    fake_project = _make_fake_project(project_id=1)
    db_override, _ = _db_override_for_projects(project=fake_project)
    app.dependency_overrides[get_db] = db_override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/projects/1/api-keys?name=ci")

    app.dependency_overrides.clear()

    data = response.json()
    assert "key" in data
    assert data["key"].startswith("ms_live_")


@pytest.mark.asyncio
async def test_create_api_key_raw_key_not_stored(app):
    """The raw API key in the response is NOT the hash stored in the DB."""
    from libs.db import get_db

    fake_project = _make_fake_project(project_id=1)
    db_override, created = _db_override_for_projects(project=fake_project)
    app.dependency_overrides[get_db] = db_override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/projects/1/api-keys?name=test")

    app.dependency_overrides.clear()

    from libs.models import ApiKey

    raw_key_in_response = response.json()["key"]
    api_key_rows = [o for o in created if isinstance(o, ApiKey)]
    assert len(api_key_rows) == 1
    # The raw key is NOT stored directly
    assert api_key_rows[0].key_hash != raw_key_in_response
    # But hashing the returned key should match what was stored
    assert api_key_rows[0].key_hash == hash_api_key(raw_key_in_response)


@pytest.mark.asyncio
async def test_create_api_key_returns_404_for_missing_project(app):
    """POST /projects/{id}/api-keys returns 404 when the project doesn't exist."""
    from libs.db import get_db

    db_override, _ = _db_override_for_projects(project=None)
    app.dependency_overrides[get_db] = db_override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/projects/9999/api-keys?name=test")

    app.dependency_overrides.clear()

    assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /projects/{id}/api-keys
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_api_keys_returns_200(app):
    """GET /projects/{id}/api-keys returns 200 with a list of key metadata."""
    from libs.db import get_db

    fake_project = _make_fake_project(project_id=2)
    ak1, _ = _make_fake_api_key_row(key_id=1, project_id=2, name="prod")
    ak2, _ = _make_fake_api_key_row(key_id=2, project_id=2, name="dev")
    db_override, _ = _db_override_for_projects(project=fake_project, api_keys=[ak1, ak2])
    app.dependency_overrides[get_db] = db_override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/projects/2/api-keys")

    app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2


@pytest.mark.asyncio
async def test_list_api_keys_does_not_include_raw_key(app):
    """GET /projects/{id}/api-keys response does NOT include the raw key field."""
    from libs.db import get_db

    fake_project = _make_fake_project(project_id=1)
    ak, _ = _make_fake_api_key_row(key_id=1, project_id=1, name="prod")
    db_override, _ = _db_override_for_projects(project=fake_project, api_keys=[ak])
    app.dependency_overrides[get_db] = db_override

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/projects/1/api-keys")

    app.dependency_overrides.clear()

    data = response.json()
    assert len(data) == 1
    assert "key" not in data[0]
    assert "key_hash" not in data[0]
    assert "key_prefix" in data[0]
    assert "is_active" in data[0]

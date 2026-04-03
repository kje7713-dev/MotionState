"""Tests that the OpenAPI schema exposes both security schemes correctly.

Verifies:
- ``ApiKeyAuth`` (X-API-Key) appears in the security scheme components.
- ``AdminTokenAuth`` (X-Admin-Token) appears in the security scheme components.
- Project routes declare ``ApiKeyAuth`` as their security requirement.
- Admin routes declare ``AdminTokenAuth`` as their security requirement.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def app():
    """Return the FastAPI app with DB engine mocked."""
    from unittest.mock import patch

    with patch("apps.api.main.engine") as mock_engine:
        mock_conn = AsyncMock()
        mock_conn.run_sync = AsyncMock()
        mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=False)

        from apps.api.main import app as fastapi_app

        yield fastapi_app


@pytest.mark.asyncio
async def test_openapi_has_api_key_auth_scheme(app):
    """OpenAPI components contain an ApiKeyAuth security scheme for X-API-Key."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()

    schemes = schema.get("components", {}).get("securitySchemes", {})
    assert "ApiKeyAuth" in schemes, (
        f"Expected 'ApiKeyAuth' in securitySchemes, got: {list(schemes.keys())}"
    )
    api_key_scheme = schemes["ApiKeyAuth"]
    assert api_key_scheme["type"] == "apiKey"
    assert api_key_scheme["in"] == "header"
    assert api_key_scheme["name"] == "X-API-Key"


@pytest.mark.asyncio
async def test_openapi_has_admin_token_auth_scheme(app):
    """OpenAPI components contain an AdminTokenAuth security scheme for X-Admin-Token."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()

    schemes = schema.get("components", {}).get("securitySchemes", {})
    assert "AdminTokenAuth" in schemes, (
        f"Expected 'AdminTokenAuth' in securitySchemes, got: {list(schemes.keys())}"
    )
    admin_scheme = schemes["AdminTokenAuth"]
    assert admin_scheme["type"] == "apiKey"
    assert admin_scheme["in"] == "header"
    assert admin_scheme["name"] == "X-Admin-Token"


@pytest.mark.asyncio
async def test_openapi_both_schemes_present(app):
    """Both ApiKeyAuth and AdminTokenAuth appear as distinct security schemes."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()

    schemes = schema.get("components", {}).get("securitySchemes", {})
    assert "ApiKeyAuth" in schemes
    assert "AdminTokenAuth" in schemes
    # They must be distinct entries
    assert schemes["ApiKeyAuth"] != schemes["AdminTokenAuth"]


@pytest.mark.asyncio
async def test_project_routes_use_api_key_scheme(app):
    """GET /videos/{id} declares ApiKeyAuth (not AdminTokenAuth) as its security."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()

    paths = schema.get("paths", {})
    video_get = paths.get("/videos/{video_id}", {}).get("get", {})
    security = video_get.get("security", [])
    scheme_names = [list(s.keys())[0] for s in security if s]
    assert "ApiKeyAuth" in scheme_names, (
        f"Expected ApiKeyAuth on GET /videos/{{video_id}}, got security: {security}"
    )


@pytest.mark.asyncio
async def test_admin_routes_use_admin_token_scheme(app):
    """GET /admin/health/summary declares AdminTokenAuth as its security."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()

    paths = schema.get("paths", {})
    admin_get = paths.get("/admin/health/summary", {}).get("get", {})
    security = admin_get.get("security", [])
    scheme_names = [list(s.keys())[0] for s in security if s]
    assert "AdminTokenAuth" in scheme_names, (
        f"Expected AdminTokenAuth on GET /admin/health/summary, got security: {security}"
    )

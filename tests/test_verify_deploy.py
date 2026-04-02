"""Tests for the deploy verification script helpers.

These tests exercise the pure-Python check helpers in scripts/verify_deploy.py
using httpx mock transport so no running server is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

# Make scripts/ importable when running from the repo root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.verify_deploy import check_admin_health, check_api_smoke, check_public_health

# ---------------------------------------------------------------------------
# Helpers to build a mock httpx transport
# ---------------------------------------------------------------------------


class _MockTransport(httpx.BaseTransport):
    """Simple mock transport that returns pre-configured responses."""

    def __init__(self, routes: dict[tuple[str, str], tuple[int, dict]]) -> None:
        # routes: {(method, path): (status_code, json_body)}
        self._routes = routes

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        key = (request.method, request.url.path)
        if key in self._routes:
            status, body = self._routes[key]
            return httpx.Response(status, json=body)
        return httpx.Response(404, json={"detail": "not found"})


def _make_client(routes: dict) -> httpx.Client:
    return httpx.Client(transport=_MockTransport(routes), base_url="http://testserver")


class _ErrorTransport(httpx.BaseTransport):
    """Mock transport that always raises a ConnectError."""

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")


def _make_error_client() -> httpx.Client:
    return httpx.Client(transport=_ErrorTransport(), base_url="http://testserver")


# ---------------------------------------------------------------------------
# check_public_health
# ---------------------------------------------------------------------------


def test_public_health_ok():
    client = _make_client({("GET", "/health"): (200, {"status": "ok"})})
    assert check_public_health(client) is True


def test_public_health_wrong_status():
    client = _make_client({("GET", "/health"): (200, {"status": "degraded"})})
    assert check_public_health(client) is False


def test_public_health_server_error():
    client = _make_client({("GET", "/health"): (500, {"detail": "internal error"})})
    assert check_public_health(client) is False


def test_public_health_connection_error():
    """Connection errors are caught and return False."""
    assert check_public_health(_make_error_client()) is False


# ---------------------------------------------------------------------------
# check_admin_health
# ---------------------------------------------------------------------------


def test_admin_health_all_ok():
    body = {"app": "ok", "db": "ok", "redis": "ok", "storage": "ok"}
    client = _make_client({("GET", "/admin/health/summary"): (200, body)})
    assert check_admin_health(client, "secret-token") is True


def test_admin_health_one_failed():
    body = {"app": "ok", "db": "error: connection refused", "redis": "ok", "storage": "ok"}
    client = _make_client({("GET", "/admin/health/summary"): (200, body)})
    assert check_admin_health(client, "secret-token") is False


def test_admin_health_403_wrong_token():
    client = _make_client({("GET", "/admin/health/summary"): (403, {"detail": "forbidden"})})
    assert check_admin_health(client, "wrong-token") is False


def test_admin_health_server_error():
    client = _make_client({("GET", "/admin/health/summary"): (500, {"detail": "error"})})
    assert check_admin_health(client, "token") is False


def test_admin_health_connection_error():
    assert check_admin_health(_make_error_client(), "token") is False


# ---------------------------------------------------------------------------
# check_api_smoke
# ---------------------------------------------------------------------------


def test_api_smoke_success():
    client = _make_client({("GET", "/health"): (200, {"status": "ok"})})
    assert check_api_smoke(client, "ms_live_testkey") is True


def test_api_smoke_auth_failure():
    client = _make_client({("GET", "/health"): (401, {"detail": "Unauthorized"})})
    assert check_api_smoke(client, "bad-key") is False


def test_api_smoke_connection_error():
    assert check_api_smoke(_make_error_client(), "ms_live_testkey") is False

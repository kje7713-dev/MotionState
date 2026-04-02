"""Tests for MotionState Python SDK – error mapping."""

from __future__ import annotations

import json
import os
import sys

import httpx
import pytest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "sdk", "python"),
)

from motionstate_client import MotionStateClient
from motionstate_client.errors import (
    AuthError,
    MotionStateError,
    NotFoundError,
    QuotaError,
    ServerError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(status_code: int, body: object) -> httpx.Response:
    content = json.dumps(body).encode()
    return httpx.Response(
        status_code=status_code,
        headers={"content-type": "application/json"},
        content=content,
    )


def _client_with_status(status_code: int, body: object) -> MotionStateClient:
    def handler(_: httpx.Request) -> httpx.Response:
        return _make_response(status_code, body)

    client = MotionStateClient(base_url="http://test", api_key="key")
    client._http = httpx.Client(
        base_url="http://test",
        headers={"X-API-Key": "key"},
        transport=httpx.MockTransport(handler),
    )
    return client


# ---------------------------------------------------------------------------
# Error mapping tests
# ---------------------------------------------------------------------------


def test_401_raises_auth_error():
    client = _client_with_status(401, {"detail": "Invalid API key"})
    with pytest.raises(AuthError) as exc_info:
        client.get_video(1)
    assert exc_info.value.status_code == 401
    assert "Invalid API key" in str(exc_info.value)


def test_403_raises_auth_error():
    client = _client_with_status(403, {"detail": "Account suspended"})
    with pytest.raises(AuthError) as exc_info:
        client.get_video(1)
    assert exc_info.value.status_code == 403


def test_404_raises_not_found_error():
    client = _client_with_status(404, {"detail": "Video not found"})
    with pytest.raises(NotFoundError) as exc_info:
        client.get_video(99)
    assert exc_info.value.status_code == 404
    assert "Video not found" in str(exc_info.value)


def test_429_raises_quota_error():
    client = _client_with_status(429, {"detail": "Monthly video limit reached"})
    with pytest.raises(QuotaError) as exc_info:
        client.get_video(1)
    assert exc_info.value.status_code == 429


def test_500_raises_server_error():
    client = _client_with_status(500, {"detail": "Internal server error"})
    with pytest.raises(ServerError) as exc_info:
        client.get_video(1)
    assert exc_info.value.status_code == 500


def test_503_raises_server_error():
    client = _client_with_status(503, {"detail": "Service unavailable"})
    with pytest.raises(ServerError) as exc_info:
        client.get_video(1)
    assert exc_info.value.status_code == 503


def test_generic_4xx_raises_motionstate_error():
    client = _client_with_status(422, {"detail": "Validation error"})
    with pytest.raises(MotionStateError) as exc_info:
        client.get_video(1)
    assert exc_info.value.status_code == 422


def test_error_with_non_json_body():
    """Errors with plain-text bodies should not crash the SDK."""
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=500,
            headers={"content-type": "text/plain"},
            content=b"Internal Server Error",
        )

    client = MotionStateClient(base_url="http://test", api_key="key")
    client._http = httpx.Client(
        base_url="http://test",
        headers={"X-API-Key": "key"},
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ServerError) as exc_info:
        client.get_video(1)
    assert "Internal Server Error" in str(exc_info.value)


def test_204_returns_none():
    """HTTP 204 No Content should return None without raising."""
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=204, content=b"")

    client = MotionStateClient(base_url="http://test", api_key="key")
    client._http = httpx.Client(
        base_url="http://test",
        headers={"X-API-Key": "key"},
        transport=httpx.MockTransport(handler),
    )
    result = client._request("DELETE", "/some/resource")
    assert result is None


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


def test_auth_error_is_motionstate_error():
    assert issubclass(AuthError, MotionStateError)


def test_quota_error_is_motionstate_error():
    assert issubclass(QuotaError, MotionStateError)


def test_not_found_error_is_motionstate_error():
    assert issubclass(NotFoundError, MotionStateError)


def test_server_error_is_motionstate_error():
    assert issubclass(ServerError, MotionStateError)

"""Tests for database URL normalization (Railway / asyncpg compatibility)."""

from unittest.mock import MagicMock, patch

import pytest

from libs.db import normalize_database_url


# ---------------------------------------------------------------------------
# normalize_database_url unit tests
# ---------------------------------------------------------------------------


def test_plain_postgresql_url_is_normalized():
    """Railway-style postgresql:// URL is rewritten to postgresql+asyncpg://."""
    raw = "postgresql://user:pass@host:5432/db"
    assert normalize_database_url(raw) == "postgresql+asyncpg://user:pass@host:5432/db"


def test_asyncpg_url_is_unchanged():
    """Already-correct postgresql+asyncpg:// URL is returned as-is."""
    url = "postgresql+asyncpg://user:pass@host:5432/db"
    assert normalize_database_url(url) == url


def test_non_postgres_url_is_unchanged():
    """Non-postgres URLs are returned unchanged."""
    for url in [
        "sqlite:///./test.db",
        "mysql+aiomysql://user:pass@localhost/db",
        "redis://localhost:6379/0",
        "",
    ]:
        assert normalize_database_url(url) == url


def test_normalize_preserves_query_string():
    """Query parameters in the URL survive normalization."""
    raw = "postgresql://user:pass@host:5432/db?sslmode=require"
    result = normalize_database_url(raw)
    assert result == "postgresql+asyncpg://user:pass@host:5432/db?sslmode=require"


# ---------------------------------------------------------------------------
# Engine creation uses normalized URL
# ---------------------------------------------------------------------------


def test_engine_creation_uses_normalized_url():
    """create_async_engine receives the normalized URL, not the raw env string."""
    import importlib
    import libs.db

    raw_url = "postgresql://user:pass@host:5432/db"
    expected_url = "postgresql+asyncpg://user:pass@host:5432/db"

    mock_settings = MagicMock()
    mock_settings.database_url = raw_url
    mock_settings.debug = False

    captured = {}

    def fake_engine(url, **kwargs):
        captured["url"] = url
        return MagicMock()

    # Patch at the source so the re-imported name inside the module gets the mock.
    with patch("sqlalchemy.ext.asyncio.create_async_engine", side_effect=fake_engine), \
         patch("libs.config.settings", mock_settings):
        importlib.reload(libs.db)

    assert "url" in captured, "create_async_engine was not called during module load"
    assert captured["url"] == expected_url, (
        f"Expected normalized URL {expected_url!r}, got {captured['url']!r}"
    )

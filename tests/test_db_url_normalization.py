"""Tests for the DATABASE_URL normalization helper in libs/db.py."""

from libs.db import normalize_database_url


# ---------------------------------------------------------------------------
# normalize_database_url
# ---------------------------------------------------------------------------


def test_plain_postgresql_url_is_rewritten():
    """Plain Railway-style postgresql:// URL is rewritten to postgresql+asyncpg://."""
    url = "postgresql://user:pass@host:5432/db"
    assert normalize_database_url(url) == "postgresql+asyncpg://user:pass@host:5432/db"


def test_asyncpg_url_is_unchanged():
    """Already-correct postgresql+asyncpg:// URL is left as-is."""
    url = "postgresql+asyncpg://user:pass@host:5432/db"
    assert normalize_database_url(url) == url


def test_non_postgres_url_is_unchanged():
    """Non-Postgres URLs (e.g. sqlite) are returned unchanged."""
    url = "sqlite+aiosqlite:///./test.db"
    assert normalize_database_url(url) == url


def test_mysql_url_is_unchanged():
    """MySQL URL is returned unchanged."""
    url = "mysql+aiomysql://user:pass@host/db"
    assert normalize_database_url(url) == url


def test_empty_string_is_unchanged():
    """Empty string is returned unchanged without raising."""
    assert normalize_database_url("") == ""


def test_plain_postgresql_preserves_full_credentials_and_path():
    """All parts of the URL (user, password, host, port, db) survive the rewrite."""
    url = "postgresql://alice:s3cr3t@db.example.com:5433/mydb?sslmode=require"
    result = normalize_database_url(url)
    assert result == "postgresql+asyncpg://alice:s3cr3t@db.example.com:5433/mydb?sslmode=require"

"""Tests for the bootstrap script configuration checks.

These tests exercise the pure-Python helper functions in scripts/bootstrap.py
that do not require a database or Redis connection, so they run without
any external services.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

# Make scripts/ importable when running from the repo root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.bootstrap import (
    check_admin_token,
    check_insecure_defaults,
    check_required_vars,
    create_local_dirs,
    load_dotenv,
)

# ---------------------------------------------------------------------------
# check_required_vars
# ---------------------------------------------------------------------------


def test_check_required_vars_missing_both():
    """Returns both DATABASE_URL and REDIS_URL when neither is set."""
    env = {}
    with patch.dict(os.environ, env, clear=True):
        missing = check_required_vars("local")
    assert "DATABASE_URL" in missing
    assert "REDIS_URL" in missing


def test_check_required_vars_missing_one():
    """Returns only the missing variable when one is set."""
    env = {"DATABASE_URL": "postgresql+asyncpg://user:pw@host/db"}
    with patch.dict(os.environ, env, clear=True):
        missing = check_required_vars("local")
    assert "DATABASE_URL" not in missing
    assert "REDIS_URL" in missing


def test_check_required_vars_all_present():
    """Returns an empty list when all required vars are set."""
    env = {
        "DATABASE_URL": "postgresql+asyncpg://user:pw@host/db",
        "REDIS_URL": "redis://localhost:6379/0",
    }
    with patch.dict(os.environ, env, clear=True):
        missing = check_required_vars("local")
    assert missing == []


# ---------------------------------------------------------------------------
# check_insecure_defaults
# ---------------------------------------------------------------------------


def test_check_insecure_defaults_detects_default_secret():
    """Detects when API_KEY_HMAC_SECRET holds the known insecure default."""
    env = {"API_KEY_HMAC_SECRET": "change-me-in-production"}
    with patch.dict(os.environ, env, clear=True):
        bad = check_insecure_defaults("local")
    assert "API_KEY_HMAC_SECRET" in bad


def test_check_insecure_defaults_passes_with_custom_secret():
    """Returns empty list when API_KEY_HMAC_SECRET is set to a custom value."""
    env = {"API_KEY_HMAC_SECRET": "a" * 64}
    with patch.dict(os.environ, env, clear=True):
        bad = check_insecure_defaults("local")
    assert "API_KEY_HMAC_SECRET" not in bad


def test_check_insecure_defaults_uses_default_when_var_absent():
    """When API_KEY_HMAC_SECRET is not set it compares against the hardcoded default."""
    with patch.dict(os.environ, {}, clear=True):
        bad = check_insecure_defaults("local")
    # The var is absent → compared against default ("change-me-in-production") → flagged.
    assert "API_KEY_HMAC_SECRET" in bad


# ---------------------------------------------------------------------------
# check_admin_token
# ---------------------------------------------------------------------------


def test_check_admin_token_absent():
    """Returns False when ADMIN_TOKEN is not set."""
    with patch.dict(os.environ, {}, clear=True):
        assert check_admin_token() is False


def test_check_admin_token_empty_string():
    """Returns False when ADMIN_TOKEN is an empty string."""
    with patch.dict(os.environ, {"ADMIN_TOKEN": ""}, clear=True):
        assert check_admin_token() is False


def test_check_admin_token_whitespace_only():
    """Returns False when ADMIN_TOKEN contains only whitespace."""
    with patch.dict(os.environ, {"ADMIN_TOKEN": "   "}, clear=True):
        assert check_admin_token() is False


def test_check_admin_token_set():
    """Returns True when ADMIN_TOKEN holds a non-empty value."""
    with patch.dict(os.environ, {"ADMIN_TOKEN": "secret123"}, clear=True):
        assert check_admin_token() is True


# ---------------------------------------------------------------------------
# create_local_dirs
# ---------------------------------------------------------------------------


def test_create_local_dirs_creates_directories(tmp_path):
    """create_local_dirs() creates the three data directories."""
    uploads = tmp_path / "uploads"
    normalized = tmp_path / "normalized"
    artifacts = tmp_path / "artifacts"

    env = {
        "UPLOAD_DIR": str(uploads),
        "NORMALIZED_DIR": str(normalized),
        "ARTIFACTS_DIR": str(artifacts),
    }
    with patch.dict(os.environ, env, clear=True):
        create_local_dirs()

    assert uploads.is_dir()
    assert normalized.is_dir()
    assert artifacts.is_dir()


def test_create_local_dirs_idempotent(tmp_path):
    """create_local_dirs() does not raise when directories already exist."""
    uploads = tmp_path / "uploads"
    uploads.mkdir()

    env = {
        "UPLOAD_DIR": str(uploads),
        "NORMALIZED_DIR": str(tmp_path / "normalized"),
        "ARTIFACTS_DIR": str(tmp_path / "artifacts"),
    }
    with patch.dict(os.environ, env, clear=True):
        create_local_dirs()  # should not raise

    assert uploads.is_dir()


def test_create_local_dirs_nested_path(tmp_path):
    """create_local_dirs() creates nested directories (parents=True)."""
    deep = tmp_path / "a" / "b" / "c"
    env = {
        "UPLOAD_DIR": str(tmp_path / "uploads"),
        "NORMALIZED_DIR": str(tmp_path / "normalized"),
        "ARTIFACTS_DIR": str(deep),
    }
    with patch.dict(os.environ, env, clear=True):
        create_local_dirs()

    assert deep.is_dir()


# ---------------------------------------------------------------------------
# load_dotenv
# ---------------------------------------------------------------------------


def test_load_dotenv_sets_values(tmp_path, monkeypatch):
    """load_dotenv() reads key=value pairs from .env into os.environ."""
    env_file = tmp_path / ".env"
    env_file.write_text("MY_TEST_VAR=hello\nANOTHER_VAR=world\n")

    monkeypatch.chdir(tmp_path)
    # Clear relevant vars so load_dotenv can set them.
    monkeypatch.delenv("MY_TEST_VAR", raising=False)
    monkeypatch.delenv("ANOTHER_VAR", raising=False)

    load_dotenv()

    assert os.environ.get("MY_TEST_VAR") == "hello"
    assert os.environ.get("ANOTHER_VAR") == "world"


def test_load_dotenv_does_not_overwrite(tmp_path, monkeypatch):
    """load_dotenv() does not overwrite already-set environment variables."""
    env_file = tmp_path / ".env"
    env_file.write_text("EXISTING_VAR=from_file\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EXISTING_VAR", "already_set")

    load_dotenv()

    assert os.environ["EXISTING_VAR"] == "already_set"


def test_load_dotenv_ignores_comments(tmp_path, monkeypatch):
    """load_dotenv() ignores comment lines and blank lines."""
    env_file = tmp_path / ".env"
    env_file.write_text("# This is a comment\n\nCOMMENT_TEST=value\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("COMMENT_TEST", raising=False)

    load_dotenv()

    assert os.environ.get("COMMENT_TEST") == "value"


def test_load_dotenv_no_file(tmp_path, monkeypatch):
    """load_dotenv() does nothing when .env does not exist."""
    monkeypatch.chdir(tmp_path)
    # Should not raise even without a .env file.
    load_dotenv()


def test_load_dotenv_strips_quotes(tmp_path, monkeypatch):
    """load_dotenv() strips surrounding quotes from values."""
    env_file = tmp_path / ".env"
    env_file.write_text('QUOTED_VAR="my value"\nSINGLE_VAR=\'another\'\n')

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("QUOTED_VAR", raising=False)
    monkeypatch.delenv("SINGLE_VAR", raising=False)

    load_dotenv()

    assert os.environ.get("QUOTED_VAR") == "my value"
    assert os.environ.get("SINGLE_VAR") == "another"

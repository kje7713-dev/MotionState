"""Unit tests for the local filesystem storage backend."""

from __future__ import annotations

import pytest

from libs.storage.local import LocalStorage


@pytest.mark.asyncio
async def test_local_save_and_load(tmp_path):
    """save() writes bytes; load() reads them back correctly."""
    storage = LocalStorage(root=tmp_path)
    data = b"hello local storage"
    returned_path = await storage.save(data, "test/hello.bin")

    loaded = await storage.load(returned_path)
    assert loaded == data


@pytest.mark.asyncio
async def test_local_save_creates_nested_dirs(tmp_path):
    """save() creates intermediate directories automatically."""
    storage = LocalStorage(root=tmp_path)
    await storage.save(b"nested", "a/b/c/file.txt")
    assert (tmp_path / "a" / "b" / "c" / "file.txt").exists()


@pytest.mark.asyncio
async def test_local_save_returns_full_path(tmp_path):
    """save() returns the full filesystem path under the root."""
    storage = LocalStorage(root=tmp_path)
    returned = await storage.save(b"x", "artifacts/1/state.json")
    assert returned == str(tmp_path / "artifacts" / "1" / "state.json")


@pytest.mark.asyncio
async def test_local_exists_true(tmp_path):
    """exists() returns True for a file that was saved."""
    storage = LocalStorage(root=tmp_path)
    path = await storage.save(b"data", "exists.txt")
    assert await storage.exists(path) is True


@pytest.mark.asyncio
async def test_local_exists_false(tmp_path):
    """exists() returns False for a path that has not been saved."""
    storage = LocalStorage(root=tmp_path)
    assert await storage.exists(str(tmp_path / "no_such_file.txt")) is False


def test_local_full_path(tmp_path):
    """full_path() returns a Path object for the given string."""
    from pathlib import Path

    storage = LocalStorage(root=tmp_path)
    p = storage.full_path(str(tmp_path / "foo.txt"))
    assert isinstance(p, Path)
    assert str(p) == str(tmp_path / "foo.txt")


@pytest.mark.asyncio
async def test_local_generate_upload_url_returns_none(tmp_path):
    """generate_upload_url() returns None for local storage (not supported)."""
    storage = LocalStorage(root=tmp_path)
    result = await storage.generate_upload_url("videos/1/source.mp4", expires_in=3600)
    assert result is None


@pytest.mark.asyncio
async def test_local_save_overwrites_existing(tmp_path):
    """save() overwrites an existing file with new content."""
    storage = LocalStorage(root=tmp_path)
    path = await storage.save(b"original", "overwrite.txt")
    await storage.save(b"updated", "overwrite.txt")
    assert await storage.load(path) == b"updated"

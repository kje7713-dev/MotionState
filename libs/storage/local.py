"""Local filesystem storage backend."""

from pathlib import Path

import aiofiles
import aiofiles.os

from libs.storage.base import StorageBackend


class LocalStorage(StorageBackend):
    """Stores files on the local filesystem under a configurable root directory."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    async def save(self, data: bytes, relative_path: str) -> str:
        """Write *data* to *root/relative_path* and return the full path string."""
        dest = self._root / relative_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(dest, "wb") as fh:
            await fh.write(data)
        return str(dest)

    async def load(self, path: str) -> bytes:
        """Read and return bytes from *path*."""
        async with aiofiles.open(path, "rb") as fh:
            return await fh.read()

    async def exists(self, path: str) -> bool:
        """Return True if *path* exists on disk."""
        return await aiofiles.os.path.exists(path)

    def full_path(self, path: str) -> Path:
        """Return a Path object for *path*."""
        return Path(path)

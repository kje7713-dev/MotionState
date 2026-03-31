"""Abstract base class for object storage backends."""

from abc import ABC, abstractmethod
from pathlib import Path


class StorageBackend(ABC):
    """Interface for storing and retrieving files.

    Implementations may write to local disk, S3, Cloudflare R2, etc.
    """

    @abstractmethod
    async def save(self, data: bytes, relative_path: str) -> str:
        """Persist *data* and return the canonical path/key for later retrieval."""
        ...

    @abstractmethod
    async def load(self, path: str) -> bytes:
        """Return the raw bytes at *path*."""
        ...

    @abstractmethod
    async def exists(self, path: str) -> bool:
        """Return True if the object at *path* exists."""
        ...

    @abstractmethod
    def full_path(self, path: str) -> Path:
        """Return a local filesystem Path for *path* (may be a temp copy for remote backends)."""
        ...

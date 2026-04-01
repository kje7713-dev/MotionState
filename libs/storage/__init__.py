"""Storage package: factory function and canonical object-key helpers.

Usage
-----
Get the configured storage backend::

    from libs.storage import get_storage
    storage = get_storage()
    key = await storage.save(data, artifact_key(video_id, "state.json"))

Canonical key helpers
---------------------
All storage key layouts follow these patterns so the same keys work
identically on local disk and in object storage:

    videos/{video_id}/source{ext}          – raw uploaded source video
    videos/{video_id}/normalized.mp4       – normalized video
    artifacts/{video_id}/state.json        – pipeline state summary
    artifacts/{video_id}/detections.json
    artifacts/{video_id}/tracks.json
    artifacts/{video_id}/poses.json
    artifacts/{video_id}/features.json
    artifacts/{video_id}/segments.json
    artifacts/{video_id}/timeline_manifest.json
    artifacts/{video_id}/clips/<filename>  – segment clip files
"""

from __future__ import annotations

from libs.storage.base import StorageBackend

# ---------------------------------------------------------------------------
# Canonical key helpers
# ---------------------------------------------------------------------------


def source_video_key(video_id: int, ext: str = ".mp4") -> str:
    """Return the canonical storage key for a raw uploaded source video."""
    suffix = ext if ext.startswith(".") else f".{ext}"
    return f"videos/{video_id}/source{suffix}"


def normalized_video_key(video_id: int) -> str:
    """Return the canonical storage key for the normalized video."""
    return f"videos/{video_id}/normalized.mp4"


def artifact_key(video_id: int, filename: str) -> str:
    """Return the canonical storage key for an artifact file.

    Args:
        video_id: The numeric video primary key.
        filename: Relative filename within the video artifact directory,
            e.g. ``"state.json"`` or ``"clips/segment_000_low_motion.mp4"``.
    """
    return f"artifacts/{video_id}/{filename}"


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------


def get_storage(cfg=None) -> StorageBackend:
    """Return the configured storage backend singleton.

    The backend is selected by ``cfg.storage_backend`` (or
    ``settings.storage_backend`` when *cfg* is ``None``):

    * ``"local"`` (default) – :class:`~libs.storage.local.LocalStorage` rooted
      at ``cfg.artifacts_dir``.
    * ``"s3"`` – :class:`~libs.storage.s3.S3Storage` using the S3/R2 settings
      from *cfg*.

    Args:
        cfg: An object with storage-related attributes (compatible with
            :class:`~libs.config.Settings`).  When ``None``, the global
            :data:`~libs.config.settings` instance is used.  Callers that
            want to use a custom or mocked configuration should pass it here.

    A new instance is created on every call; callers that need a singleton
    should cache the result themselves.
    """
    if cfg is None:
        from libs.config import settings as _settings

        cfg = _settings

    if cfg.storage_backend == "s3":
        from libs.storage.s3 import S3Storage

        return S3Storage(
            bucket=cfg.s3_bucket,
            region=cfg.s3_region,
            endpoint_url=cfg.s3_endpoint_url,
            access_key_id=cfg.s3_access_key_id,
            secret_access_key=cfg.s3_secret_access_key,
        )

    from pathlib import Path

    from libs.storage.local import LocalStorage

    # The canonical artifact keys include an ``artifacts/`` prefix
    # (e.g. ``artifacts/1/state.json``).  To ensure these keys produce paths
    # directly under ``artifacts_dir``, use the *parent* of ``artifacts_dir``
    # as the local root so that ``root / "artifacts/1/state.json"`` resolves
    # to ``artifacts_dir / "1" / "state.json"``.
    root = Path(cfg.artifacts_dir).parent
    return LocalStorage(root=root)

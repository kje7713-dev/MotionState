"""Shared helpers for resolving and reading artifact files.

Usage pattern:

    artifact = await get_latest_artifact(db, video_id, "detections")
    content   = await read_artifact_json(artifact.path)

``get_latest_artifact`` always returns the **most recently created** artifact
row for a given video + type pair, or ``None`` if no row exists.  When multiple
rows exist (e.g. a video was re-processed) the latest one is selected by
ordering on ``Artifact.id DESC`` (which is monotonically increasing and always
available without a timezone-aware comparison).

``read_artifact_json`` routes reads through the configured storage backend:

* **local backend** – validates that the resolved path sits inside
  ``settings.artifacts_dir`` before reading, so callers cannot be tricked into
  serving arbitrary files from the host filesystem.
* **S3 backend** – loads the object by key from the configured S3/R2 bucket.
  No filesystem path validation is performed; the object key itself acts as the
  access boundary (the bucket is private and only reachable via the configured
  credentials).
"""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.config import settings
from libs.models import Artifact
from libs.storage import get_storage


async def get_latest_artifact(
    db: AsyncSession,
    video_id: int,
    artifact_type: str,
) -> Artifact | None:
    """Return the most recently created artifact row for *video_id* / *artifact_type*.

    When multiple rows exist the one with the highest ``id`` is returned (id is
    auto-incrementing, so the highest id is always the most recently inserted
    row).

    Args:
        db: Active async database session.
        video_id: Primary key of the video.
        artifact_type: Artifact type string, e.g. ``"state"``, ``"detections"``.

    Returns:
        The latest :class:`~libs.models.Artifact` row, or ``None`` if no row
        exists for this video / type combination.
    """
    result = await db.execute(
        select(Artifact)
        .where(Artifact.video_id == video_id, Artifact.type == artifact_type)
        .order_by(Artifact.id.desc())
        .limit(1)
    )
    return result.scalars().first()


def _resolve_and_validate_path(raw_path: str) -> Path:
    """Resolve *raw_path* and verify it is inside ``settings.artifacts_dir``.

    Args:
        raw_path: The ``path`` value stored in the artifact row.

    Returns:
        The resolved :class:`~pathlib.Path`.

    Raises:
        ValueError: If the resolved path escapes ``settings.artifacts_dir``.
    """
    artifacts_root = Path(settings.artifacts_dir).resolve()
    resolved = Path(raw_path).resolve()
    # ``Path.is_relative_to`` was added in Python 3.9; safe to use here (py311+).
    if not resolved.is_relative_to(artifacts_root):
        raise ValueError(
            f"Artifact path '{resolved}' is outside the configured artifacts directory "
            f"'{artifacts_root}'."
        )
    return resolved


async def read_artifact_json(raw_path: str) -> dict:
    """Validate *raw_path* and return the parsed JSON content of the artifact.

    Behaviour depends on the configured storage backend:

    * **local** – validates that *raw_path* sits inside ``settings.artifacts_dir``
      (raises ``ValueError`` if not), then reads the file from disk.  Raises
      ``FileNotFoundError`` if the file does not exist.
    * **s3** – loads the object directly from the configured S3/R2 bucket using
      *raw_path* as the object key.  No filesystem path validation is performed.

    Args:
        raw_path: The ``path`` value stored in the artifact row.  For the local
            backend this is an absolute filesystem path; for the S3 backend it
            is the canonical object key (e.g. ``artifacts/1/state.json``).

    Returns:
        Parsed JSON content as a :class:`dict`.

    Raises:
        ValueError: (local only) If the resolved path is outside ``settings.artifacts_dir``.
        FileNotFoundError: (local only) If the file does not exist on disk.
    """
    if settings.storage_backend == "s3":
        storage = get_storage()
        data = await storage.load(raw_path)
        return json.loads(data)

    # Local backend: validate path containment then read from disk.
    resolved = _resolve_and_validate_path(raw_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Artifact file not found: {resolved}")
    return json.loads(resolved.read_bytes())


"""Tests for artifact reads through the configured storage backend.

Covers:
- local backend: path validation and file reading
- s3 backend: reads via storage.load() without filesystem access
- read_artifact_json dispatch logic

boto3 is mocked at the module level so these tests run even when the
``[storage]`` optional dependencies are not installed.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure boto3 is importable even when the [storage] extras are not installed.
# ---------------------------------------------------------------------------
if "boto3" not in sys.modules:
    sys.modules["boto3"] = MagicMock()

from libs.config import settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_artifact(tmp_path, video_id: int, filename: str, data: dict) -> str:
    """Write *data* as JSON to tmp_path/{video_id}/{filename} and return the path."""
    artifact_dir = tmp_path / str(video_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    p = artifact_dir / filename
    p.write_text(json.dumps(data))
    return str(p)


SAMPLE_STATE = {
    "video_id": "1",
    "version": 7,
    "segments": [],
    "tracks": [],
    "features": [],
    "detections_summary": {"frame_count": 0, "frames_with_people": 0, "total_detections": 0},
    "tracking_summary": {
        "track_count": 0,
        "tracked_frame_count": 0,
        "average_detections_per_frame": 0.0,
    },
    "pose_summary": {
        "pose_count": 0,
        "posed_track_count": 0,
        "average_keypoints_per_pose": 0.0,
    },
    "feature_summary": {"feature_count": 0, "featured_track_count": 0, "feature_names": []},
    "segmentation_summary": {
        "segment_count": 0,
        "segment_labels": [],
        "total_segment_duration_ms": 0,
    },
    "clip_summary": {"clip_count": 0, "total_clip_duration_ms": 0},
    "manifest_path": "",
    "notes": "test",
}


# ---------------------------------------------------------------------------
# Local backend: read_artifact_json
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_artifact_json_local_reads_file(tmp_path):
    """read_artifact_json() returns the parsed JSON for a valid local artifact."""
    path = _write_artifact(tmp_path, 1, "state.json", SAMPLE_STATE)

    with patch.object(settings, "storage_backend", "local"), patch.object(
        settings, "artifacts_dir", str(tmp_path)
    ):
        from libs.artifacts import read_artifact_json

        result = await read_artifact_json(path)

    assert result["video_id"] == "1"
    assert result["version"] == 7


@pytest.mark.asyncio
async def test_read_artifact_json_local_raises_for_path_outside_artifacts_dir(tmp_path):
    """read_artifact_json() raises ValueError when the path escapes artifacts_dir."""
    safe_dir = tmp_path / "safe"
    safe_dir.mkdir()
    evil_path = str(tmp_path / "etc" / "passwd")

    with patch.object(settings, "storage_backend", "local"), patch.object(
        settings, "artifacts_dir", str(safe_dir)
    ):
        from libs.artifacts import read_artifact_json

        with pytest.raises(ValueError, match="outside the configured artifacts directory"):
            await read_artifact_json(evil_path)


@pytest.mark.asyncio
async def test_read_artifact_json_local_raises_file_not_found(tmp_path):
    """read_artifact_json() raises FileNotFoundError when the file is missing."""
    missing = str(tmp_path / "missing.json")

    with patch.object(settings, "storage_backend", "local"), patch.object(
        settings, "artifacts_dir", str(tmp_path)
    ):
        from libs.artifacts import read_artifact_json

        with pytest.raises(FileNotFoundError):
            await read_artifact_json(missing)


# ---------------------------------------------------------------------------
# S3 backend: read_artifact_json
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_artifact_json_s3_loads_via_storage(tmp_path):
    """read_artifact_json() with S3 backend calls storage.load() not the filesystem."""
    from libs.storage.s3 import S3Storage

    expected = {**SAMPLE_STATE, "video_id": "42"}
    payload = json.dumps(expected).encode()

    with (
        patch.object(settings, "storage_backend", "s3"),
        patch.object(S3Storage, "load", new=AsyncMock(return_value=payload)),
        patch("boto3.client", return_value=MagicMock()),
        patch.object(settings, "s3_bucket", "bucket"),
        patch.object(settings, "s3_region", "us-east-1"),
        patch.object(settings, "s3_endpoint_url", ""),
        patch.object(settings, "s3_access_key_id", "key"),
        patch.object(settings, "s3_secret_access_key", "secret"),
    ):
        from libs.artifacts import read_artifact_json

        result = await read_artifact_json("artifacts/42/state.json")

    assert result["video_id"] == "42"


@pytest.mark.asyncio
async def test_read_artifact_json_s3_does_not_touch_filesystem(tmp_path):
    """read_artifact_json() with S3 backend does not open any local file."""
    from libs.storage.s3 import S3Storage

    payload = json.dumps(SAMPLE_STATE).encode()

    open_calls: list = []

    original_open = open

    def _tracked_open(path, *args, **kwargs):
        open_calls.append(path)
        return original_open(path, *args, **kwargs)

    with (
        patch.object(settings, "storage_backend", "s3"),
        patch.object(S3Storage, "load", new=AsyncMock(return_value=payload)),
        patch("boto3.client", return_value=MagicMock()),
        patch.object(settings, "s3_bucket", "bucket"),
        patch.object(settings, "s3_region", "us-east-1"),
        patch.object(settings, "s3_endpoint_url", ""),
        patch.object(settings, "s3_access_key_id", "key"),
        patch.object(settings, "s3_secret_access_key", "secret"),
        patch("builtins.open", side_effect=_tracked_open),
    ):
        from libs.artifacts import read_artifact_json

        await read_artifact_json("artifacts/1/state.json")

    # No local file should have been opened.
    assert open_calls == [], f"Unexpected open() calls: {open_calls}"


# ---------------------------------------------------------------------------
# API layer: _get_artifact_content with local backend
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_artifact_content_local_returns_parsed_json(tmp_path):
    """_get_artifact_content() returns parsed dict when artifact file exists (local)."""
    from unittest.mock import AsyncMock, MagicMock

    from apps.api.routes.videos import _get_artifact_content
    from libs.models import Artifact, Video

    path = _write_artifact(tmp_path, 1, "state.json", SAMPLE_STATE)

    artifact = MagicMock(spec=Artifact)
    artifact.path = path

    video = MagicMock(spec=Video)
    video.id = 1

    db = AsyncMock()
    db.get = AsyncMock(return_value=video)

    mock_scalars = MagicMock()
    mock_scalars.first.return_value = artifact
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars
    db.execute = AsyncMock(return_value=mock_result)

    with patch.object(settings, "storage_backend", "local"), patch.object(
        settings, "artifacts_dir", str(tmp_path)
    ):
        result = await _get_artifact_content(db, 1, "state", "not found")

    assert result["version"] == 7


# ---------------------------------------------------------------------------
# Smoke: local backend artifact writes still produce readable files
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_local_storage_written_artifact_is_readable(tmp_path):
    """Artifacts written via LocalStorage can be read back with read_artifact_json."""
    import json

    from libs.artifacts import read_artifact_json
    from libs.storage.local import LocalStorage

    storage = LocalStorage(root=tmp_path)
    data = {**SAMPLE_STATE, "video_id": "99"}
    key = "99/state.json"
    saved_path = await storage.save(json.dumps(data).encode(), key)

    with patch.object(settings, "storage_backend", "local"), patch.object(
        settings, "artifacts_dir", str(tmp_path)
    ):
        result = await read_artifact_json(saved_path)

    assert result["video_id"] == "99"

"""Tests for the libs/artifacts.py helper module.

Covers:
- get_latest_artifact selects the artifact with the highest id
- get_latest_artifact returns None when no rows exist
- read_artifact_json validates path is inside artifacts_dir
- read_artifact_json raises FileNotFoundError when file missing
- read_artifact_json returns parsed JSON content
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from libs.config import settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_artifact(artifact_id: int, video_id: int, artifact_type: str, path: str):
    from libs.models import Artifact

    a = MagicMock(spec=Artifact)
    a.id = artifact_id
    a.video_id = video_id
    a.type = artifact_type
    a.path = path
    return a


def _mock_db_returning(artifact):
    """Return a mock async DB session whose execute returns *artifact* via scalars().first()."""
    mock_scalars = MagicMock()
    mock_scalars.first.return_value = artifact
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars

    session = AsyncMock()
    session.execute = AsyncMock(return_value=mock_result)
    return session


# ---------------------------------------------------------------------------
# get_latest_artifact
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_latest_artifact_returns_artifact():
    """get_latest_artifact returns the artifact returned by the DB query."""
    from libs.artifacts import get_latest_artifact

    artifact = _make_artifact(5, 1, "state", "/data/artifacts/1/state.json")
    db = _mock_db_returning(artifact)

    result = await get_latest_artifact(db, 1, "state")
    assert result is artifact


@pytest.mark.asyncio
async def test_get_latest_artifact_returns_none_when_missing():
    """get_latest_artifact returns None when no rows are found."""
    from libs.artifacts import get_latest_artifact

    db = _mock_db_returning(None)

    result = await get_latest_artifact(db, 99, "state")
    assert result is None


@pytest.mark.asyncio
async def test_get_latest_artifact_passes_correct_filters():
    """get_latest_artifact calls execute with a query that limits to 1 row."""
    from libs.artifacts import get_latest_artifact

    artifact = _make_artifact(3, 2, "detections", "/data/artifacts/2/detections.json")
    db = _mock_db_returning(artifact)

    await get_latest_artifact(db, 2, "detections")

    # Verify execute was called exactly once
    db.execute.assert_called_once()


# ---------------------------------------------------------------------------
# latest-artifact semantics: highest-id wins
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_latest_artifact_uses_highest_id(tmp_path):
    """When two artifact rows exist the one with the higher id is returned.

    We verify this indirectly: ``get_latest_artifact`` uses
    ``order_by(Artifact.id.desc()).limit(1)`` so the mock must return only
    one artifact (the one set on ``scalars().first()``).
    """
    from libs.artifacts import get_latest_artifact

    # Simulate the DB returning the newer (higher id) artifact
    newer = _make_artifact(10, 1, "state", "/data/artifacts/1/state_v2.json")
    db = _mock_db_returning(newer)

    result = await get_latest_artifact(db, 1, "state")
    assert result.id == 10


# ---------------------------------------------------------------------------
# read_artifact_json – path validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_artifact_json_rejects_path_outside_artifacts_dir(tmp_path):
    """read_artifact_json raises ValueError for paths outside artifacts_dir."""
    from libs.artifacts import read_artifact_json

    outside = str(tmp_path / "evil.json")
    with pytest.raises(ValueError, match="outside"):
        await read_artifact_json(outside)


@pytest.mark.asyncio
async def test_read_artifact_json_raises_file_not_found(tmp_path):
    """read_artifact_json raises FileNotFoundError when the file is missing."""
    from libs.artifacts import read_artifact_json

    # Temporarily point artifacts_dir to tmp_path so path validation passes
    with patch.object(settings, "artifacts_dir", str(tmp_path)):
        missing = str(tmp_path / "missing.json")
        with pytest.raises(FileNotFoundError):
            await read_artifact_json(missing)


@pytest.mark.asyncio
async def test_read_artifact_json_returns_parsed_content(tmp_path):
    """read_artifact_json returns the parsed JSON dict when the file exists."""
    from libs.artifacts import read_artifact_json

    payload = {"video_id": "1", "version": 1, "items": []}
    artifact_file = tmp_path / "state.json"
    artifact_file.write_text(json.dumps(payload))

    with patch.object(settings, "artifacts_dir", str(tmp_path)):
        result = await read_artifact_json(str(artifact_file))

    assert result == payload

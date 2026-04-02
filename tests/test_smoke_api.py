"""API smoke tests for MotionState read endpoints.

These tests prove that the API read path works against actually generated
pipeline artifacts, not only mocked payloads.

Endpoints exercised:
  GET /videos/{video_id}/state
  GET /videos/{video_id}/timeline

What this proves:
  - The state endpoint returns 200 with the correct structure when backed by
    a real pipeline-generated state.json file.
  - The timeline endpoint returns 200 with the correct structure when backed
    by a real timeline_manifest.json file.

What this does NOT prove:
  - Numeric correctness of CV output
  - Production backend behaviour
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from libs.config import settings

# ---------------------------------------------------------------------------
# App fixture (matches pattern used in other API tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def app():
    """Return the FastAPI app with DB creation disabled."""
    with patch("apps.api.main.engine") as mock_engine:
        mock_conn = AsyncMock()
        mock_conn.run_sync = AsyncMock()
        mock_engine.begin.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_engine.begin.return_value.__aexit__ = AsyncMock(return_value=False)

        from apps.api.main import app as fastapi_app
        from libs.auth import get_current_project
        from tests.conftest import make_auth_override

        fastapi_app.dependency_overrides[get_current_project] = make_auth_override()
        yield fastapi_app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_artifact_row(artifact_id: int, video_id: int, artifact_type: str, path: str):
    from libs.models import Artifact

    a = MagicMock(spec=Artifact)
    a.id = artifact_id
    a.video_id = video_id
    a.type = artifact_type
    a.path = path
    a.metadata_json = {"version": 7 if artifact_type == "state" else 1}
    return a


def _db_override_for_smoke(video, artifact):
    """Return a get_db override that yields a session returning fake video and artifact."""
    mock_scalars = MagicMock()
    mock_scalars.first.return_value = artifact
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars

    async def _get_db():
        session = AsyncMock()
        session.get = AsyncMock(return_value=video)
        session.execute = AsyncMock(return_value=mock_result)
        yield session

    return _get_db


def _make_fake_video(video_id: int = 1):
    from libs.models import Video

    v = MagicMock(spec=Video)
    v.id = video_id
    v.project_id = 1
    return v


def _build_real_state(tmp_path, video_id: int = 1) -> tuple[dict, str]:
    """Run the pipeline with stub backends and write state.json to tmp_path."""
    from libs.pipeline.run_pipeline import run_pipeline

    state, _det, _trk, _pose, _feat, _seg = run_pipeline(video_id, frames=[])
    state["clip_summary"] = {"clip_count": 0, "total_clip_duration_ms": 0}
    state["manifest_path"] = str(tmp_path / str(video_id) / "timeline_manifest.json")

    artifact_dir = tmp_path / str(video_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    state_path = artifact_dir / "state.json"
    state_path.write_text(json.dumps(state, indent=2))
    return state, str(state_path)


def _build_real_manifest(tmp_path, video_id: int = 1) -> tuple[dict, str]:
    """Build a timeline_manifest.json referencing files under tmp_path."""
    base = str(tmp_path / str(video_id))
    manifest = {
        "video_id": str(video_id),
        "version": 1,
        "duration_seconds": 3.0,
        "artifacts": {
            "state": f"{base}/state.json",
            "detections": f"{base}/detections.json",
            "tracks": f"{base}/tracks.json",
            "poses": f"{base}/poses.json",
            "features": f"{base}/features.json",
            "segments": f"{base}/segments.json",
        },
        "timeline": [],
    }
    artifact_dir = tmp_path / str(video_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact_dir / "timeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest, str(manifest_path)


# ---------------------------------------------------------------------------
# Smoke: GET /videos/{video_id}/state
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_smoke_api_state_endpoint(app, tmp_path):
    """API smoke: GET /state returns 200 with correct structure for a real artifact."""
    from libs.db import get_db

    video_id = 1
    _, state_path = _build_real_state(tmp_path, video_id)

    fake_video = _make_fake_video(video_id)
    artifact = _make_artifact_row(1, video_id, "state", state_path)
    app.dependency_overrides[get_db] = _db_override_for_smoke(video=fake_video, artifact=artifact)

    with patch.object(settings, "artifacts_dir", str(tmp_path)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(f"/videos/{video_id}/state")

    app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["video_id"] == str(video_id)
    assert "version" in data
    assert "detections_summary" in data
    assert "tracking_summary" in data
    assert "pose_summary" in data
    assert "feature_summary" in data
    assert "segmentation_summary" in data
    assert "clip_summary" in data


# ---------------------------------------------------------------------------
# Smoke: GET /videos/{video_id}/timeline
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_smoke_api_timeline_endpoint(app, tmp_path):
    """API smoke: GET /timeline returns 200 with correct structure for a real manifest."""
    from libs.db import get_db

    video_id = 1
    _, manifest_path = _build_real_manifest(tmp_path, video_id)

    fake_video = _make_fake_video(video_id)
    artifact = _make_artifact_row(1, video_id, "timeline_manifest", manifest_path)
    app.dependency_overrides[get_db] = _db_override_for_smoke(video=fake_video, artifact=artifact)

    with patch.object(settings, "artifacts_dir", str(tmp_path)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(f"/videos/{video_id}/timeline")

    app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["video_id"] == str(video_id)
    assert "version" in data
    assert "duration_seconds" in data
    assert "artifacts" in data
    assert "timeline" in data

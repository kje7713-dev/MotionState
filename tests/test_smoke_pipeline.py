"""End-to-end smoke test for the full MotionState processing pipeline.

Proves that pipeline wiring and artifact generation work as a system:

    fixture video → normalize → frame extract → CV pipeline (stub backends)
                 → clips → timeline manifest → all expected artifact files written

What this proves:
  - normalized video is written
  - all JSON artifact files are produced (state, detections, tracks, poses,
    features, segments, timeline_manifest)
  - clips directory is created
  - state.json contains the expected top-level summary keys
  - timeline_manifest.json contains the expected top-level keys
  - artifact rows are created for all artifact types

What this does NOT prove:
  - CV model quality or detection accuracy
  - Domain-specific motion correctness
  - Production backend behaviour (YOLOv8, MediaPipe, ByteTrack, etc.)
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from libs.models import Artifact

# ---------------------------------------------------------------------------
# Fixture video location
# ---------------------------------------------------------------------------

FIXTURE_VIDEO = Path(__file__).parent / "fixtures" / "fixture.mp4"


def _ffmpeg_available() -> bool:
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ---------------------------------------------------------------------------
# Mock DB helper
# ---------------------------------------------------------------------------


def _make_mock_db(source_path: str, *, video_id: int = 1, job_id: int = 1):
    """Return a mock async DB session wired to realistic fake video/job objects."""
    fake_video = MagicMock()
    fake_video.id = video_id
    fake_video.source_path = source_path
    fake_video.status = None
    fake_video.normalized_path = None

    fake_job = MagicMock()
    fake_job.id = job_id
    fake_job.status = None
    fake_job.error = None

    added_objects: list = []

    mock_db = AsyncMock()
    mock_db.get = AsyncMock(
        side_effect=lambda model, pk: fake_job if model.__name__ == "Job" else fake_video
    )
    mock_db.add = MagicMock(side_effect=lambda obj: added_objects.append(obj))
    mock_db.commit = AsyncMock()
    mock_db.__aenter__ = AsyncMock(return_value=mock_db)
    mock_db.__aexit__ = AsyncMock(return_value=False)

    return mock_db, fake_video, fake_job, added_objects


# ---------------------------------------------------------------------------
# Smoke: fixture video sanity check
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_smoke_fixture_video_exists():
    """The committed fixture video must be present and non-empty."""
    assert FIXTURE_VIDEO.exists(), f"Fixture video not found: {FIXTURE_VIDEO}"
    assert FIXTURE_VIDEO.stat().st_size > 0, "Fixture video is empty"


# ---------------------------------------------------------------------------
# Smoke: full pipeline end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_smoke_full_pipeline(tmp_path):
    """End-to-end smoke: fixture video → all expected artifact files written."""
    if not _ffmpeg_available():
        pytest.skip("ffmpeg not available")

    video_id = 1

    # Copy fixture to a writable upload location.
    upload_path = tmp_path / "upload" / "fixture.mp4"
    upload_path.parent.mkdir(parents=True)
    shutil.copy(FIXTURE_VIDEO, upload_path)

    mock_db, _, _, added_objects = _make_mock_db(str(upload_path), video_id=video_id)
    artifact_dir = tmp_path / "artifacts" / str(video_id)

    with (
        patch("apps.worker.jobs.process_video.AsyncSessionLocal", return_value=mock_db),
        patch("apps.worker.jobs.process_video.settings") as mock_settings,
    ):
        mock_settings.normalized_dir = str(tmp_path / "normalized")
        mock_settings.artifacts_dir = str(tmp_path / "artifacts")
        mock_settings.frame_sample_fps = 1.0  # low FPS keeps the test fast
        mock_settings.detector_backend = "stub"
        mock_settings.detector_model = "yolov8n.pt"
        mock_settings.tracker_backend = "stub"
        mock_settings.tracker_iou_threshold = 0.3
        mock_settings.tracker_max_age = 30
        mock_settings.pose_backend = "stub"
        mock_settings.pose_min_confidence = 0.3

        from apps.worker.jobs.process_video import handle_process_video

        await handle_process_video({"job_id": 1, "payload": {"video_id": video_id}})

    # --- Normalized video must be written ---
    norm_path = tmp_path / "normalized" / f"{video_id}_normalized.mp4"
    assert norm_path.exists(), "normalized video missing"

    # --- All JSON artifact files must exist ---
    assert (artifact_dir / "state.json").exists(), "state.json missing"
    assert (artifact_dir / "detections.json").exists(), "detections.json missing"
    assert (artifact_dir / "tracks.json").exists(), "tracks.json missing"
    assert (artifact_dir / "poses.json").exists(), "poses.json missing"
    assert (artifact_dir / "features.json").exists(), "features.json missing"
    assert (artifact_dir / "segments.json").exists(), "segments.json missing"
    assert (artifact_dir / "timeline_manifest.json").exists(), "timeline_manifest.json missing"

    # --- Clips directory must exist ---
    assert (artifact_dir / "clips").exists(), "clips/ directory missing"

    # --- state.json must contain expected top-level summary keys ---
    state = json.loads((artifact_dir / "state.json").read_text())
    for key in (
        "video_id",
        "version",
        "detections_summary",
        "tracking_summary",
        "pose_summary",
        "feature_summary",
        "segmentation_summary",
        "clip_summary",
        "manifest_path",
    ):
        assert key in state, f"state.json missing key: {key}"

    # --- timeline_manifest.json must contain expected top-level keys ---
    manifest = json.loads((artifact_dir / "timeline_manifest.json").read_text())
    for key in ("video_id", "version", "duration_seconds", "artifacts", "timeline"):
        assert key in manifest, f"timeline_manifest.json missing key: {key}"

    # --- Artifact rows must be created for all required types ---
    artifact_types = {o.type for o in added_objects if isinstance(o, Artifact)}
    for expected_type in (
        "state",
        "detections",
        "tracks",
        "poses",
        "features",
        "segments",
        "timeline_manifest",
    ):
        assert expected_type in artifact_types, f"Artifact row missing for type: {expected_type}"

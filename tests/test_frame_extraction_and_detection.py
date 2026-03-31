"""Tests for the frame extraction + person detection pipeline stage.

Covers:
- frame extraction helper returns expected metadata
- pipeline produces non-placeholder output when a mock detector is used
- worker creates both state and detections artifacts
- GET /videos/{id}/artifacts returns artifact records
"""

import json
from datetime import UTC
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from libs.pipeline.contracts import BoundingBox, Detection, Frame
from libs.pipeline.detector import StubDetector
from libs.pipeline.run_pipeline import run_pipeline
from libs.video.frames import FrameMeta, extract_frames

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockDetector(StubDetector):
    """Detector that always returns one fake person detection per frame."""

    def detect(self, frame: Frame) -> list[Detection]:
        return [
            Detection(
                frame_index=frame.index,
                bbox=BoundingBox(x=10.0, y=20.0, width=100.0, height=200.0, confidence=0.93),
                class_id=0,
                class_label="person",
            )
        ]


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


def test_extract_frames_returns_metadata(tmp_path):
    """extract_frames returns FrameMeta objects matching written files."""
    # Create fake JPEG files to simulate ffmpeg output.
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for i in range(3):
        (frames_dir / f"frame_{i:06d}.jpg").write_bytes(b"\xff\xd8\xff\xe0")  # minimal JPEG header

    # Patch subprocess.run so we don't need a real ffmpeg binary.
    with patch("libs.video.frames.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        result = extract_frames("/fake/video.mp4", frames_dir, sample_fps=2.0)

    assert len(result) == 3
    assert all(isinstance(f, FrameMeta) for f in result)
    assert result[0].frame_index == 0
    assert result[0].timestamp_ms == pytest.approx(0.0)
    assert result[1].frame_index == 1
    assert result[1].timestamp_ms == pytest.approx(500.0)  # 1000 / 2
    assert result[2].frame_index == 2
    assert result[2].timestamp_ms == pytest.approx(1000.0)
    # Paths should point into frames_dir
    for meta in result:
        assert Path(meta.path).parent == frames_dir


def test_extract_frames_creates_output_dir(tmp_path):
    """extract_frames creates the output directory if it does not exist."""
    new_dir = tmp_path / "does_not_exist" / "frames"
    assert not new_dir.exists()

    with patch("libs.video.frames.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        extract_frames("/fake/video.mp4", new_dir, sample_fps=1.0)

    assert new_dir.exists()


# ---------------------------------------------------------------------------
# Pipeline – non-placeholder output
# ---------------------------------------------------------------------------


def test_run_pipeline_with_mock_detector_produces_detections(tmp_path):
    """Pipeline output is non-placeholder when a real (mock) detector is used."""
    # Write fake JPEG frame files.
    frame_path = tmp_path / "frame_000000.jpg"
    frame_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

    frames = [FrameMeta(frame_index=0, timestamp_ms=0.0, path=str(frame_path))]

    state, detections = run_pipeline(
        video_id="test-123",
        frames=frames,
        detector=MockDetector(),
        sample_fps=2.0,
    )

    # State artifact checks
    assert state["video_id"] == "test-123"
    assert state["version"] == 2
    assert "placeholder" not in state.get("notes", "")
    summary = state["detections_summary"]
    assert summary["frame_count"] == 1
    assert summary["frames_with_people"] == 1
    assert summary["total_detections"] == 1

    # Detections artifact checks
    assert detections["video_id"] == "test-123"
    assert detections["sample_fps"] == 2.0
    assert len(detections["frames"]) == 1
    frame_entry = detections["frames"][0]
    assert frame_entry["frame_index"] == 0
    assert len(frame_entry["detections"]) == 1
    det = frame_entry["detections"][0]
    assert det["class_label"] == "person"
    assert det["bbox"]["confidence"] == pytest.approx(0.93)


def test_run_pipeline_empty_frames_gives_zero_summary():
    """Pipeline with no frames returns all-zero detections summary."""
    state, detections = run_pipeline(video_id="42", frames=[], sample_fps=2.0)

    assert state["detections_summary"]["frame_count"] == 0
    assert state["detections_summary"]["total_detections"] == 0
    assert detections["frames"] == []


# ---------------------------------------------------------------------------
# Worker – artifact creation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_process_video_creates_both_artifacts(tmp_path):
    """Worker writes state.json and detections.json and persists two Artifact rows."""

    fake_state = {
        "video_id": "1",
        "version": 2,
        "segments": [],
        "tracks": [],
        "features": [],
        "detections_summary": {"frame_count": 1, "frames_with_people": 0, "total_detections": 0},
        "notes": "first real CV stage: frame extraction and person detection",
    }
    fake_detections = {
        "video_id": "1",
        "version": 1,
        "sample_fps": 2.0,
        "frames": [],
    }

    fake_video = MagicMock()
    fake_video.id = 1
    fake_video.source_path = str(tmp_path / "input.mp4")
    fake_video.status = None
    fake_video.normalized_path = None

    (tmp_path / "input.mp4").write_bytes(b"fake")

    fake_job = MagicMock()
    fake_job.id = 1
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

    artifacts_dir = tmp_path / "artifacts"

    with (
        patch("apps.worker.jobs.process_video.AsyncSessionLocal", return_value=mock_db),
        patch("apps.worker.jobs.process_video.normalize_video"),
        patch("apps.worker.jobs.process_video.probe_video", return_value={
            "duration_seconds": 5.0, "fps": 30.0, "width": 1280, "height": 720,
        }),
        patch("apps.worker.jobs.process_video.extract_frames", return_value=[]),
        patch(
            "apps.worker.jobs.process_video.run_pipeline",
            return_value=(fake_state, fake_detections),
        ),
        patch("apps.worker.jobs.process_video.settings") as mock_settings,
    ):
        mock_settings.normalized_dir = str(tmp_path / "normalized")
        mock_settings.artifacts_dir = str(artifacts_dir)
        mock_settings.frame_sample_fps = 2.0
        mock_settings.detector_backend = "stub"
        mock_settings.detector_model = "yolov8n.pt"

        from apps.worker.jobs.process_video import handle_process_video

        await handle_process_video({"job_id": 1, "payload": {"video_id": 1}})

    # Two Artifact rows should have been added.
    from libs.models import Artifact

    artifact_rows = [o for o in added_objects if isinstance(o, Artifact)]
    assert len(artifact_rows) == 2
    types = {a.type for a in artifact_rows}
    assert types == {"state", "detections"}


# ---------------------------------------------------------------------------
# API – artifacts endpoint
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

        yield fastapi_app


@pytest.mark.asyncio
async def test_list_artifacts_returns_records(app):
    """GET /videos/{id}/artifacts returns artifact list for the video."""
    from datetime import datetime

    from libs.db import get_db
    from libs.models import Artifact, Video

    fake_video = MagicMock(spec=Video)
    fake_video.id = 5

    fake_artifact_1 = MagicMock(spec=Artifact)
    fake_artifact_1.id = 1
    fake_artifact_1.video_id = 5
    fake_artifact_1.type = "state"
    fake_artifact_1.path = "/data/artifacts/5/state.json"
    fake_artifact_1.metadata_json = {"version": 2}
    fake_artifact_1.created_at = datetime(2024, 1, 1, tzinfo=UTC)

    fake_artifact_2 = MagicMock(spec=Artifact)
    fake_artifact_2.id = 2
    fake_artifact_2.video_id = 5
    fake_artifact_2.type = "detections"
    fake_artifact_2.path = "/data/artifacts/5/detections.json"
    fake_artifact_2.metadata_json = {"version": 1, "sample_fps": 2.0}
    fake_artifact_2.created_at = datetime(2024, 1, 1, tzinfo=UTC)

    mock_scalars = MagicMock()
    mock_scalars.all.return_value = [fake_artifact_1, fake_artifact_2]
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars

    async def override_get_db():
        session = AsyncMock()
        session.get = AsyncMock(return_value=fake_video)
        session.execute = AsyncMock(return_value=mock_result)
        yield session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/videos/5/artifacts")

    app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    types = {item["type"] for item in data}
    assert types == {"state", "detections"}


@pytest.mark.asyncio
async def test_list_artifacts_video_not_found(app):
    """GET /videos/{id}/artifacts returns 404 for unknown video."""
    from libs.db import get_db

    async def override_get_db():
        session = AsyncMock()
        session.get = AsyncMock(return_value=None)
        yield session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/videos/9999/artifacts")

    app.dependency_overrides.clear()
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# YoloDetector – optional import
# ---------------------------------------------------------------------------


def test_yolo_detector_raises_import_error_without_ultralytics():
    """YoloDetector raises ImportError when ultralytics is not available."""
    import sys

    # Remove ultralytics from sys.modules if present to simulate missing package.
    with patch.dict(sys.modules, {"ultralytics": None}):
        from libs.pipeline.detector_yolo import YoloDetector

        with pytest.raises(ImportError, match="ultralytics"):
            YoloDetector()


def test_detections_json_shape(tmp_path):
    """detections artifact dict matches the expected schema shape."""
    frame_path = tmp_path / "frame_000000.jpg"
    frame_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

    frames = [FrameMeta(frame_index=0, timestamp_ms=0.0, path=str(frame_path))]
    _, detections = run_pipeline(
        video_id="schema-test",
        frames=frames,
        detector=MockDetector(),
        sample_fps=2.0,
    )

    # Validate schema shape
    assert "video_id" in detections
    assert "version" in detections
    assert "sample_fps" in detections
    assert "frames" in detections
    frame = detections["frames"][0]
    assert "frame_index" in frame
    assert "timestamp_ms" in frame
    assert "path" in frame
    assert "detections" in frame
    det = frame["detections"][0]
    assert "class_label" in det
    assert "bbox" in det
    bbox = det["bbox"]
    for key in ("x", "y", "width", "height", "confidence"):
        assert key in bbox

    # Serialise round-trip to confirm JSON-safe
    json_str = json.dumps(detections)
    parsed = json.loads(json_str)
    assert parsed["video_id"] == "schema-test"

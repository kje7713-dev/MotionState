"""Tests for timeline manifest generation in apps/worker/jobs/process_video.py."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from libs.models import Artifact

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------


def _make_fake_state(video_id: str = "1") -> dict:
    return {
        "video_id": video_id,
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
        "feature_summary": {
            "feature_count": 0,
            "featured_track_count": 0,
            "feature_names": [],
        },
        "segmentation_summary": {
            "segment_count": 0,
            "segment_labels": [],
            "total_segment_duration_ms": 0,
        },
        "clip_summary": {"clip_count": 0, "total_clip_duration_ms": 0},
        "manifest_path": "",
        "notes": (
            "first real CV stages: frame extraction, person detection, tracking, "
            "pose estimation, feature derivation, temporal segmentation, clip generation"
        ),
    }


def _make_fake_segments_with_data(video_id: str = "1") -> dict:
    return {
        "video_id": video_id,
        "version": 1,
        "segment_count": 2,
        "segments": [
            {"start_ms": 0.0, "end_ms": 2500.0, "label": "low_motion", "confidence": 0.88},
            {"start_ms": 2500.0, "end_ms": 5000.0, "label": "active_motion", "confidence": 0.92},
        ],
    }


def _make_fake_segments_empty(video_id: str = "1") -> dict:
    return {"video_id": video_id, "version": 1, "segment_count": 0, "segments": []}


def _make_mock_db(tmp_path, *, video_id: int = 1, job_id: int = 1):
    fake_video = MagicMock()
    fake_video.id = video_id
    fake_video.source_path = str(tmp_path / "input.mp4")
    fake_video.status = None
    fake_video.normalized_path = None
    (tmp_path / "input.mp4").write_bytes(b"fake")

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


def _patch_settings(mock_settings, tmp_path):
    mock_settings.normalized_dir = str(tmp_path / "normalized")
    mock_settings.artifacts_dir = str(tmp_path / "artifacts")
    mock_settings.frame_sample_fps = 2.0
    mock_settings.detector_backend = "stub"
    mock_settings.detector_model = "yolov8n.pt"
    mock_settings.tracker_backend = "stub"
    mock_settings.tracker_iou_threshold = 0.3
    mock_settings.tracker_max_age = 30
    mock_settings.pose_backend = "stub"
    mock_settings.pose_min_confidence = 0.3


def _fake_clips_info() -> list[dict]:
    return [
        {
            "segment_index": 0,
            "label": "low_motion",
            "start_ms": 0.0,
            "end_ms": 2500.0,
            "path": "data/artifacts/1/clips/segment_000_low_motion.mp4",
        },
        {
            "segment_index": 1,
            "label": "active_motion",
            "start_ms": 2500.0,
            "end_ms": 5000.0,
            "path": "data/artifacts/1/clips/segment_001_active_motion.mp4",
        },
    ]


# ---------------------------------------------------------------------------
# timeline_manifest.json is written
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timeline_manifest_json_is_written(tmp_path):
    """handle_process_video writes timeline_manifest.json to the artifacts directory."""
    mock_db, _, _, _ = _make_mock_db(tmp_path)

    with (
        patch("apps.worker.jobs.process_video.AsyncSessionLocal", return_value=mock_db),
        patch("apps.worker.jobs.process_video.normalize_video"),
        patch(
            "apps.worker.jobs.process_video.probe_video",
            return_value={"duration_seconds": 5.0, "fps": 30.0, "width": 1280, "height": 720},
        ),
        patch("apps.worker.jobs.process_video.extract_frames", return_value=[]),
        patch("apps.worker.jobs.process_video.generate_clips", return_value=[]),
        patch(
            "apps.worker.jobs.process_video.run_pipeline",
            return_value=(
                _make_fake_state(),
                {"video_id": "1", "version": 1, "sample_fps": 2.0, "frames": []},
                {"video_id": "1", "version": 1, "track_count": 0, "tracks": []},
                {"video_id": "1", "version": 1, "pose_count": 0, "poses": []},
                {"video_id": "1", "version": 1, "feature_count": 0, "features": []},
                _make_fake_segments_empty(),
            ),
        ),
        patch("apps.worker.jobs.process_video.settings") as mock_settings,
    ):
        _patch_settings(mock_settings, tmp_path)
        from apps.worker.jobs.process_video import handle_process_video

        await handle_process_video({"job_id": 1, "payload": {"video_id": 1}})

    manifest_path = tmp_path / "artifacts" / "1" / "timeline_manifest.json"
    assert manifest_path.exists(), "timeline_manifest.json was not written"


@pytest.mark.asyncio
async def test_timeline_manifest_has_required_top_level_keys(tmp_path):
    """The manifest contains video_id, version, duration_seconds, artifacts, timeline."""
    mock_db, _, _, _ = _make_mock_db(tmp_path)

    with (
        patch("apps.worker.jobs.process_video.AsyncSessionLocal", return_value=mock_db),
        patch("apps.worker.jobs.process_video.normalize_video"),
        patch(
            "apps.worker.jobs.process_video.probe_video",
            return_value={"duration_seconds": 5.0, "fps": 30.0, "width": 1280, "height": 720},
        ),
        patch("apps.worker.jobs.process_video.extract_frames", return_value=[]),
        patch("apps.worker.jobs.process_video.generate_clips", return_value=[]),
        patch(
            "apps.worker.jobs.process_video.run_pipeline",
            return_value=(
                _make_fake_state(),
                {"video_id": "1", "version": 1, "sample_fps": 2.0, "frames": []},
                {"video_id": "1", "version": 1, "track_count": 0, "tracks": []},
                {"video_id": "1", "version": 1, "pose_count": 0, "poses": []},
                {"video_id": "1", "version": 1, "feature_count": 0, "features": []},
                _make_fake_segments_empty(),
            ),
        ),
        patch("apps.worker.jobs.process_video.settings") as mock_settings,
    ):
        _patch_settings(mock_settings, tmp_path)
        from apps.worker.jobs.process_video import handle_process_video

        await handle_process_video({"job_id": 1, "payload": {"video_id": 1}})

    manifest = json.loads(
        (tmp_path / "artifacts" / "1" / "timeline_manifest.json").read_text()
    )
    assert "video_id" in manifest
    assert "version" in manifest
    assert "duration_seconds" in manifest
    assert "artifacts" in manifest
    assert "timeline" in manifest


@pytest.mark.asyncio
async def test_timeline_manifest_artifacts_section_references_all_json_files(tmp_path):
    """The manifest artifacts dict references state, detections, tracks, poses, features,
    segments."""
    mock_db, _, _, _ = _make_mock_db(tmp_path)

    with (
        patch("apps.worker.jobs.process_video.AsyncSessionLocal", return_value=mock_db),
        patch("apps.worker.jobs.process_video.normalize_video"),
        patch(
            "apps.worker.jobs.process_video.probe_video",
            return_value={"duration_seconds": 5.0, "fps": 30.0, "width": 1280, "height": 720},
        ),
        patch("apps.worker.jobs.process_video.extract_frames", return_value=[]),
        patch("apps.worker.jobs.process_video.generate_clips", return_value=[]),
        patch(
            "apps.worker.jobs.process_video.run_pipeline",
            return_value=(
                _make_fake_state(),
                {"video_id": "1", "version": 1, "sample_fps": 2.0, "frames": []},
                {"video_id": "1", "version": 1, "track_count": 0, "tracks": []},
                {"video_id": "1", "version": 1, "pose_count": 0, "poses": []},
                {"video_id": "1", "version": 1, "feature_count": 0, "features": []},
                _make_fake_segments_empty(),
            ),
        ),
        patch("apps.worker.jobs.process_video.settings") as mock_settings,
    ):
        _patch_settings(mock_settings, tmp_path)
        from apps.worker.jobs.process_video import handle_process_video

        await handle_process_video({"job_id": 1, "payload": {"video_id": 1}})

    manifest = json.loads(
        (tmp_path / "artifacts" / "1" / "timeline_manifest.json").read_text()
    )
    assert "state" in manifest["artifacts"]
    assert "detections" in manifest["artifacts"]
    assert "tracks" in manifest["artifacts"]
    assert "poses" in manifest["artifacts"]
    assert "features" in manifest["artifacts"]
    assert "segments" in manifest["artifacts"]


@pytest.mark.asyncio
async def test_timeline_manifest_timeline_has_one_entry_per_clip(tmp_path):
    """The timeline list contains one entry per generated clip."""
    mock_db, _, _, _ = _make_mock_db(tmp_path)
    fake_clips = _fake_clips_info()

    with (
        patch("apps.worker.jobs.process_video.AsyncSessionLocal", return_value=mock_db),
        patch("apps.worker.jobs.process_video.normalize_video"),
        patch(
            "apps.worker.jobs.process_video.probe_video",
            return_value={"duration_seconds": 5.0, "fps": 30.0, "width": 1280, "height": 720},
        ),
        patch("apps.worker.jobs.process_video.extract_frames", return_value=[]),
        patch("apps.worker.jobs.process_video.generate_clips", return_value=fake_clips),
        patch(
            "apps.worker.jobs.process_video.run_pipeline",
            return_value=(
                _make_fake_state(),
                {"video_id": "1", "version": 1, "sample_fps": 2.0, "frames": []},
                {"video_id": "1", "version": 1, "track_count": 0, "tracks": []},
                {"video_id": "1", "version": 1, "pose_count": 0, "poses": []},
                {"video_id": "1", "version": 1, "feature_count": 0, "features": []},
                _make_fake_segments_with_data(),
            ),
        ),
        patch("apps.worker.jobs.process_video.settings") as mock_settings,
    ):
        _patch_settings(mock_settings, tmp_path)
        from apps.worker.jobs.process_video import handle_process_video

        await handle_process_video({"job_id": 1, "payload": {"video_id": 1}})

    manifest = json.loads(
        (tmp_path / "artifacts" / "1" / "timeline_manifest.json").read_text()
    )
    assert len(manifest["timeline"]) == len(fake_clips)


@pytest.mark.asyncio
async def test_timeline_entry_has_required_fields(tmp_path):
    """Each timeline entry contains segment_index, start_ms, end_ms, label, confidence,
    clip_path, and related_artifacts."""
    mock_db, _, _, _ = _make_mock_db(tmp_path)
    fake_clips = _fake_clips_info()

    with (
        patch("apps.worker.jobs.process_video.AsyncSessionLocal", return_value=mock_db),
        patch("apps.worker.jobs.process_video.normalize_video"),
        patch(
            "apps.worker.jobs.process_video.probe_video",
            return_value={"duration_seconds": 5.0, "fps": 30.0, "width": 1280, "height": 720},
        ),
        patch("apps.worker.jobs.process_video.extract_frames", return_value=[]),
        patch("apps.worker.jobs.process_video.generate_clips", return_value=fake_clips),
        patch(
            "apps.worker.jobs.process_video.run_pipeline",
            return_value=(
                _make_fake_state(),
                {"video_id": "1", "version": 1, "sample_fps": 2.0, "frames": []},
                {"video_id": "1", "version": 1, "track_count": 0, "tracks": []},
                {"video_id": "1", "version": 1, "pose_count": 0, "poses": []},
                {"video_id": "1", "version": 1, "feature_count": 0, "features": []},
                _make_fake_segments_with_data(),
            ),
        ),
        patch("apps.worker.jobs.process_video.settings") as mock_settings,
    ):
        _patch_settings(mock_settings, tmp_path)
        from apps.worker.jobs.process_video import handle_process_video

        await handle_process_video({"job_id": 1, "payload": {"video_id": 1}})

    manifest = json.loads(
        (tmp_path / "artifacts" / "1" / "timeline_manifest.json").read_text()
    )
    for entry in manifest["timeline"]:
        assert "segment_index" in entry
        assert "start_ms" in entry
        assert "end_ms" in entry
        assert "label" in entry
        assert "confidence" in entry
        assert "clip_path" in entry
        assert "related_artifacts" in entry


# ---------------------------------------------------------------------------
# State artifact updated with clip_summary and manifest_path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_state_json_includes_clip_summary(tmp_path):
    """state.json written by the worker contains a clip_summary key."""
    mock_db, _, _, _ = _make_mock_db(tmp_path)

    with (
        patch("apps.worker.jobs.process_video.AsyncSessionLocal", return_value=mock_db),
        patch("apps.worker.jobs.process_video.normalize_video"),
        patch(
            "apps.worker.jobs.process_video.probe_video",
            return_value={"duration_seconds": 5.0, "fps": 30.0, "width": 1280, "height": 720},
        ),
        patch("apps.worker.jobs.process_video.extract_frames", return_value=[]),
        patch("apps.worker.jobs.process_video.generate_clips", return_value=[]),
        patch(
            "apps.worker.jobs.process_video.run_pipeline",
            return_value=(
                _make_fake_state(),
                {"video_id": "1", "version": 1, "sample_fps": 2.0, "frames": []},
                {"video_id": "1", "version": 1, "track_count": 0, "tracks": []},
                {"video_id": "1", "version": 1, "pose_count": 0, "poses": []},
                {"video_id": "1", "version": 1, "feature_count": 0, "features": []},
                _make_fake_segments_empty(),
            ),
        ),
        patch("apps.worker.jobs.process_video.settings") as mock_settings,
    ):
        _patch_settings(mock_settings, tmp_path)
        from apps.worker.jobs.process_video import handle_process_video

        await handle_process_video({"job_id": 1, "payload": {"video_id": 1}})

    state = json.loads((tmp_path / "artifacts" / "1" / "state.json").read_text())
    assert "clip_summary" in state
    assert "clip_count" in state["clip_summary"]
    assert "total_clip_duration_ms" in state["clip_summary"]


@pytest.mark.asyncio
async def test_state_json_includes_manifest_path(tmp_path):
    """state.json written by the worker contains a non-empty manifest_path."""
    mock_db, _, _, _ = _make_mock_db(tmp_path)

    with (
        patch("apps.worker.jobs.process_video.AsyncSessionLocal", return_value=mock_db),
        patch("apps.worker.jobs.process_video.normalize_video"),
        patch(
            "apps.worker.jobs.process_video.probe_video",
            return_value={"duration_seconds": 5.0, "fps": 30.0, "width": 1280, "height": 720},
        ),
        patch("apps.worker.jobs.process_video.extract_frames", return_value=[]),
        patch("apps.worker.jobs.process_video.generate_clips", return_value=[]),
        patch(
            "apps.worker.jobs.process_video.run_pipeline",
            return_value=(
                _make_fake_state(),
                {"video_id": "1", "version": 1, "sample_fps": 2.0, "frames": []},
                {"video_id": "1", "version": 1, "track_count": 0, "tracks": []},
                {"video_id": "1", "version": 1, "pose_count": 0, "poses": []},
                {"video_id": "1", "version": 1, "feature_count": 0, "features": []},
                _make_fake_segments_empty(),
            ),
        ),
        patch("apps.worker.jobs.process_video.settings") as mock_settings,
    ):
        _patch_settings(mock_settings, tmp_path)
        from apps.worker.jobs.process_video import handle_process_video

        await handle_process_video({"job_id": 1, "payload": {"video_id": 1}})

    state = json.loads((tmp_path / "artifacts" / "1" / "state.json").read_text())
    assert state["manifest_path"] != ""
    assert "timeline_manifest.json" in state["manifest_path"]


# ---------------------------------------------------------------------------
# Clip artifact rows persisted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clip_artifact_rows_are_persisted(tmp_path):
    """One segment_clip Artifact row is added per generated clip."""
    mock_db, _, _, added_objects = _make_mock_db(tmp_path)
    fake_clips = _fake_clips_info()

    with (
        patch("apps.worker.jobs.process_video.AsyncSessionLocal", return_value=mock_db),
        patch("apps.worker.jobs.process_video.normalize_video"),
        patch(
            "apps.worker.jobs.process_video.probe_video",
            return_value={"duration_seconds": 5.0, "fps": 30.0, "width": 1280, "height": 720},
        ),
        patch("apps.worker.jobs.process_video.extract_frames", return_value=[]),
        patch("apps.worker.jobs.process_video.generate_clips", return_value=fake_clips),
        patch(
            "apps.worker.jobs.process_video.run_pipeline",
            return_value=(
                _make_fake_state(),
                {"video_id": "1", "version": 1, "sample_fps": 2.0, "frames": []},
                {"video_id": "1", "version": 1, "track_count": 0, "tracks": []},
                {"video_id": "1", "version": 1, "pose_count": 0, "poses": []},
                {"video_id": "1", "version": 1, "feature_count": 0, "features": []},
                _make_fake_segments_with_data(),
            ),
        ),
        patch("apps.worker.jobs.process_video.settings") as mock_settings,
    ):
        _patch_settings(mock_settings, tmp_path)
        from apps.worker.jobs.process_video import handle_process_video

        await handle_process_video({"job_id": 1, "payload": {"video_id": 1}})

    clip_rows = [o for o in added_objects if isinstance(o, Artifact) and o.type == "segment_clip"]
    assert len(clip_rows) == len(fake_clips)


@pytest.mark.asyncio
async def test_clip_artifact_metadata_includes_required_fields(tmp_path):
    """Each segment_clip Artifact row metadata contains segment_index, label, start_ms, end_ms."""
    mock_db, _, _, added_objects = _make_mock_db(tmp_path)
    fake_clips = _fake_clips_info()

    with (
        patch("apps.worker.jobs.process_video.AsyncSessionLocal", return_value=mock_db),
        patch("apps.worker.jobs.process_video.normalize_video"),
        patch(
            "apps.worker.jobs.process_video.probe_video",
            return_value={"duration_seconds": 5.0, "fps": 30.0, "width": 1280, "height": 720},
        ),
        patch("apps.worker.jobs.process_video.extract_frames", return_value=[]),
        patch("apps.worker.jobs.process_video.generate_clips", return_value=fake_clips),
        patch(
            "apps.worker.jobs.process_video.run_pipeline",
            return_value=(
                _make_fake_state(),
                {"video_id": "1", "version": 1, "sample_fps": 2.0, "frames": []},
                {"video_id": "1", "version": 1, "track_count": 0, "tracks": []},
                {"video_id": "1", "version": 1, "pose_count": 0, "poses": []},
                {"video_id": "1", "version": 1, "feature_count": 0, "features": []},
                _make_fake_segments_with_data(),
            ),
        ),
        patch("apps.worker.jobs.process_video.settings") as mock_settings,
    ):
        _patch_settings(mock_settings, tmp_path)
        from apps.worker.jobs.process_video import handle_process_video

        await handle_process_video({"job_id": 1, "payload": {"video_id": 1}})

    clip_rows = [o for o in added_objects if isinstance(o, Artifact) and o.type == "segment_clip"]
    for row in clip_rows:
        assert "segment_index" in row.metadata_json
        assert "label" in row.metadata_json
        assert "start_ms" in row.metadata_json
        assert "end_ms" in row.metadata_json


@pytest.mark.asyncio
async def test_timeline_manifest_artifact_row_is_persisted(tmp_path):
    """A timeline_manifest Artifact row is added."""
    mock_db, _, _, added_objects = _make_mock_db(tmp_path)

    with (
        patch("apps.worker.jobs.process_video.AsyncSessionLocal", return_value=mock_db),
        patch("apps.worker.jobs.process_video.normalize_video"),
        patch(
            "apps.worker.jobs.process_video.probe_video",
            return_value={"duration_seconds": 5.0, "fps": 30.0, "width": 1280, "height": 720},
        ),
        patch("apps.worker.jobs.process_video.extract_frames", return_value=[]),
        patch("apps.worker.jobs.process_video.generate_clips", return_value=[]),
        patch(
            "apps.worker.jobs.process_video.run_pipeline",
            return_value=(
                _make_fake_state(),
                {"video_id": "1", "version": 1, "sample_fps": 2.0, "frames": []},
                {"video_id": "1", "version": 1, "track_count": 0, "tracks": []},
                {"video_id": "1", "version": 1, "pose_count": 0, "poses": []},
                {"video_id": "1", "version": 1, "feature_count": 0, "features": []},
                _make_fake_segments_empty(),
            ),
        ),
        patch("apps.worker.jobs.process_video.settings") as mock_settings,
    ):
        _patch_settings(mock_settings, tmp_path)
        from apps.worker.jobs.process_video import handle_process_video

        await handle_process_video({"job_id": 1, "payload": {"video_id": 1}})

    manifest_rows = [
        o for o in added_objects if isinstance(o, Artifact) and o.type == "timeline_manifest"
    ]
    assert len(manifest_rows) == 1

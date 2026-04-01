"""Focused tests for libs/pipeline/run_pipeline.py with a mocked detector."""

import json
from pathlib import Path

import pytest

from libs.pipeline.contracts import BoundingBox, Detection, Frame
from libs.pipeline.detector import StubDetector
from libs.pipeline.run_pipeline import run_pipeline
from libs.video.frames import FrameMeta

# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------


class _OnePerson(StubDetector):
    """Returns exactly one person detection per frame."""

    def detect(self, frame: Frame) -> list[Detection]:
        return [
            Detection(
                frame_index=frame.index,
                bbox=BoundingBox(x=0.0, y=0.0, width=50.0, height=100.0, confidence=0.80),
                class_id=0,
                class_label="person",
            )
        ]


class _TwoPeople(StubDetector):
    """Returns two person detections per frame."""

    def detect(self, frame: Frame) -> list[Detection]:
        return [
            Detection(
                frame_index=frame.index,
                bbox=BoundingBox(x=0.0, y=0.0, width=50.0, height=100.0, confidence=0.80),
                class_id=0,
                class_label="person",
            ),
            Detection(
                frame_index=frame.index,
                bbox=BoundingBox(x=60.0, y=0.0, width=50.0, height=100.0, confidence=0.70),
                class_id=0,
                class_label="person",
            ),
        ]


def _make_frame(tmp_path: Path, idx: int) -> FrameMeta:
    """Write a minimal JPEG to *tmp_path* and return a matching FrameMeta."""
    p = tmp_path / f"frame_{idx:06d}.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16)
    return FrameMeta(frame_index=idx, timestamp_ms=idx * 500.0, path=str(p))


# ---------------------------------------------------------------------------
# detections artifact structure
# ---------------------------------------------------------------------------


class TestDetectionsArtifact:
    def test_video_id_in_artifact(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        _, det, _ = run_pipeline("vid-1", frames=[frame], detector=_OnePerson(), sample_fps=2.0)
        assert det["video_id"] == "vid-1"

    def test_version_field_present(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        _, det, _ = run_pipeline("vid-1", frames=[frame], detector=_OnePerson(), sample_fps=2.0)
        assert "version" in det

    def test_sample_fps_stored(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        _, det, _ = run_pipeline("vid-1", frames=[frame], detector=_OnePerson(), sample_fps=5.0)
        assert det["sample_fps"] == pytest.approx(5.0)

    def test_frames_list_length_matches_input(self, tmp_path):
        frames = [_make_frame(tmp_path, i) for i in range(3)]
        _, det, _ = run_pipeline("v", frames=frames, detector=_OnePerson(), sample_fps=2.0)
        assert len(det["frames"]) == 3

    def test_each_frame_entry_has_required_keys(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        _, det, _ = run_pipeline("v", frames=[frame], detector=_OnePerson(), sample_fps=2.0)
        entry = det["frames"][0]
        for key in ("frame_index", "timestamp_ms", "path", "detections"):
            assert key in entry, f"missing key: {key}"

    def test_detection_entry_has_class_label_and_bbox(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        _, det, _ = run_pipeline("v", frames=[frame], detector=_OnePerson(), sample_fps=2.0)
        detection = det["frames"][0]["detections"][0]
        assert detection["class_label"] == "person"
        for key in ("x", "y", "width", "height", "confidence"):
            assert key in detection["bbox"], f"bbox missing key: {key}"

    def test_artifact_is_json_serialisable(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        _, det, _ = run_pipeline("v", frames=[frame], detector=_OnePerson(), sample_fps=2.0)
        dumped = json.dumps(det)
        parsed = json.loads(dumped)
        assert parsed["video_id"] == "v"


# ---------------------------------------------------------------------------
# detections_summary in state artifact
# ---------------------------------------------------------------------------


class TestStateSummary:
    def test_frame_count_matches_input(self, tmp_path):
        frames = [_make_frame(tmp_path, i) for i in range(4)]
        state, _, _ = run_pipeline("v", frames=frames, detector=_OnePerson(), sample_fps=2.0)
        assert state["detections_summary"]["frame_count"] == 4

    def test_frames_with_people_one_detection_per_frame(self, tmp_path):
        frames = [_make_frame(tmp_path, i) for i in range(3)]
        state, _, _ = run_pipeline("v", frames=frames, detector=_OnePerson(), sample_fps=2.0)
        assert state["detections_summary"]["frames_with_people"] == 3

    def test_total_detections_two_per_frame(self, tmp_path):
        frames = [_make_frame(tmp_path, i) for i in range(2)]
        state, _, _ = run_pipeline("v", frames=frames, detector=_TwoPeople(), sample_fps=2.0)
        assert state["detections_summary"]["total_detections"] == 4

    def test_stub_detector_gives_zero_people(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        state, _, _ = run_pipeline("v", frames=[frame], sample_fps=2.0)
        assert state["detections_summary"]["frames_with_people"] == 0
        assert state["detections_summary"]["total_detections"] == 0

    def test_empty_frames_gives_zero_summary(self):
        state, det, _ = run_pipeline("v", frames=[], sample_fps=2.0)
        assert state["detections_summary"]["frame_count"] == 0
        assert state["detections_summary"]["frames_with_people"] == 0
        assert state["detections_summary"]["total_detections"] == 0
        assert det["frames"] == []

    def test_state_has_required_top_level_keys(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        state, _, _ = run_pipeline("v", frames=[frame], detector=_OnePerson(), sample_fps=2.0)
        for key in ("video_id", "version", "segments", "tracks", "features", "detections_summary"):
            assert key in state, f"state missing key: {key}"

    def test_missing_frame_file_produces_empty_detections(self, tmp_path):
        """Frames whose file is absent are skipped with empty detection list."""
        meta = FrameMeta(frame_index=0, timestamp_ms=0.0, path=str(tmp_path / "gone.jpg"))
        state, det, _ = run_pipeline("v", frames=[meta], detector=_OnePerson(), sample_fps=2.0)
        assert det["frames"][0]["detections"] == []
        assert state["detections_summary"]["total_detections"] == 0

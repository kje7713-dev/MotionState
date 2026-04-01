"""Tests for tracking integration in libs/pipeline/run_pipeline.py."""

import json
from pathlib import Path

import pytest

from libs.pipeline.contracts import BoundingBox, Detection, Frame
from libs.pipeline.detector import StubDetector
from libs.pipeline.run_pipeline import run_pipeline
from libs.pipeline.tracker import StubTracker
from libs.pipeline.tracker_bytetrack import IOUTracker
from libs.video.frames import FrameMeta

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _OnePerson(StubDetector):
    """Returns one person detection per frame at a fixed position."""

    def detect(self, frame: Frame) -> list[Detection]:
        return [
            Detection(
                frame_index=frame.index,
                bbox=BoundingBox(x=10.0, y=20.0, width=50.0, height=100.0, confidence=0.9),
                class_id=0,
                class_label="person",
            )
        ]


class _TwoPeople(StubDetector):
    """Returns two non-overlapping person detections per frame."""

    def detect(self, frame: Frame) -> list[Detection]:
        return [
            Detection(
                frame_index=frame.index,
                bbox=BoundingBox(x=0.0, y=0.0, width=50.0, height=100.0, confidence=0.85),
                class_id=0,
                class_label="person",
            ),
            Detection(
                frame_index=frame.index,
                bbox=BoundingBox(x=300.0, y=0.0, width=50.0, height=100.0, confidence=0.75),
                class_id=0,
                class_label="person",
            ),
        ]


def _make_frame(tmp_path: Path, idx: int) -> FrameMeta:
    p = tmp_path / f"frame_{idx:06d}.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16)
    return FrameMeta(frame_index=idx, timestamp_ms=idx * 500.0, path=str(p))


# ---------------------------------------------------------------------------
# Tracks artifact structure
# ---------------------------------------------------------------------------


class TestTracksArtifact:
    def test_tracks_artifact_has_required_top_level_keys(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        _, _, tracks, _ = run_pipeline(
            "v", frames=[frame], detector=_OnePerson(), tracker=IOUTracker(), sample_fps=2.0
        )
        for key in ("video_id", "version", "track_count", "tracks"):
            assert key in tracks, f"tracks artifact missing key: {key}"

    def test_tracks_artifact_video_id(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        _, _, tracks, _ = run_pipeline(
            "vid-42", frames=[frame], detector=_OnePerson(), tracker=IOUTracker(), sample_fps=2.0
        )
        assert tracks["video_id"] == "vid-42"

    def test_tracks_artifact_version_is_one(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        _, _, tracks, _ = run_pipeline(
            "v", frames=[frame], detector=_OnePerson(), tracker=IOUTracker(), sample_fps=2.0
        )
        assert tracks["version"] == 1

    def test_one_person_produces_one_track(self, tmp_path):
        frames = [_make_frame(tmp_path, i) for i in range(3)]
        _, _, tracks, _ = run_pipeline(
            "v", frames=frames, detector=_OnePerson(), tracker=IOUTracker(), sample_fps=2.0
        )
        assert tracks["track_count"] == 1
        assert len(tracks["tracks"]) == 1

    def test_two_people_produce_two_tracks(self, tmp_path):
        frames = [_make_frame(tmp_path, i) for i in range(2)]
        _, _, tracks, _ = run_pipeline(
            "v", frames=frames, detector=_TwoPeople(), tracker=IOUTracker(), sample_fps=2.0
        )
        assert tracks["track_count"] == 2
        assert len(tracks["tracks"]) == 2

    def test_track_entry_has_required_keys(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        _, _, tracks, _ = run_pipeline(
            "v", frames=[frame], detector=_OnePerson(), tracker=IOUTracker(), sample_fps=2.0
        )
        entry = tracks["tracks"][0]
        assert "track_id" in entry
        assert "detections" in entry

    def test_detection_entry_has_required_keys(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        _, _, tracks, _ = run_pipeline(
            "v", frames=[frame], detector=_OnePerson(), tracker=IOUTracker(), sample_fps=2.0
        )
        det = tracks["tracks"][0]["detections"][0]
        for key in ("frame_index", "timestamp_ms", "class_label", "bbox"):
            assert key in det, f"detection entry missing key: {key}"

    def test_bbox_entry_has_required_keys(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        _, _, tracks, _ = run_pipeline(
            "v", frames=[frame], detector=_OnePerson(), tracker=IOUTracker(), sample_fps=2.0
        )
        bbox = tracks["tracks"][0]["detections"][0]["bbox"]
        for key in ("x", "y", "width", "height", "confidence"):
            assert key in bbox, f"bbox missing key: {key}"

    def test_timestamp_ms_matches_frame_meta(self, tmp_path):
        frame = _make_frame(tmp_path, 3)  # timestamp_ms = 3 * 500 = 1500
        _, _, tracks, _ = run_pipeline(
            "v", frames=[frame], detector=_OnePerson(), tracker=IOUTracker(), sample_fps=2.0
        )
        assert tracks["tracks"][0]["detections"][0]["timestamp_ms"] == pytest.approx(1500.0)

    def test_tracks_artifact_is_json_serialisable(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        _, _, tracks, _ = run_pipeline(
            "v", frames=[frame], detector=_OnePerson(), tracker=IOUTracker(), sample_fps=2.0
        )
        dumped = json.dumps(tracks)
        parsed = json.loads(dumped)
        assert parsed["video_id"] == "v"

    def test_empty_frames_gives_empty_tracks(self):
        _, _, tracks, _ = run_pipeline("v", frames=[], tracker=IOUTracker(), sample_fps=2.0)
        assert tracks["track_count"] == 0
        assert tracks["tracks"] == []

    def test_stub_tracker_gives_empty_tracks(self, tmp_path):
        frames = [_make_frame(tmp_path, i) for i in range(2)]
        _, _, tracks, _ = run_pipeline(
            "v", frames=frames, detector=_OnePerson(), tracker=StubTracker(), sample_fps=2.0
        )
        assert tracks["track_count"] == 0
        assert tracks["tracks"] == []


# ---------------------------------------------------------------------------
# Persistent IDs across frames
# ---------------------------------------------------------------------------


class TestPersistentTrackIDs:
    def test_one_person_three_frames_one_track_three_detections(self, tmp_path):
        frames = [_make_frame(tmp_path, i) for i in range(3)]
        _, _, tracks, _ = run_pipeline(
            "v", frames=frames, detector=_OnePerson(), tracker=IOUTracker(), sample_fps=2.0
        )
        assert tracks["track_count"] == 1
        assert len(tracks["tracks"][0]["detections"]) == 3

    def test_two_people_each_track_has_correct_detection_count(self, tmp_path):
        frames = [_make_frame(tmp_path, i) for i in range(3)]
        _, _, tracks, _ = run_pipeline(
            "v", frames=frames, detector=_TwoPeople(), tracker=IOUTracker(), sample_fps=2.0
        )
        assert tracks["track_count"] == 2
        for track_entry in tracks["tracks"]:
            assert len(track_entry["detections"]) == 3

    def test_track_ids_are_unique(self, tmp_path):
        frames = [_make_frame(tmp_path, i) for i in range(2)]
        _, _, tracks, _ = run_pipeline(
            "v", frames=frames, detector=_TwoPeople(), tracker=IOUTracker(), sample_fps=2.0
        )
        ids = [t["track_id"] for t in tracks["tracks"]]
        assert len(ids) == len(set(ids)), "Track IDs are not unique"


# ---------------------------------------------------------------------------
# State artifact tracking_summary
# ---------------------------------------------------------------------------


class TestStateSummaryTracking:
    def test_state_version_is_four(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        state, _, _, _ = run_pipeline("v", frames=[frame], detector=_OnePerson(), sample_fps=2.0)
        assert state["version"] == 4

    def test_state_has_tracking_summary_key(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        state, _, _, _ = run_pipeline("v", frames=[frame], detector=_OnePerson(), sample_fps=2.0)
        assert "tracking_summary" in state

    def test_tracking_summary_has_required_keys(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        state, _, _, _ = run_pipeline("v", frames=[frame], detector=_OnePerson(), sample_fps=2.0)
        ts = state["tracking_summary"]
        for key in ("track_count", "tracked_frame_count", "average_detections_per_frame"):
            assert key in ts, f"tracking_summary missing key: {key}"

    def test_tracking_summary_with_iou_tracker_one_person(self, tmp_path):
        frames = [_make_frame(tmp_path, i) for i in range(3)]
        state, _, _, _ = run_pipeline(
            "v", frames=frames, detector=_OnePerson(), tracker=IOUTracker(), sample_fps=2.0
        )
        ts = state["tracking_summary"]
        assert ts["track_count"] == 1
        assert ts["tracked_frame_count"] == 3
        assert ts["average_detections_per_frame"] == pytest.approx(1.0)

    def test_tracking_summary_with_stub_tracker_is_zero(self, tmp_path):
        frames = [_make_frame(tmp_path, i) for i in range(3)]
        state, _, _, _ = run_pipeline(
            "v", frames=frames, detector=_OnePerson(), tracker=StubTracker(), sample_fps=2.0
        )
        ts = state["tracking_summary"]
        assert ts["track_count"] == 0
        assert ts["tracked_frame_count"] == 0
        assert ts["average_detections_per_frame"] == pytest.approx(0.0)

    def test_state_notes_mention_tracking(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        state, _, _, _ = run_pipeline("v", frames=[frame], detector=_OnePerson(), sample_fps=2.0)
        assert "tracking" in state["notes"]

    def test_state_tracks_field_matches_tracks_artifact(self, tmp_path):
        frames = [_make_frame(tmp_path, i) for i in range(2)]
        state, _, tracks, _ = run_pipeline(
            "v", frames=frames, detector=_OnePerson(), tracker=IOUTracker(), sample_fps=2.0
        )
        assert state["tracks"] == tracks["tracks"]


# ---------------------------------------------------------------------------
# Pipeline returns non-empty tracks when detections are present
# ---------------------------------------------------------------------------


class TestPipelineReturnsNonEmptyTracks:
    def test_iou_tracker_produces_tracks_with_detections(self, tmp_path):
        frames = [_make_frame(tmp_path, i) for i in range(2)]
        _, _, tracks, _ = run_pipeline(
            "v", frames=frames, detector=_OnePerson(), tracker=IOUTracker(), sample_fps=2.0
        )
        assert tracks["track_count"] > 0
        assert all(len(t["detections"]) > 0 for t in tracks["tracks"])

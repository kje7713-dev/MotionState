"""Tests for motion feature derivation integration in libs/pipeline/run_pipeline.py."""

import json
from pathlib import Path

import pytest

from libs.pipeline.contracts import (
    BoundingBox,
    Detection,
    FeatureDeriver,
    Frame,
    Keypoint,
    MotionFeature,
    PoseEstimate,
    Track,
)
from libs.pipeline.detector import StubDetector
from libs.pipeline.pose import StubPoseEstimator
from libs.pipeline.run_pipeline import run_pipeline
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


class _FullBodyPoseEstimator(StubPoseEstimator):
    """Returns a pose estimate with a full set of body keypoints per tracked person."""

    _KEYPOINTS = [
        ("left_shoulder",  10.0, 20.0),
        ("right_shoulder", 30.0, 20.0),
        ("left_hip",       10.0, 50.0),
        ("right_hip",      30.0, 50.0),
        ("left_elbow",      5.0, 35.0),
        ("right_elbow",    35.0, 35.0),
        ("left_wrist",      0.0, 50.0),
        ("right_wrist",    40.0, 50.0),
        ("left_knee",      10.0, 70.0),
        ("right_knee",     30.0, 70.0),
        ("left_ankle",     10.0, 90.0),
        ("right_ankle",    30.0, 90.0),
    ]

    def estimate(self, frame: Frame, tracks: list[Track]) -> list[PoseEstimate]:
        poses = []
        for track in tracks:
            for det in track.detections:
                if det.frame_index == frame.index:
                    keypoints = [
                        Keypoint(name=n, x=x, y=y, confidence=0.95)
                        for n, x, y in self._KEYPOINTS
                    ]
                    poses.append(
                        PoseEstimate(
                            frame_index=frame.index,
                            track_id=track.track_id,
                            keypoints=keypoints,
                        )
                    )
                    break
        return poses


class _MockFeatureDeriver(FeatureDeriver):
    """Returns one fixed MotionFeature regardless of input."""

    def derive(
        self, tracks: list[Track], poses: list[PoseEstimate]
    ) -> list[MotionFeature]:
        return [
            MotionFeature(track_id=1, name="mock_feature", start_ms=0.0, end_ms=0.0, value=42.0)
        ]


def _make_frame(tmp_path: Path, idx: int) -> FrameMeta:
    p = tmp_path / f"frame_{idx:06d}.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16)
    return FrameMeta(frame_index=idx, timestamp_ms=idx * 500.0, path=str(p))


# ---------------------------------------------------------------------------
# run_pipeline returns 6-tuple
# ---------------------------------------------------------------------------


class TestPipelineReturnsTuple:
    def test_run_pipeline_returns_six_tuple(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        result = run_pipeline("v", frames=[frame], detector=_OnePerson(), sample_fps=2.0)
        assert len(result) == 6, "run_pipeline must return a 6-tuple"

    def test_run_pipeline_empty_frames_returns_six_tuple(self):
        result = run_pipeline("v", frames=[], sample_fps=2.0)
        assert len(result) == 6


# ---------------------------------------------------------------------------
# features artifact structure
# ---------------------------------------------------------------------------


class TestFeaturesArtifact:
    def test_features_artifact_has_required_top_level_keys(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        _, _, _, _, feat, _ = run_pipeline(
            "v",
            frames=[frame],
            detector=_OnePerson(),
            tracker=IOUTracker(),
            pose_estimator=_FullBodyPoseEstimator(),
            sample_fps=2.0,
        )
        for key in ("video_id", "version", "feature_count", "features"):
            assert key in feat, f"features artifact missing key: {key}"

    def test_features_artifact_video_id(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        _, _, _, _, feat, _ = run_pipeline(
            "vid-42",
            frames=[frame],
            detector=_OnePerson(),
            tracker=IOUTracker(),
            pose_estimator=_FullBodyPoseEstimator(),
            sample_fps=2.0,
        )
        assert feat["video_id"] == "vid-42"

    def test_features_artifact_version_is_one(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        _, _, _, _, feat, _ = run_pipeline(
            "v",
            frames=[frame],
            detector=_OnePerson(),
            tracker=IOUTracker(),
            pose_estimator=_FullBodyPoseEstimator(),
            sample_fps=2.0,
        )
        assert feat["version"] == 1

    def test_features_artifact_feature_count_matches_list(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        _, _, _, _, feat, _ = run_pipeline(
            "v",
            frames=[frame],
            detector=_OnePerson(),
            tracker=IOUTracker(),
            pose_estimator=_FullBodyPoseEstimator(),
            sample_fps=2.0,
        )
        assert feat["feature_count"] == len(feat["features"])

    def test_full_body_pose_produces_non_empty_features(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        _, _, _, _, feat, _ = run_pipeline(
            "v",
            frames=[frame],
            detector=_OnePerson(),
            tracker=IOUTracker(),
            pose_estimator=_FullBodyPoseEstimator(),
            sample_fps=2.0,
        )
        assert feat["feature_count"] > 0
        assert len(feat["features"]) > 0

    def test_feature_entry_has_required_keys(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        _, _, _, _, feat, _ = run_pipeline(
            "v",
            frames=[frame],
            detector=_OnePerson(),
            tracker=IOUTracker(),
            pose_estimator=_FullBodyPoseEstimator(),
            sample_fps=2.0,
        )
        entry = feat["features"][0]
        for key in ("track_id", "name", "start_ms", "end_ms", "value"):
            assert key in entry, f"feature entry missing key: {key}"

    def test_stub_pose_gives_empty_features(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        _, _, _, _, feat, _ = run_pipeline(
            "v",
            frames=[frame],
            detector=_OnePerson(),
            tracker=IOUTracker(),
            pose_estimator=StubPoseEstimator(),
            sample_fps=2.0,
        )
        assert feat["feature_count"] == 0
        assert feat["features"] == []

    def test_empty_frames_gives_empty_features(self):
        _, _, _, _, feat, _ = run_pipeline("v", frames=[], sample_fps=2.0)
        assert feat["feature_count"] == 0
        assert feat["features"] == []

    def test_features_artifact_is_json_serialisable(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        _, _, _, _, feat, _ = run_pipeline(
            "v",
            frames=[frame],
            detector=_OnePerson(),
            tracker=IOUTracker(),
            pose_estimator=_FullBodyPoseEstimator(),
            sample_fps=2.0,
        )
        dumped = json.dumps(feat)
        parsed = json.loads(dumped)
        assert parsed["video_id"] == "v"

    def test_mock_deriver_result_appears_in_artifact(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        _, _, _, _, feat, _ = run_pipeline(
            "v",
            frames=[frame],
            detector=_OnePerson(),
            feature_deriver=_MockFeatureDeriver(),
            sample_fps=2.0,
        )
        assert feat["feature_count"] == 1
        assert feat["features"][0]["name"] == "mock_feature"
        assert feat["features"][0]["value"] == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# state artifact feature_summary
# ---------------------------------------------------------------------------


class TestStateFeatureSummary:
    def test_state_version_is_six(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        state, _, _, _, _, _ = run_pipeline(
            "v", frames=[frame], detector=_OnePerson(), sample_fps=2.0
        )
        assert state["version"] == 7

    def test_state_has_feature_summary_key(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        state, _, _, _, _, _ = run_pipeline(
            "v", frames=[frame], detector=_OnePerson(), sample_fps=2.0
        )
        assert "feature_summary" in state

    def test_feature_summary_has_required_keys(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        state, _, _, _, _, _ = run_pipeline(
            "v", frames=[frame], detector=_OnePerson(), sample_fps=2.0
        )
        fs = state["feature_summary"]
        for key in ("feature_count", "featured_track_count", "feature_names"):
            assert key in fs, f"feature_summary missing key: {key}"

    def test_feature_summary_counts_match_with_mock_deriver(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        state, _, _, _, _, _ = run_pipeline(
            "v",
            frames=[frame],
            detector=_OnePerson(),
            feature_deriver=_MockFeatureDeriver(),
            sample_fps=2.0,
        )
        fs = state["feature_summary"]
        assert fs["feature_count"] == 1
        assert fs["featured_track_count"] == 1
        assert "mock_feature" in fs["feature_names"]

    def test_feature_summary_empty_with_stub_pose(self, tmp_path):
        frames = [_make_frame(tmp_path, i) for i in range(2)]
        state, _, _, _, _, _ = run_pipeline(
            "v",
            frames=frames,
            detector=_OnePerson(),
            tracker=IOUTracker(),
            pose_estimator=StubPoseEstimator(),
            sample_fps=2.0,
        )
        fs = state["feature_summary"]
        assert fs["feature_count"] == 0
        assert fs["featured_track_count"] == 0
        assert fs["feature_names"] == []

    def test_feature_summary_non_empty_with_full_body_pose(self, tmp_path):
        frames = [_make_frame(tmp_path, i) for i in range(3)]
        state, _, _, _, _, _ = run_pipeline(
            "v",
            frames=frames,
            detector=_OnePerson(),
            tracker=IOUTracker(),
            pose_estimator=_FullBodyPoseEstimator(),
            sample_fps=2.0,
        )
        fs = state["feature_summary"]
        assert fs["feature_count"] > 0
        assert fs["featured_track_count"] >= 1
        assert len(fs["feature_names"]) > 0

    def test_state_notes_mention_feature_derivation(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        state, _, _, _, _, _ = run_pipeline(
            "v", frames=[frame], detector=_OnePerson(), sample_fps=2.0
        )
        assert "feature derivation" in state["notes"]

    def test_state_notes_still_mention_pose_estimation(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        state, _, _, _, _, _ = run_pipeline(
            "v", frames=[frame], detector=_OnePerson(), sample_fps=2.0
        )
        assert "pose estimation" in state["notes"]

    def test_state_notes_still_mention_tracking(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        state, _, _, _, _, _ = run_pipeline(
            "v", frames=[frame], detector=_OnePerson(), sample_fps=2.0
        )
        assert "tracking" in state["notes"]


# ---------------------------------------------------------------------------
# Pose timestamp propagation
# ---------------------------------------------------------------------------


class TestTimestampPropagation:
    def test_pose_timestamp_ms_is_populated(self, tmp_path):
        """Poses must carry their frame's timestamp_ms so feature deriver can use it."""
        frame = _make_frame(tmp_path, 4)  # timestamp_ms = 4 * 500 = 2000
        _, _, _, _, feat, _ = run_pipeline(
            "v",
            frames=[frame],
            detector=_OnePerson(),
            tracker=IOUTracker(),
            pose_estimator=_FullBodyPoseEstimator(),
            sample_fps=2.0,
        )
        # All single-frame features should have start_ms == end_ms == 2000
        per_frame = [f for f in feat["features"] if f["name"] == "shoulder_width"]
        assert len(per_frame) == 1
        assert per_frame[0]["start_ms"] == pytest.approx(2000.0)
        assert per_frame[0]["end_ms"] == pytest.approx(2000.0)

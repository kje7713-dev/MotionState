"""Tests for pose estimation integration in libs/pipeline/run_pipeline.py."""

import json
from pathlib import Path

import pytest

from libs.pipeline.contracts import BoundingBox, Detection, Frame, Keypoint, PoseEstimate, Track
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


class _MockPoseEstimator(StubPoseEstimator):
    """Returns a fixed pose estimate per tracked person visible in the frame."""

    def estimate(self, frame: Frame, tracks: list[Track]) -> list[PoseEstimate]:
        poses = []
        for track in tracks:
            for det in track.detections:
                if det.frame_index == frame.index:
                    poses.append(
                        PoseEstimate(
                            frame_index=frame.index,
                            track_id=track.track_id,
                            keypoints=[
                                Keypoint(name="nose", x=35.0, y=25.0, confidence=0.91),
                                Keypoint(
                                    name="left_shoulder", x=25.0, y=70.0, confidence=0.87
                                ),
                            ],
                        )
                    )
                    break
        return poses


def _make_frame(tmp_path: Path, idx: int) -> FrameMeta:
    p = tmp_path / f"frame_{idx:06d}.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16)
    return FrameMeta(frame_index=idx, timestamp_ms=idx * 500.0, path=str(p))


# ---------------------------------------------------------------------------
# Poses artifact structure
# ---------------------------------------------------------------------------


class TestPosesArtifact:
    def test_poses_artifact_has_required_top_level_keys(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        _, _, _, poses, _, _ = run_pipeline(
            "v",
            frames=[frame],
            detector=_OnePerson(),
            tracker=IOUTracker(),
            pose_estimator=_MockPoseEstimator(),
            sample_fps=2.0,
        )
        for key in ("video_id", "version", "pose_count", "poses"):
            assert key in poses, f"poses artifact missing key: {key}"

    def test_poses_artifact_video_id(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        _, _, _, poses, _, _ = run_pipeline(
            "vid-99",
            frames=[frame],
            detector=_OnePerson(),
            tracker=IOUTracker(),
            pose_estimator=_MockPoseEstimator(),
            sample_fps=2.0,
        )
        assert poses["video_id"] == "vid-99"

    def test_poses_artifact_version_is_one(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        _, _, _, poses, _, _ = run_pipeline(
            "v",
            frames=[frame],
            detector=_OnePerson(),
            tracker=IOUTracker(),
            pose_estimator=_MockPoseEstimator(),
            sample_fps=2.0,
        )
        assert poses["version"] == 1

    def test_one_person_one_frame_produces_one_pose(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        _, _, _, poses, _, _ = run_pipeline(
            "v",
            frames=[frame],
            detector=_OnePerson(),
            tracker=IOUTracker(),
            pose_estimator=_MockPoseEstimator(),
            sample_fps=2.0,
        )
        assert poses["pose_count"] == 1
        assert len(poses["poses"]) == 1

    def test_one_person_three_frames_produces_three_poses(self, tmp_path):
        frames = [_make_frame(tmp_path, i) for i in range(3)]
        _, _, _, poses, _, _ = run_pipeline(
            "v",
            frames=frames,
            detector=_OnePerson(),
            tracker=IOUTracker(),
            pose_estimator=_MockPoseEstimator(),
            sample_fps=2.0,
        )
        assert poses["pose_count"] == 3
        assert len(poses["poses"]) == 3

    def test_pose_entry_has_required_keys(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        _, _, _, poses, _, _ = run_pipeline(
            "v",
            frames=[frame],
            detector=_OnePerson(),
            tracker=IOUTracker(),
            pose_estimator=_MockPoseEstimator(),
            sample_fps=2.0,
        )
        entry = poses["poses"][0]
        for key in ("frame_index", "timestamp_ms", "track_id", "keypoints"):
            assert key in entry, f"pose entry missing key: {key}"

    def test_pose_entry_keypoint_has_required_keys(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        _, _, _, poses, _, _ = run_pipeline(
            "v",
            frames=[frame],
            detector=_OnePerson(),
            tracker=IOUTracker(),
            pose_estimator=_MockPoseEstimator(),
            sample_fps=2.0,
        )
        kp = poses["poses"][0]["keypoints"][0]
        for key in ("name", "x", "y", "confidence"):
            assert key in kp, f"keypoint missing key: {key}"

    def test_pose_timestamp_ms_matches_frame_meta(self, tmp_path):
        frame = _make_frame(tmp_path, 4)  # timestamp_ms = 4 * 500 = 2000
        _, _, _, poses, _, _ = run_pipeline(
            "v",
            frames=[frame],
            detector=_OnePerson(),
            tracker=IOUTracker(),
            pose_estimator=_MockPoseEstimator(),
            sample_fps=2.0,
        )
        assert poses["poses"][0]["timestamp_ms"] == pytest.approx(2000.0)

    def test_stub_pose_estimator_gives_empty_poses(self, tmp_path):
        frames = [_make_frame(tmp_path, i) for i in range(2)]
        _, _, _, poses, _, _ = run_pipeline(
            "v",
            frames=frames,
            detector=_OnePerson(),
            tracker=IOUTracker(),
            pose_estimator=StubPoseEstimator(),
            sample_fps=2.0,
        )
        assert poses["pose_count"] == 0
        assert poses["poses"] == []

    def test_empty_frames_gives_empty_poses(self):
        _, _, _, poses, _, _ = run_pipeline("v", frames=[], sample_fps=2.0)
        assert poses["pose_count"] == 0
        assert poses["poses"] == []

    def test_poses_artifact_is_json_serialisable(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        _, _, _, poses, _, _ = run_pipeline(
            "v",
            frames=[frame],
            detector=_OnePerson(),
            tracker=IOUTracker(),
            pose_estimator=_MockPoseEstimator(),
            sample_fps=2.0,
        )
        dumped = json.dumps(poses)
        parsed = json.loads(dumped)
        assert parsed["video_id"] == "v"


# ---------------------------------------------------------------------------
# State artifact pose_summary
# ---------------------------------------------------------------------------


class TestStatePoseSummary:
    def test_state_version_is_four(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        state, _, _, _, _, _ = run_pipeline(
            "v", frames=[frame], detector=_OnePerson(), sample_fps=2.0
        )
        assert state["version"] == 6

    def test_state_has_pose_summary_key(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        state, _, _, _, _, _ = run_pipeline(
            "v", frames=[frame], detector=_OnePerson(), sample_fps=2.0
        )
        assert "pose_summary" in state

    def test_pose_summary_has_required_keys(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        state, _, _, _, _, _ = run_pipeline(
            "v", frames=[frame], detector=_OnePerson(), sample_fps=2.0
        )
        ps = state["pose_summary"]
        for key in ("pose_count", "posed_track_count", "average_keypoints_per_pose"):
            assert key in ps, f"pose_summary missing key: {key}"

    def test_pose_summary_with_mock_estimator(self, tmp_path):
        frames = [_make_frame(tmp_path, i) for i in range(3)]
        state, _, _, _, _, _ = run_pipeline(
            "v",
            frames=frames,
            detector=_OnePerson(),
            tracker=IOUTracker(),
            pose_estimator=_MockPoseEstimator(),
            sample_fps=2.0,
        )
        ps = state["pose_summary"]
        assert ps["pose_count"] == 3
        assert ps["posed_track_count"] == 1
        assert ps["average_keypoints_per_pose"] == pytest.approx(2.0)

    def test_pose_summary_with_stub_estimator_is_zero(self, tmp_path):
        frames = [_make_frame(tmp_path, i) for i in range(3)]
        state, _, _, _, _, _ = run_pipeline(
            "v",
            frames=frames,
            detector=_OnePerson(),
            tracker=IOUTracker(),
            pose_estimator=StubPoseEstimator(),
            sample_fps=2.0,
        )
        ps = state["pose_summary"]
        assert ps["pose_count"] == 0
        assert ps["posed_track_count"] == 0
        assert ps["average_keypoints_per_pose"] == pytest.approx(0.0)

    def test_state_notes_mention_pose_estimation(self, tmp_path):
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
# Pipeline returns non-empty poses when tracking input exists
# ---------------------------------------------------------------------------


class TestPipelineReturnsNonEmptyPoses:
    def test_mock_estimator_produces_poses_with_keypoints(self, tmp_path):
        frames = [_make_frame(tmp_path, i) for i in range(2)]
        _, _, _, poses, _, _ = run_pipeline(
            "v",
            frames=frames,
            detector=_OnePerson(),
            tracker=IOUTracker(),
            pose_estimator=_MockPoseEstimator(),
            sample_fps=2.0,
        )
        assert poses["pose_count"] > 0
        assert all(len(p["keypoints"]) > 0 for p in poses["poses"])

    def test_pose_track_id_matches_tracker_output(self, tmp_path):
        frames = [_make_frame(tmp_path, i) for i in range(2)]
        _, _, tracks, poses, _, _ = run_pipeline(
            "v",
            frames=frames,
            detector=_OnePerson(),
            tracker=IOUTracker(),
            pose_estimator=_MockPoseEstimator(),
            sample_fps=2.0,
        )
        track_ids_in_tracks = {t["track_id"] for t in tracks["tracks"]}
        track_ids_in_poses = {p["track_id"] for p in poses["poses"]}
        # All pose track IDs should correspond to known tracks
        assert track_ids_in_poses.issubset(track_ids_in_tracks)

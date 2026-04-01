"""Focused tests for pose estimator contracts and implementations."""

from libs.pipeline.contracts import BoundingBox, Detection, Frame, Keypoint, PoseEstimate, Track
from libs.pipeline.pose import StubPoseEstimator

# ---------------------------------------------------------------------------
# StubPoseEstimator
# ---------------------------------------------------------------------------


class TestStubPoseEstimator:
    def test_returns_empty_list(self):
        estimator = StubPoseEstimator()
        frame = Frame(index=0, timestamp_ms=0.0, data=b"\xff\xd8\xff")
        result = estimator.estimate(frame, [])
        assert result == []

    def test_returns_empty_list_with_tracks(self):
        estimator = StubPoseEstimator()
        frame = Frame(index=0, timestamp_ms=0.0, data=b"\xff\xd8\xff")
        track = Track(
            track_id=1,
            detections=[
                Detection(
                    frame_index=0,
                    bbox=BoundingBox(x=10.0, y=20.0, width=50.0, height=100.0, confidence=0.9),
                )
            ],
        )
        result = estimator.estimate(frame, [track])
        assert result == []


# ---------------------------------------------------------------------------
# PoseEstimate contract
# ---------------------------------------------------------------------------


class TestPoseEstimateContract:
    def test_pose_estimate_has_required_fields(self):
        pose = PoseEstimate(
            frame_index=5,
            track_id=2,
            keypoints=[
                Keypoint(name="nose", x=120.0, y=80.0, confidence=0.92),
                Keypoint(name="left_shoulder", x=100.0, y=140.0, confidence=0.88),
            ],
        )
        assert pose.frame_index == 5
        assert pose.track_id == 2
        assert len(pose.keypoints) == 2

    def test_keypoint_has_required_fields(self):
        kp = Keypoint(name="nose", x=50.0, y=30.0, confidence=0.95)
        assert kp.name == "nose"
        assert kp.x == 50.0
        assert kp.y == 30.0
        assert kp.confidence == 0.95

    def test_keypoint_default_confidence(self):
        kp = Keypoint(name="left_eye", x=10.0, y=20.0)
        assert kp.confidence == 1.0

    def test_pose_estimate_default_empty_keypoints(self):
        pose = PoseEstimate(frame_index=0, track_id=1)
        assert pose.keypoints == []


# ---------------------------------------------------------------------------
# Mocked pose estimator returning structured PoseEstimate objects
# ---------------------------------------------------------------------------


class _MockedPoseEstimator(StubPoseEstimator):
    """Returns a structured PoseEstimate for each active track in the frame."""

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
                                Keypoint(name="nose", x=50.0, y=30.0, confidence=0.92),
                                Keypoint(
                                    name="left_shoulder", x=40.0, y=60.0, confidence=0.85
                                ),
                            ],
                        )
                    )
                    break
        return poses


class TestMockedPoseEstimator:
    def _make_track(self, track_id: int, frame_index: int) -> Track:
        return Track(
            track_id=track_id,
            detections=[
                Detection(
                    frame_index=frame_index,
                    bbox=BoundingBox(x=10.0, y=20.0, width=50.0, height=100.0, confidence=0.9),
                )
            ],
        )

    def test_returns_pose_for_active_track(self):
        estimator = _MockedPoseEstimator()
        frame = Frame(index=0, timestamp_ms=0.0, data=b"\xff\xd8\xff")
        track = self._make_track(track_id=1, frame_index=0)
        result = estimator.estimate(frame, [track])
        assert len(result) == 1

    def test_pose_has_correct_frame_index(self):
        estimator = _MockedPoseEstimator()
        frame = Frame(index=3, timestamp_ms=1500.0, data=b"\xff\xd8\xff")
        track = self._make_track(track_id=1, frame_index=3)
        result = estimator.estimate(frame, [track])
        assert result[0].frame_index == 3

    def test_pose_has_correct_track_id(self):
        estimator = _MockedPoseEstimator()
        frame = Frame(index=0, timestamp_ms=0.0, data=b"\xff\xd8\xff")
        track = self._make_track(track_id=7, frame_index=0)
        result = estimator.estimate(frame, [track])
        assert result[0].track_id == 7

    def test_pose_contains_keypoints(self):
        estimator = _MockedPoseEstimator()
        frame = Frame(index=0, timestamp_ms=0.0, data=b"\xff\xd8\xff")
        track = self._make_track(track_id=1, frame_index=0)
        result = estimator.estimate(frame, [track])
        assert len(result[0].keypoints) == 2

    def test_keypoint_fields_are_correct(self):
        estimator = _MockedPoseEstimator()
        frame = Frame(index=0, timestamp_ms=0.0, data=b"\xff\xd8\xff")
        track = self._make_track(track_id=1, frame_index=0)
        result = estimator.estimate(frame, [track])
        kp = result[0].keypoints[0]
        assert kp.name == "nose"
        assert kp.x == 50.0
        assert kp.y == 30.0
        assert kp.confidence == 0.92

    def test_returns_empty_when_no_active_tracks_in_frame(self):
        estimator = _MockedPoseEstimator()
        # Track has detection on frame 1, but we're querying frame 0
        frame = Frame(index=0, timestamp_ms=0.0, data=b"\xff\xd8\xff")
        track = self._make_track(track_id=1, frame_index=1)
        result = estimator.estimate(frame, [track])
        assert result == []

    def test_two_tracks_produce_two_poses(self):
        estimator = _MockedPoseEstimator()
        frame = Frame(index=0, timestamp_ms=0.0, data=b"\xff\xd8\xff")
        tracks = [
            self._make_track(track_id=1, frame_index=0),
            self._make_track(track_id=2, frame_index=0),
        ]
        result = estimator.estimate(frame, tracks)
        assert len(result) == 2
        track_ids = {p.track_id for p in result}
        assert track_ids == {1, 2}

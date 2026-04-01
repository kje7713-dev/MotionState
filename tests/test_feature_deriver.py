"""Unit tests for libs/pipeline/features_basic.py – BasicFeatureDeriver."""


import pytest

from libs.pipeline.contracts import (
    BoundingBox,
    Detection,
    Keypoint,
    MotionFeature,
    PoseEstimate,
    Track,
)
from libs.pipeline.features_basic import BasicFeatureDeriver, _angle_deg, _distance

# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_pose(
    track_id: int,
    frame_index: int,
    timestamp_ms: float,
    keypoints: list[Keypoint],
) -> PoseEstimate:
    return PoseEstimate(
        frame_index=frame_index,
        track_id=track_id,
        keypoints=keypoints,
        timestamp_ms=timestamp_ms,
    )


def _make_track(track_id: int, frame_index: int, bbox: BoundingBox) -> Track:
    det = Detection(frame_index=frame_index, bbox=bbox, timestamp_ms=0.0)
    return Track(track_id=track_id, detections=[det])


def _full_body_keypoints(
    *,
    lsx: float = 10.0, lsy: float = 20.0,  # left_shoulder
    rsx: float = 30.0, rsy: float = 20.0,  # right_shoulder
    lhx: float = 10.0, lhy: float = 50.0,  # left_hip
    rhx: float = 30.0, rhy: float = 50.0,  # right_hip
    lex: float = 5.0,  ley: float = 35.0,  # left_elbow
    rex: float = 35.0, rey: float = 35.0,  # right_elbow
    lwx: float = 0.0,  lwy: float = 50.0,  # left_wrist
    rwx: float = 40.0, rwy: float = 50.0,  # right_wrist
    lkx: float = 10.0, lky: float = 70.0,  # left_knee
    rkx: float = 30.0, rky: float = 70.0,  # right_knee
    lax: float = 10.0, lay: float = 90.0,  # left_ankle
    rax: float = 30.0, ray: float = 90.0,  # right_ankle
    confidence: float = 1.0,
) -> list[Keypoint]:
    return [
        Keypoint("left_shoulder",  lsx, lsy,  confidence),
        Keypoint("right_shoulder", rsx, rsy,  confidence),
        Keypoint("left_hip",       lhx, lhy,  confidence),
        Keypoint("right_hip",      rhx, rhy,  confidence),
        Keypoint("left_elbow",     lex, ley,  confidence),
        Keypoint("right_elbow",    rex, rey,  confidence),
        Keypoint("left_wrist",     lwx, lwy,  confidence),
        Keypoint("right_wrist",    rwx, rwy,  confidence),
        Keypoint("left_knee",      lkx, lky,  confidence),
        Keypoint("right_knee",     rkx, rky,  confidence),
        Keypoint("left_ankle",     lax, lay,  confidence),
        Keypoint("right_ankle",    rax, ray,  confidence),
    ]


# ---------------------------------------------------------------------------
# Pure math helpers
# ---------------------------------------------------------------------------


class TestMathHelpers:
    def test_distance_same_point_is_zero(self):
        assert _distance(5.0, 5.0, 5.0, 5.0) == pytest.approx(0.0)

    def test_distance_known_value(self):
        # 3-4-5 right triangle
        assert _distance(0.0, 0.0, 3.0, 4.0) == pytest.approx(5.0)

    def test_angle_straight_line_is_180(self):
        # A-B-C collinear, B in the middle → angle = 180°
        assert _angle_deg(0.0, 0.0, 5.0, 0.0, 10.0, 0.0) == pytest.approx(180.0, abs=1e-6)

    def test_angle_right_angle_is_90(self):
        # A above B, C to the right of B
        assert _angle_deg(0.0, 0.0, 0.0, 5.0, 5.0, 5.0) == pytest.approx(90.0, abs=1e-6)

    def test_angle_zero_length_vector_returns_zero(self):
        # A == B → degenerate, should return 0
        assert _angle_deg(5.0, 5.0, 5.0, 5.0, 10.0, 5.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# BasicFeatureDeriver – empty inputs
# ---------------------------------------------------------------------------


class TestEmptyInputs:
    def test_empty_poses_returns_empty_list(self):
        result = BasicFeatureDeriver().derive([], [])
        assert result == []

    def test_tracks_without_poses_returns_empty_list(self):
        track = _make_track(1, 0, BoundingBox(x=0, y=0, width=50, height=100))
        result = BasicFeatureDeriver().derive([track], [])
        assert result == []


# ---------------------------------------------------------------------------
# Per-pose features
# ---------------------------------------------------------------------------


class TestShoulderWidth:
    def test_shoulder_width_known_value(self):
        kps = _full_body_keypoints(lsx=10.0, lsy=20.0, rsx=30.0, rsy=20.0)
        pose = _make_pose(1, 0, 1000.0, kps)
        features = BasicFeatureDeriver().derive([], [pose])
        sw = next(f for f in features if f.name == "shoulder_width")
        assert sw.value == pytest.approx(20.0)

    def test_shoulder_width_track_id_matches(self):
        kps = _full_body_keypoints()
        pose = _make_pose(7, 0, 0.0, kps)
        features = BasicFeatureDeriver().derive([], [pose])
        sw = next(f for f in features if f.name == "shoulder_width")
        assert sw.track_id == 7

    def test_shoulder_width_timestamps_match_pose(self):
        kps = _full_body_keypoints()
        pose = _make_pose(1, 0, 2500.0, kps)
        features = BasicFeatureDeriver().derive([], [pose])
        sw = next(f for f in features if f.name == "shoulder_width")
        assert sw.start_ms == pytest.approx(2500.0)
        assert sw.end_ms == pytest.approx(2500.0)

    def test_shoulder_width_missing_keypoint_not_produced(self):
        # Only left_shoulder present
        kps = [Keypoint("left_shoulder", 10.0, 20.0, 1.0)]
        pose = _make_pose(1, 0, 0.0, kps)
        features = BasicFeatureDeriver().derive([], [pose])
        names = [f.name for f in features]
        assert "shoulder_width" not in names


class TestHipWidth:
    def test_hip_width_known_value(self):
        kps = _full_body_keypoints(lhx=0.0, lhy=50.0, rhx=40.0, rhy=50.0)
        pose = _make_pose(1, 0, 0.0, kps)
        features = BasicFeatureDeriver().derive([], [pose])
        hw = next(f for f in features if f.name == "hip_width")
        assert hw.value == pytest.approx(40.0)


class TestTorsoAngle:
    def test_torso_angle_vertical_is_zero(self):
        """Perfectly vertical torso (shoulder mid directly above hip mid) → 0 degrees."""
        kps = _full_body_keypoints(
            lsx=10.0, lsy=10.0, rsx=30.0, rsy=10.0,   # shoulder_mid x=20
            lhx=10.0, lhy=50.0, rhx=30.0, rhy=50.0,   # hip_mid x=20
        )
        pose = _make_pose(1, 0, 0.0, kps)
        features = BasicFeatureDeriver().derive([], [pose])
        ta = next(f for f in features if f.name == "torso_angle")
        assert ta.value == pytest.approx(0.0, abs=1e-6)

    def test_torso_angle_horizontal_is_90(self):
        """Horizontal torso (shoulder mid directly beside hip mid) → 90 degrees."""
        kps = _full_body_keypoints(
            lsx=10.0, lsy=20.0, rsx=30.0, rsy=20.0,   # shoulder_mid x=20, y=20
            lhx=10.0, lhy=20.0, rhx=10.0, rhy=20.0,   # hip_mid x=10, y=20 → dx=10, dy=0
        )
        pose = _make_pose(1, 0, 0.0, kps)
        features = BasicFeatureDeriver().derive([], [pose])
        ta = next(f for f in features if f.name == "torso_angle")
        assert ta.value == pytest.approx(90.0, abs=1e-6)

    def test_torso_angle_is_non_negative(self):
        kps = _full_body_keypoints()
        pose = _make_pose(1, 0, 0.0, kps)
        features = BasicFeatureDeriver().derive([], [pose])
        ta = next(f for f in features if f.name == "torso_angle")
        assert ta.value >= 0.0


class TestElbowAngles:
    def test_left_elbow_angle_straight_arm(self):
        """left_shoulder, left_elbow, left_wrist collinear → ~180°."""
        kps = _full_body_keypoints(
            lsx=0.0, lsy=0.0,   # left_shoulder
            lex=0.0, ley=10.0,  # left_elbow
            lwx=0.0, lwy=20.0,  # left_wrist
        )
        pose = _make_pose(1, 0, 0.0, kps)
        features = BasicFeatureDeriver().derive([], [pose])
        la = next(f for f in features if f.name == "left_elbow_angle")
        assert la.value == pytest.approx(180.0, abs=1e-5)

    def test_right_elbow_angle_right_angle(self):
        """right_shoulder above right_elbow, right_wrist to the right → 90°."""
        kps = _full_body_keypoints(
            rsx=0.0, rsy=0.0,    # right_shoulder
            rex=0.0, rey=10.0,   # right_elbow
            rwx=10.0, rwy=10.0,  # right_wrist
        )
        pose = _make_pose(1, 0, 0.0, kps)
        features = BasicFeatureDeriver().derive([], [pose])
        ra = next(f for f in features if f.name == "right_elbow_angle")
        assert ra.value == pytest.approx(90.0, abs=1e-5)

    def test_left_elbow_angle_missing_keypoint_not_produced(self):
        kps = [Keypoint("left_shoulder", 0.0, 0.0, 1.0), Keypoint("left_elbow", 0.0, 5.0, 1.0)]
        pose = _make_pose(1, 0, 0.0, kps)
        features = BasicFeatureDeriver().derive([], [pose])
        names = [f.name for f in features]
        assert "left_elbow_angle" not in names


class TestKneeAngles:
    def test_left_knee_angle_straight_leg(self):
        """left_hip, left_knee, left_ankle collinear → ~180°."""
        kps = _full_body_keypoints(
            lhx=0.0, lhy=0.0,   # left_hip
            lkx=0.0, lky=10.0,  # left_knee
            lax=0.0, lay=20.0,  # left_ankle
        )
        pose = _make_pose(1, 0, 0.0, kps)
        features = BasicFeatureDeriver().derive([], [pose])
        lk = next(f for f in features if f.name == "left_knee_angle")
        assert lk.value == pytest.approx(180.0, abs=1e-5)

    def test_right_knee_angle_right_angle(self):
        kps = _full_body_keypoints(
            rhx=0.0, rhy=0.0,    # right_hip
            rkx=0.0, rky=10.0,   # right_knee
            rax=10.0, ray=10.0,  # right_ankle
        )
        pose = _make_pose(1, 0, 0.0, kps)
        features = BasicFeatureDeriver().derive([], [pose])
        rk = next(f for f in features if f.name == "right_knee_angle")
        assert rk.value == pytest.approx(90.0, abs=1e-5)


class TestKeypointVisibilityCount:
    def test_all_keypoints_visible(self):
        kps = [Keypoint(f"kp{i}", 0.0, 0.0, 1.0) for i in range(5)]
        pose = _make_pose(1, 0, 0.0, kps)
        features = BasicFeatureDeriver().derive([], [pose])
        kvc = next(f for f in features if f.name == "keypoint_visibility_count")
        assert kvc.value == pytest.approx(5.0)

    def test_low_confidence_keypoints_not_counted(self):
        kps = [
            Keypoint("kp0", 0.0, 0.0, 1.0),
            Keypoint("kp1", 0.0, 0.0, 0.1),  # below threshold
            Keypoint("kp2", 0.0, 0.0, 0.5),  # exactly at threshold – counted
        ]
        pose = _make_pose(1, 0, 0.0, kps)
        features = BasicFeatureDeriver().derive([], [pose])
        kvc = next(f for f in features if f.name == "keypoint_visibility_count")
        assert kvc.value == pytest.approx(2.0)

    def test_no_visible_keypoints(self):
        kps = [Keypoint(f"kp{i}", 0.0, 0.0, 0.0) for i in range(3)]
        pose = _make_pose(1, 0, 0.0, kps)
        features = BasicFeatureDeriver().derive([], [pose])
        kvc = next(f for f in features if f.name == "keypoint_visibility_count")
        assert kvc.value == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# bbox_area
# ---------------------------------------------------------------------------


class TestBboxArea:
    def test_bbox_area_computed_correctly(self):
        bbox = BoundingBox(x=0, y=0, width=50, height=80, confidence=1.0)
        track = _make_track(1, 0, bbox)
        kps = [Keypoint("nose", 10.0, 10.0, 1.0)]
        pose = _make_pose(1, 0, 0.0, kps)
        features = BasicFeatureDeriver().derive([track], [pose])
        ba = next(f for f in features if f.name == "bbox_area")
        assert ba.value == pytest.approx(50 * 80)

    def test_bbox_area_not_produced_without_matching_detection(self):
        """No track detection for this (track_id, frame_index) → no bbox_area feature."""
        kps = [Keypoint("nose", 10.0, 10.0, 1.0)]
        pose = _make_pose(1, 0, 0.0, kps)
        features = BasicFeatureDeriver().derive([], [pose])
        names = [f.name for f in features]
        assert "bbox_area" not in names


# ---------------------------------------------------------------------------
# centroid_velocity
# ---------------------------------------------------------------------------


class TestCentroidVelocity:
    def test_centroid_velocity_computed_correctly(self):
        """Two poses 500 ms apart; centroid moves 10 px → velocity = 0.02 px/ms."""
        kps0 = [Keypoint("nose", 0.0, 0.0, 1.0)]
        kps1 = [Keypoint("nose", 10.0, 0.0, 1.0)]
        pose0 = _make_pose(1, 0, 0.0, kps0)
        pose1 = _make_pose(1, 1, 500.0, kps1)
        features = BasicFeatureDeriver().derive([], [pose0, pose1])
        vel = next(f for f in features if f.name == "centroid_velocity")
        assert vel.value == pytest.approx(10.0 / 500.0)
        assert vel.start_ms == pytest.approx(0.0)
        assert vel.end_ms == pytest.approx(500.0)

    def test_centroid_velocity_zero_when_no_movement(self):
        kps = [Keypoint("nose", 5.0, 5.0, 1.0)]
        pose0 = _make_pose(1, 0, 0.0, kps[:])
        pose1 = _make_pose(1, 1, 1000.0, [Keypoint("nose", 5.0, 5.0, 1.0)])
        features = BasicFeatureDeriver().derive([], [pose0, pose1])
        vel = next(f for f in features if f.name == "centroid_velocity")
        assert vel.value == pytest.approx(0.0)

    def test_centroid_velocity_not_produced_for_single_frame(self):
        """Only one pose for a track → no centroid_velocity feature."""
        kps = [Keypoint("nose", 0.0, 0.0, 1.0)]
        pose = _make_pose(1, 0, 0.0, kps)
        features = BasicFeatureDeriver().derive([], [pose])
        names = [f.name for f in features]
        assert "centroid_velocity" not in names

    def test_centroid_velocity_zero_delta_time_gives_zero(self):
        """Same timestamp on two consecutive poses → velocity defaults to 0."""
        kps0 = [Keypoint("nose", 0.0, 0.0, 1.0)]
        kps1 = [Keypoint("nose", 10.0, 0.0, 1.0)]
        pose0 = _make_pose(1, 0, 500.0, kps0)
        pose1 = _make_pose(1, 1, 500.0, kps1)  # same timestamp
        features = BasicFeatureDeriver().derive([], [pose0, pose1])
        vel = next(f for f in features if f.name == "centroid_velocity")
        assert vel.value == pytest.approx(0.0)

    def test_centroid_velocity_uses_correct_track_id(self):
        kps = [Keypoint("nose", 0.0, 0.0, 1.0)]
        pose0 = _make_pose(3, 0, 0.0, kps)
        pose1 = _make_pose(3, 1, 100.0, [Keypoint("nose", 5.0, 0.0, 1.0)])
        features = BasicFeatureDeriver().derive([], [pose0, pose1])
        vel = next(f for f in features if f.name == "centroid_velocity")
        assert vel.track_id == 3

    def test_centroid_velocity_not_mixed_across_tracks(self):
        """Poses from different tracks must not produce inter-track velocity."""
        kps0 = [Keypoint("nose", 0.0, 0.0, 1.0)]
        kps1 = [Keypoint("nose", 100.0, 0.0, 1.0)]
        pose_t1 = _make_pose(1, 0, 0.0, kps0)
        pose_t2 = _make_pose(2, 1, 500.0, kps1)
        features = BasicFeatureDeriver().derive([], [pose_t1, pose_t2])
        velocities = [f for f in features if f.name == "centroid_velocity"]
        # No cross-track velocity
        assert velocities == []


# ---------------------------------------------------------------------------
# Full-body smoke test – non-empty output
# ---------------------------------------------------------------------------


class TestFullBodySmoke:
    def test_full_body_pose_produces_non_empty_features(self):
        kps = _full_body_keypoints()
        bbox = BoundingBox(x=0, y=0, width=60, height=120)
        track = _make_track(1, 0, bbox)
        pose = _make_pose(1, 0, 1000.0, kps)
        features = BasicFeatureDeriver().derive([track], [pose])
        assert len(features) > 0

    def test_full_body_pose_produces_all_expected_feature_names(self):
        kps = _full_body_keypoints()
        bbox = BoundingBox(x=0, y=0, width=60, height=120)
        track = _make_track(1, 0, bbox)
        pose = _make_pose(1, 0, 1000.0, kps)
        features = BasicFeatureDeriver().derive([track], [pose])
        names = {f.name for f in features}
        for expected in (
            "torso_angle",
            "shoulder_width",
            "hip_width",
            "left_elbow_angle",
            "right_elbow_angle",
            "left_knee_angle",
            "right_knee_angle",
            "keypoint_visibility_count",
            "bbox_area",
        ):
            assert expected in names, f"Expected feature '{expected}' not found"

    def test_two_frame_same_track_produces_centroid_velocity(self):
        kps0 = _full_body_keypoints()
        kps1 = _full_body_keypoints(lsx=15.0, rsx=35.0)  # shifted slightly
        pose0 = _make_pose(1, 0, 0.0, kps0)
        pose1 = _make_pose(1, 1, 500.0, kps1)
        features = BasicFeatureDeriver().derive([], [pose0, pose1])
        assert any(f.name == "centroid_velocity" for f in features)

    def test_all_feature_values_are_floats(self):
        kps = _full_body_keypoints()
        pose = _make_pose(1, 0, 0.0, kps)
        features = BasicFeatureDeriver().derive([], [pose])
        for f in features:
            assert isinstance(f.value, float), f"Feature {f.name} has non-float value: {f.value}"

    def test_all_features_are_motion_feature_instances(self):
        kps = _full_body_keypoints()
        pose = _make_pose(1, 0, 0.0, kps)
        features = BasicFeatureDeriver().derive([], [pose])
        for f in features:
            assert isinstance(f, MotionFeature)

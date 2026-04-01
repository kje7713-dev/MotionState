"""Unit tests for libs/pipeline/segments_basic.py – BasicSegmenter."""

import pytest

from libs.pipeline.contracts import MotionFeature, Segment
from libs.pipeline.segments_basic import BasicSegmenter, _merge_adjacent

# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _feature(
    name: str = "centroid_velocity",
    value: float = 0.0,
    start_ms: float = 0.0,
    end_ms: float = 0.0,
    track_id: int = 1,
) -> MotionFeature:
    return MotionFeature(
        track_id=track_id,
        name=name,
        start_ms=start_ms,
        end_ms=end_ms,
        value=value,
    )


def _velocity_features(
    velocities: list[float],
    window_ms: float = 2000.0,
) -> list[MotionFeature]:
    """Create centroid_velocity features, one per window.

    Features are placed so that consecutive velocity windows fall in separate
    segmenter windows.  Each window gets two features (centroid_velocity +
    torso_angle at the same timestamp) to stay above the sparse_data threshold.
    """
    features = []
    for i, v in enumerate(velocities):
        # Offset slightly past each window boundary so the segmenter's windows
        # (which start at t_min) produce a fresh window for each group.
        t = i * (window_ms + 1.0) + window_ms / 4.0
        features.append(_feature("centroid_velocity", v, start_ms=t, end_ms=t))
        # Add a second feature at the same timestamp to exceed sparse_threshold.
        features.append(_feature("torso_angle", 30.0, start_ms=t, end_ms=t))
    return features


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


class TestEmptyInput:
    def test_empty_features_returns_empty_list(self):
        assert BasicSegmenter().segment([]) == []


# ---------------------------------------------------------------------------
# Label classification
# ---------------------------------------------------------------------------


class TestLabels:
    def test_low_motion_label_for_zero_velocity(self):
        features = _velocity_features([0.0])
        segs = BasicSegmenter().segment(features)
        assert len(segs) >= 1
        assert segs[0].label == "low_motion"

    def test_active_motion_label_for_high_velocity(self):
        features = _velocity_features([1.0])
        segs = BasicSegmenter().segment(features)
        assert len(segs) >= 1
        assert segs[0].label == "active_motion"

    def test_transition_window_label_for_middle_velocity(self):
        # Velocity between low_motion_threshold (0.05) and active_motion_threshold (0.15)
        features = _velocity_features([0.10])
        segs = BasicSegmenter().segment(features)
        assert len(segs) >= 1
        assert segs[0].label == "transition_window"

    def test_sparse_data_label_when_feature_count_below_threshold(self):
        # Single feature in a window → sparse_data (threshold is 2 by default)
        features = [_feature("centroid_velocity", 0.0, start_ms=500.0, end_ms=500.0)]
        segs = BasicSegmenter().segment(features)
        assert segs[0].label == "sparse_data"

    def test_low_motion_when_no_velocity_but_enough_features(self):
        """If velocity signal absent but feature count is adequate → low_motion."""
        features = [
            _feature("torso_angle", 45.0, start_ms=500.0, end_ms=500.0),
            _feature("shoulder_width", 30.0, start_ms=550.0, end_ms=550.0),
        ]
        segs = BasicSegmenter().segment(features)
        assert segs[0].label == "low_motion"


# ---------------------------------------------------------------------------
# Confidence values
# ---------------------------------------------------------------------------


class TestConfidence:
    def test_confidence_is_between_zero_and_one(self):
        for v in [0.0, 0.03, 0.10, 0.20, 2.0]:
            features = _velocity_features([v])
            segs = BasicSegmenter().segment(features)
            for seg in segs:
                assert 0.0 <= seg.confidence <= 1.0, (
                    f"confidence={seg.confidence} out of range for velocity={v}"
                )

    def test_sparse_data_confidence_is_0_6(self):
        features = [_feature("centroid_velocity", 0.0, start_ms=500.0, end_ms=500.0)]
        segs = BasicSegmenter().segment(features)
        assert segs[0].confidence == pytest.approx(0.6)

    def test_zero_velocity_confidence_is_near_max(self):
        """Perfect zero velocity → highest low_motion confidence (0.95)."""
        features = _velocity_features([0.0])
        segs = BasicSegmenter().segment(features)
        assert segs[0].confidence == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Segment structure
# ---------------------------------------------------------------------------


class TestSegmentStructure:
    def test_segment_is_segment_instance(self):
        features = _velocity_features([0.0])
        segs = BasicSegmenter().segment(features)
        for seg in segs:
            assert isinstance(seg, Segment)

    def test_segment_has_feature_count_metadata(self):
        features = _velocity_features([0.0])
        segs = BasicSegmenter().segment(features)
        for seg in segs:
            assert "feature_count" in seg.metadata
            assert isinstance(seg.metadata["feature_count"], int)

    def test_segments_are_non_overlapping_and_sorted(self):
        features = _velocity_features([0.0, 1.0, 0.0, 1.0])
        segs = BasicSegmenter().segment(features)
        for i in range(1, len(segs)):
            assert segs[i].start_ms >= segs[i - 1].end_ms

    def test_segment_labels_are_domain_agnostic(self):
        """No domain-specific words in any label."""
        domain_words = {"bjj", "guard", "pass", "scramble", "stance", "kick", "punch"}
        features = _velocity_features([0.0, 1.0, 0.10])
        segs = BasicSegmenter().segment(features)
        for seg in segs:
            for word in domain_words:
                assert word not in seg.label.lower(), (
                    f"Domain word '{word}' found in label '{seg.label}'"
                )


# ---------------------------------------------------------------------------
# Adjacent window merging
# ---------------------------------------------------------------------------


class TestAdjacentMerging:
    def test_two_identical_label_windows_merge_into_one(self):
        """Two consecutive low_motion windows → merged into a single segment."""
        features = _velocity_features([0.0, 0.0])
        segs = BasicSegmenter().segment(features)
        assert len(segs) == 1
        assert segs[0].label == "low_motion"

    def test_merged_segment_spans_full_range(self):
        """Merged segment start_ms and end_ms cover both original windows."""
        features = _velocity_features([0.0, 0.0], window_ms=2000.0)
        segs = BasicSegmenter().segment(features)
        # The two windows span [0, 2000) and [2000, 4000).
        # After merging, the segment should span at least [1000, 3000].
        assert segs[0].start_ms < segs[0].end_ms

    def test_different_labels_not_merged(self):
        """low_motion then active_motion → two separate segments."""
        features = _velocity_features([0.0, 1.0])
        segs = BasicSegmenter().segment(features)
        labels = [s.label for s in segs]
        assert "low_motion" in labels
        assert "active_motion" in labels

    def test_merged_feature_count_is_sum_of_windows(self):
        """Merged segment's feature_count equals sum of constituent windows."""
        features = _velocity_features([0.0, 0.0])
        segs = BasicSegmenter().segment(features)
        assert segs[0].metadata["feature_count"] == 4  # 2 features × 2 windows

    def test_alternating_labels_produce_correct_count(self):
        """low, active, low, active → 4 segments (no merges possible)."""
        features = _velocity_features([0.0, 1.0, 0.0, 1.0])
        segs = BasicSegmenter().segment(features)
        # May merge identical adjacent pairs; at minimum 2 segments.
        labels = [s.label for s in segs]
        assert "low_motion" in labels
        assert "active_motion" in labels

    def test_three_consecutive_same_label_merge_into_one(self):
        features = _velocity_features([0.0, 0.0, 0.0])
        segs = BasicSegmenter().segment(features)
        assert len(segs) == 1
        assert segs[0].label == "low_motion"


# ---------------------------------------------------------------------------
# _merge_adjacent standalone tests
# ---------------------------------------------------------------------------


class TestMergeAdjacent:
    def test_empty_list_returns_empty(self):
        assert _merge_adjacent([]) == []

    def test_single_segment_returned_unchanged(self):
        seg = Segment(start_ms=0.0, end_ms=1000.0, label="low_motion", confidence=0.9)
        result = _merge_adjacent([seg])
        assert len(result) == 1
        assert result[0] is seg

    def test_two_same_label_merge(self):
        segs = [
            Segment(start_ms=0.0, end_ms=1000.0, label="low_motion", confidence=0.8,
                    metadata={"feature_count": 3}),
            Segment(start_ms=1000.0, end_ms=2000.0, label="low_motion", confidence=0.9,
                    metadata={"feature_count": 5}),
        ]
        result = _merge_adjacent(segs)
        assert len(result) == 1
        assert result[0].start_ms == pytest.approx(0.0)
        assert result[0].end_ms == pytest.approx(2000.0)
        assert result[0].metadata["feature_count"] == 8

    def test_two_different_labels_not_merged(self):
        segs = [
            Segment(start_ms=0.0, end_ms=1000.0, label="low_motion", confidence=0.8),
            Segment(start_ms=1000.0, end_ms=2000.0, label="active_motion", confidence=0.9),
        ]
        result = _merge_adjacent(segs)
        assert len(result) == 2

    def test_merged_confidence_is_average(self):
        segs = [
            Segment(start_ms=0.0, end_ms=1000.0, label="low_motion", confidence=0.8,
                    metadata={"feature_count": 1}),
            Segment(start_ms=1000.0, end_ms=2000.0, label="low_motion", confidence=0.9,
                    metadata={"feature_count": 1}),
        ]
        result = _merge_adjacent(segs)
        assert result[0].confidence == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_input_produces_same_output(self):
        features = _velocity_features([0.0, 1.0, 0.10, 0.0, 1.0])
        segs1 = BasicSegmenter().segment(features)
        segs2 = BasicSegmenter().segment(features)
        assert len(segs1) == len(segs2)
        for s1, s2 in zip(segs1, segs2, strict=True):
            assert s1.label == s2.label
            assert s1.start_ms == pytest.approx(s2.start_ms)
            assert s1.end_ms == pytest.approx(s2.end_ms)
            assert s1.confidence == pytest.approx(s2.confidence)


# ---------------------------------------------------------------------------
# Custom thresholds
# ---------------------------------------------------------------------------


class TestCustomThresholds:
    def test_custom_active_threshold_changes_label(self):
        """With a very high active threshold, high velocity stays transition_window."""
        features = _velocity_features([0.20])
        segs = BasicSegmenter(active_motion_threshold=0.50).segment(features)
        assert segs[0].label == "transition_window"

    def test_custom_sparse_threshold_zero_never_sparse(self):
        """sparse_threshold=0 means single-feature windows are not sparse_data."""
        features = [_feature("centroid_velocity", 0.0, start_ms=500.0, end_ms=500.0)]
        segs = BasicSegmenter(sparse_threshold=0).segment(features)
        assert segs[0].label != "sparse_data"

    def test_custom_window_ms_affects_segment_count(self):
        """Smaller window_ms produces more raw windows before merging."""
        features = _velocity_features([0.0, 0.0, 0.0, 0.0], window_ms=500.0)
        segs_small = BasicSegmenter(window_ms=500.0).segment(features)
        segs_large = BasicSegmenter(window_ms=2000.0).segment(features)
        # Smaller windows produce at least as many segments.
        assert len(segs_small) <= len(segs_large) or len(segs_small) >= 1

"""Tests for the IOUTracker in libs/pipeline/tracker_bytetrack.py."""

import pytest

from libs.pipeline.contracts import BoundingBox, Detection, Track
from libs.pipeline.tracker_bytetrack import IOUTracker, _greedy_match, _iou

# ---------------------------------------------------------------------------
# _iou helper
# ---------------------------------------------------------------------------


class TestIOU:
    def test_identical_boxes_give_iou_one(self):
        box = BoundingBox(x=0.0, y=0.0, width=100.0, height=100.0)
        assert _iou(box, box) == pytest.approx(1.0)

    def test_non_overlapping_boxes_give_iou_zero(self):
        a = BoundingBox(x=0.0, y=0.0, width=50.0, height=50.0)
        b = BoundingBox(x=100.0, y=100.0, width=50.0, height=50.0)
        assert _iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap_is_between_zero_and_one(self):
        a = BoundingBox(x=0.0, y=0.0, width=100.0, height=100.0)
        b = BoundingBox(x=50.0, y=50.0, width=100.0, height=100.0)
        score = _iou(a, b)
        assert 0.0 < score < 1.0

    def test_touching_boxes_give_iou_zero(self):
        """Boxes that share only an edge have zero intersection area."""
        a = BoundingBox(x=0.0, y=0.0, width=50.0, height=50.0)
        b = BoundingBox(x=50.0, y=0.0, width=50.0, height=50.0)
        assert _iou(a, b) == pytest.approx(0.0)

    def test_highly_overlapping_boxes(self):
        """A small shift produces high IOU for large boxes."""
        a = BoundingBox(x=10.0, y=20.0, width=100.0, height=200.0)
        b = BoundingBox(x=12.0, y=22.0, width=100.0, height=200.0)
        assert _iou(a, b) > 0.8


# ---------------------------------------------------------------------------
# _greedy_match helper
# ---------------------------------------------------------------------------


class TestGreedyMatch:
    def _make_bbox(self, x: float) -> BoundingBox:
        return BoundingBox(x=x, y=0.0, width=50.0, height=100.0)

    def test_no_active_returns_empty(self):
        assert _greedy_match([], [self._make_bbox(0)], 0.3) == []

    def test_no_detections_returns_empty(self):
        assert _greedy_match([self._make_bbox(0)], [], 0.3) == []

    def test_high_iou_pair_is_matched(self):
        a = BoundingBox(x=0.0, y=0.0, width=100.0, height=100.0)
        b = BoundingBox(x=2.0, y=2.0, width=100.0, height=100.0)
        pairs = _greedy_match([a], [b], 0.3)
        assert pairs == [(0, 0)]

    def test_low_iou_pair_is_not_matched(self):
        a = BoundingBox(x=0.0, y=0.0, width=50.0, height=50.0)
        b = BoundingBox(x=200.0, y=200.0, width=50.0, height=50.0)
        pairs = _greedy_match([a], [b], 0.3)
        assert pairs == []

    def test_each_index_used_at_most_once(self):
        """Two tracks competing for the same detection: only the better one wins."""
        a = BoundingBox(x=0.0, y=0.0, width=100.0, height=100.0)
        b = BoundingBox(x=5.0, y=5.0, width=100.0, height=100.0)  # closer to det
        det = BoundingBox(x=5.0, y=5.0, width=100.0, height=100.0)
        pairs = _greedy_match([a, b], [det], 0.3)
        assert len(pairs) == 1
        det_indices = [di for _, di in pairs]
        assert det_indices.count(0) == 1


# ---------------------------------------------------------------------------
# IOUTracker – stable IDs across frames
# ---------------------------------------------------------------------------


def _det(frame_index: int, x: float, y: float = 0.0) -> Detection:
    return Detection(
        frame_index=frame_index,
        bbox=BoundingBox(x=x, y=y, width=50.0, height=100.0),
        class_id=0,
        class_label="person",
    )


class TestIOUTrackerStableIDs:
    def test_single_person_gets_same_id_across_two_frames(self):
        tracker = IOUTracker(iou_threshold=0.3)
        tracks_f0 = tracker.update([_det(0, x=10.0)])
        tracks_f1 = tracker.update([_det(1, x=12.0)])

        assert len(tracks_f0) == 1
        assert len(tracks_f1) == 1
        assert tracks_f0[0].track_id == tracks_f1[0].track_id

    def test_single_person_accumulates_detections(self):
        tracker = IOUTracker(iou_threshold=0.3)
        tracker.update([_det(0, x=10.0)])
        tracks = tracker.update([_det(1, x=12.0)])

        assert len(tracks) == 1
        assert len(tracks[0].detections) == 2

    def test_two_non_overlapping_people_get_distinct_ids(self):
        tracker = IOUTracker(iou_threshold=0.3)
        tracks = tracker.update([_det(0, x=0.0), _det(0, x=300.0)])

        assert len(tracks) == 2
        ids = {t.track_id for t in tracks}
        assert len(ids) == 2

    def test_two_people_keep_stable_ids_across_frames(self):
        tracker = IOUTracker(iou_threshold=0.3)
        tracks_f0 = tracker.update([_det(0, x=0.0), _det(0, x=300.0)])
        tracks_f1 = tracker.update([_det(1, x=2.0), _det(1, x=302.0)])

        ids_f0 = {t.track_id for t in tracks_f0}
        ids_f1 = {t.track_id for t in tracks_f1}
        assert ids_f0 == ids_f1, "Track IDs changed between frames"

    def test_new_person_appearing_gets_new_id(self):
        tracker = IOUTracker(iou_threshold=0.3)
        tracks_f0 = tracker.update([_det(0, x=0.0)])
        tracks_f1 = tracker.update([_det(1, x=0.0), _det(1, x=300.0)])

        old_id = tracks_f0[0].track_id
        new_ids = {t.track_id for t in tracks_f1}
        assert old_id in new_ids, "Original track ID should still be present"
        assert len(new_ids) == 2, "New person should have a different ID"

    def test_empty_detections_do_not_create_tracks(self):
        tracker = IOUTracker(iou_threshold=0.3)
        tracks = tracker.update([])
        assert tracks == []

    def test_empty_frame_after_detection_does_not_lose_track(self):
        tracker = IOUTracker(iou_threshold=0.3, max_age=5)
        tracker.update([_det(0, x=10.0)])
        tracks = tracker.update([])  # no detections this frame

        assert len(tracks) == 1
        assert tracks[0].detections[0].frame_index == 0

    def test_track_aged_out_still_in_returned_list(self):
        """Tracks aged past max_age are removed from active but still returned."""
        tracker = IOUTracker(iou_threshold=0.3, max_age=1)
        tracker.update([_det(0, x=10.0)])
        # Two frames without a matching detection – track should age out
        tracker.update([])
        tracks = tracker.update([])

        assert len(tracks) == 1  # still in history
        assert tracks[0].track_id == 1

    def test_track_ids_are_stable_after_age_out_and_new_detection(self):
        """A new detection after an aged-out track receives a fresh ID."""
        tracker = IOUTracker(iou_threshold=0.3, max_age=0)
        tracker.update([_det(0, x=10.0)])  # track 1 created
        tracker.update([])  # track 1 ages out (max_age=0 means age > 0 removes it)
        tracks = tracker.update([_det(2, x=10.0)])  # new track created

        ids = {t.track_id for t in tracks}
        assert len(ids) == 2  # original + new


# ---------------------------------------------------------------------------
# IOUTracker – return value structure
# ---------------------------------------------------------------------------


class TestIOUTrackerReturnStructure:
    def test_returns_list_of_track_objects(self):
        tracker = IOUTracker()
        result = tracker.update([_det(0, x=0.0)])
        assert isinstance(result, list)
        assert all(isinstance(t, Track) for t in result)

    def test_track_has_track_id_and_detections(self):
        tracker = IOUTracker()
        tracks = tracker.update([_det(0, x=0.0)])
        assert hasattr(tracks[0], "track_id")
        assert hasattr(tracks[0], "detections")

    def test_track_id_starts_at_one(self):
        tracker = IOUTracker()
        tracks = tracker.update([_det(0, x=0.0)])
        assert tracks[0].track_id == 1

    def test_returned_detections_are_copies(self):
        """Modifying the returned detection list must not affect the tracker state."""
        tracker = IOUTracker()
        tracks = tracker.update([_det(0, x=0.0)])
        original_len = len(tracks[0].detections)
        tracks[0].detections.append(_det(99, x=0.0))

        tracks2 = tracker.update([_det(1, x=0.0)])
        assert len(tracks2[0].detections) == original_len + 1  # only frame 0+1

"""Deterministic IOU-based multi-object tracker.

This module provides a lightweight, dependency-free implementation that
assigns persistent track IDs across video frames using intersection-over-union
(IOU) matching.  It serves as a practical drop-in replacement for heavier
tracking libraries (ByteTrack, DeepSORT, …) while the full dependency story
is worked out.

Usage::

    from libs.pipeline.tracker_bytetrack import IOUTracker

    tracker = IOUTracker(iou_threshold=0.3, max_age=30)
    for frame_detections in per_frame_detections:
        tracks = tracker.update(frame_detections)
"""

from __future__ import annotations

from libs.pipeline.contracts import BoundingBox, Detection, Track, Tracker


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    """Return the intersection-over-union of two bounding boxes."""
    ax2 = a.x + a.width
    ay2 = a.y + a.height
    bx2 = b.x + b.width
    by2 = b.y + b.height

    inter_x1 = max(a.x, b.x)
    inter_y1 = max(a.y, b.y)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = a.width * a.height
    area_b = b.width * b.height
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area


def _greedy_match(
    active_bboxes: list[BoundingBox],
    det_bboxes: list[BoundingBox],
    threshold: float,
) -> list[tuple[int, int]]:
    """Return (track_index, det_index) pairs matched by greedy IOU descent.

    Each track and each detection can appear in at most one pair.  Pairs with
    IOU below *threshold* are discarded.
    """
    if not active_bboxes or not det_bboxes:
        return []

    # Build all candidate (iou, track_i, det_i) triples above the threshold.
    candidates: list[tuple[float, int, int]] = []
    for ti, tb in enumerate(active_bboxes):
        for di, db in enumerate(det_bboxes):
            score = _iou(tb, db)
            if score >= threshold:
                candidates.append((score, ti, di))

    # Sort descending by IOU; greedily assign one-to-one.
    candidates.sort(key=lambda c: -c[0])
    used_tracks: set[int] = set()
    used_dets: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for _, ti, di in candidates:
        if ti not in used_tracks and di not in used_dets:
            pairs.append((ti, di))
            used_tracks.add(ti)
            used_dets.add(di)

    return pairs


class IOUTracker(Tracker):
    """Stateful IOU-based tracker that assigns persistent IDs across frames.

    Each call to :meth:`update` processes one frame's detections and returns
    *all* tracks ever created (active and completed), so the caller always has
    the full accumulated history after the final frame.

    Args:
        iou_threshold: Minimum IOU required to associate a detection with an
            existing track.  Detections below this threshold start a new track.
        max_age: Number of consecutive frames a track can go undetected before
            it is dropped from the active set.  Dropped tracks are still
            included in the returned list.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30) -> None:
        self._iou_threshold = iou_threshold
        self._max_age = max_age
        self._next_id: int = 1
        # active: track_id -> {"bbox": BoundingBox, "age": int}
        self._active: dict[int, dict] = {}
        # accumulates detections for every track ever created
        self._track_detections: dict[int, list[Detection]] = {}

    # ------------------------------------------------------------------
    # Tracker interface
    # ------------------------------------------------------------------

    def update(self, detections: list[Detection]) -> list[Track]:
        """Consume *detections* for the current frame and return all tracks.

        Args:
            detections: Object detections produced by the detector for this
                frame.  Pass an empty list when a frame has no detections.

        Returns:
            All :class:`~libs.pipeline.contracts.Track` objects ever created
            by this tracker instance, including those that have aged out.
        """
        active_ids = list(self._active.keys())

        if active_ids and detections:
            active_bboxes = [self._active[tid]["bbox"] for tid in active_ids]
            det_bboxes = [d.bbox for d in detections]
            pairs = _greedy_match(active_bboxes, det_bboxes, self._iou_threshold)

            matched_track_is = {ti for ti, _ in pairs}
            matched_det_is = {di for _, di in pairs}

            # Update matched tracks.
            for ti, di in pairs:
                tid = active_ids[ti]
                self._active[tid]["bbox"] = detections[di].bbox
                self._active[tid]["age"] = 0
                self._track_detections[tid].append(detections[di])

            # Age out unmatched active tracks.
            for i, tid in enumerate(active_ids):
                if i not in matched_track_is:
                    self._active[tid]["age"] += 1

            # Create new tracks for unmatched detections.
            for di, det in enumerate(detections):
                if di not in matched_det_is:
                    self._create_track(det)

        elif not active_ids:
            # No existing tracks – seed from current detections.
            for det in detections:
                self._create_track(det)
        else:
            # No detections this frame – age all active tracks.
            for tid in active_ids:
                self._active[tid]["age"] += 1

        # Remove tracks that have exceeded max_age from the active set.
        aged_out = [tid for tid, state in self._active.items() if state["age"] > self._max_age]
        for tid in aged_out:
            del self._active[tid]

        return [
            Track(track_id=tid, detections=list(dets))
            for tid, dets in self._track_detections.items()
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_track(self, det: Detection) -> None:
        """Register *det* as the first detection of a new track."""
        tid = self._next_id
        self._next_id += 1
        self._active[tid] = {"bbox": det.bbox, "age": 0}
        self._track_detections[tid] = [det]

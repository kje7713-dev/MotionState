"""Concrete temporal segmenter – intensity thresholding with window merging.

Divides a motion-feature time-series into fixed-size windows, classifies each
window with a generic label, and merges adjacent windows that share the same
label.

Labels (all domain-agnostic):
    ``low_motion``       – low centroid velocity; person is relatively still.
    ``active_motion``    – high centroid velocity; person is actively moving.
    ``transition_window``– between the low and active thresholds; ambiguous.
    ``sparse_data``      – too few features in the window to classify reliably.
"""

from __future__ import annotations

from libs.pipeline.contracts import MotionFeature, Segment, Segmenter

# Default classification thresholds (px/ms for centroid_velocity).
_DEFAULT_LOW_MOTION_THRESHOLD: float = 0.05
_DEFAULT_ACTIVE_MOTION_THRESHOLD: float = 0.15

# Windows with fewer features than this are labelled ``sparse_data``.
_DEFAULT_SPARSE_THRESHOLD: int = 2

# Default window width in milliseconds.
_DEFAULT_WINDOW_MS: float = 2000.0


class BasicSegmenter(Segmenter):
    """Deterministic segmenter based on motion-intensity thresholding.

    Algorithm:
    1. Compute the overall time range covered by *features*.
    2. Divide the range into fixed-size windows of *window_ms* milliseconds.
    3. For each window, collect the features whose time interval overlaps it.
    4. Classify the window:
       * ``sparse_data`` if fewer than *sparse_threshold* features overlap.
       * ``low_motion`` if the average ``centroid_velocity`` is below
         *low_motion_threshold* (or no velocity features are present).
       * ``active_motion`` if the average ``centroid_velocity`` is at or above
         *active_motion_threshold*.
       * ``transition_window`` otherwise (between the two thresholds).
    5. Merge consecutive windows that share the same label.

    Args:
        window_ms: Width of each analysis window in milliseconds.
        sparse_threshold: Minimum number of features required to classify
            a window beyond ``sparse_data``.
        low_motion_threshold: ``centroid_velocity`` (px/ms) below which a
            window is considered ``low_motion``.
        active_motion_threshold: ``centroid_velocity`` (px/ms) at or above
            which a window is considered ``active_motion``.
    """

    def __init__(
        self,
        *,
        window_ms: float = _DEFAULT_WINDOW_MS,
        sparse_threshold: int = _DEFAULT_SPARSE_THRESHOLD,
        low_motion_threshold: float = _DEFAULT_LOW_MOTION_THRESHOLD,
        active_motion_threshold: float = _DEFAULT_ACTIVE_MOTION_THRESHOLD,
    ) -> None:
        self.window_ms = window_ms
        self.sparse_threshold = sparse_threshold
        self.low_motion_threshold = low_motion_threshold
        self.active_motion_threshold = active_motion_threshold

    # ------------------------------------------------------------------
    # Segmenter contract
    # ------------------------------------------------------------------

    def segment(self, features: list[MotionFeature]) -> list[Segment]:
        """Return generic temporal segments derived from *features*.

        Returns an empty list when *features* is empty.
        """
        if not features:
            return []

        all_times = [f.start_ms for f in features] + [f.end_ms for f in features]
        t_min = min(all_times)
        t_max = max(all_times)

        # Extend t_max by 1 ms so that features sitting at the exact maximum
        # timestamp are captured by the strict less-than window overlap test.
        t_max_ext = t_max + 1.0

        # Build window boundaries, ensuring at least one window exists.
        windows: list[tuple[float, float]] = []
        t = t_min
        while t < t_max_ext:
            windows.append((t, min(t + self.window_ms, t_max_ext)))
            t += self.window_ms
        if not windows:
            windows.append((t_min, t_min + self.window_ms))

        raw_segments = [self._classify_window(ws, we, features) for ws, we in windows]
        return _merge_adjacent(raw_segments)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_window(
        self,
        win_start: float,
        win_end: float,
        features: list[MotionFeature],
    ) -> Segment:
        """Classify a single time window and return a :class:`Segment`."""
        window_features = [
            f for f in features if f.start_ms < win_end and f.end_ms >= win_start
        ]
        feature_count = len(window_features)

        velocities = [
            f.value
            for f in window_features
            if f.name == "centroid_velocity" and isinstance(f.value, (int, float))
        ]
        avg_velocity: float | None = (
            sum(velocities) / len(velocities) if velocities else None
        )

        label, confidence = self._label(feature_count, avg_velocity)
        return Segment(
            start_ms=win_start,
            end_ms=win_end,
            label=label,
            confidence=confidence,
            metadata={"feature_count": feature_count},
        )

    def _label(
        self,
        feature_count: int,
        avg_velocity: float | None,
    ) -> tuple[str, float]:
        """Return *(label, confidence)* for a window."""
        if feature_count < self.sparse_threshold:
            return "sparse_data", 0.6

        if avg_velocity is None:
            # Features present but no velocity signal – default to low_motion.
            return "low_motion", 0.65

        if avg_velocity < self.low_motion_threshold:
            ratio = 1.0 - (avg_velocity / self.low_motion_threshold)
            confidence = round(0.6 + 0.35 * ratio, 4)
            return "low_motion", confidence

        if avg_velocity >= self.active_motion_threshold:
            ratio = min(
                1.0,
                (avg_velocity - self.active_motion_threshold) / self.active_motion_threshold,
            )
            confidence = round(0.7 + 0.25 * ratio, 4)
            return "active_motion", confidence

        return "transition_window", 0.65


def _merge_adjacent(segments: list[Segment]) -> list[Segment]:
    """Merge consecutive segments that share the same label.

    The merged segment spans from the first window's ``start_ms`` to the
    last window's ``end_ms``.  ``feature_count`` is summed; ``confidence``
    is averaged.
    """
    if not segments:
        return []

    merged: list[Segment] = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg.label == prev.label:
            prev_count = prev.metadata.get("feature_count", 0)
            seg_count = seg.metadata.get("feature_count", 0)
            merged[-1] = Segment(
                start_ms=prev.start_ms,
                end_ms=seg.end_ms,
                label=prev.label,
                confidence=round((prev.confidence + seg.confidence) / 2, 4),
                metadata={"feature_count": prev_count + seg_count},
            )
        else:
            merged.append(seg)

    return merged

"""Orchestrates the full CV pipeline for a single video.

This module ties together the detector, tracker, pose estimator, feature
deriver, and segmenter.  Currently all stages use stub implementations that
return empty results.  The pipeline returns a structured state dict that is
persisted as ``state.json`` alongside each processed video.
"""

from libs.pipeline.contracts import (
    Detector,
    FeatureDeriver,
    PoseEstimator,
    Segmenter,
    Tracker,
)
from libs.pipeline.detector import StubDetector
from libs.pipeline.features import StubFeatureDeriver
from libs.pipeline.pose import StubPoseEstimator
from libs.pipeline.segments import StubSegmenter
from libs.pipeline.tracker import StubTracker


def run_pipeline(
    video_id: int | str,
    *,
    detector: Detector | None = None,
    tracker: Tracker | None = None,
    pose_estimator: PoseEstimator | None = None,
    feature_deriver: FeatureDeriver | None = None,
    segmenter: Segmenter | None = None,
) -> dict:
    """Run the full pipeline and return a structured state artifact dict.

    Each stage defaults to its stub implementation.  Pass concrete instances
    to wire in real CV logic without changing this orchestration layer.

    Args:
        video_id: The ID of the video being processed.
        detector: Object detector to use (default: StubDetector).
        tracker: Multi-object tracker to use (default: StubTracker).
        pose_estimator: Pose estimator to use (default: StubPoseEstimator).
        feature_deriver: Feature deriver to use (default: StubFeatureDeriver).
        segmenter: Temporal segmenter to use (default: StubSegmenter).

    Returns:
        A dict matching the state artifact schema.
    """
    _detector = detector or StubDetector()
    _tracker = tracker or StubTracker()
    _pose = pose_estimator or StubPoseEstimator()
    _features = feature_deriver or StubFeatureDeriver()
    _segmenter = segmenter or StubSegmenter()

    # TODO: iterate frames from video, call _detector.detect(frame),
    #       _tracker.update(detections), _pose.estimate(frame, tracks)
    all_tracks: list = []
    all_poses: list = []

    features = _features.derive(all_tracks, all_poses)
    segments = _segmenter.segment(features)

    return {
        "video_id": str(video_id),
        "version": 1,
        "segments": [vars(s) for s in segments],
        "tracks": [],
        "features": [vars(f) for f in features],
        "notes": "placeholder artifact; CV pipeline not yet implemented",
    }

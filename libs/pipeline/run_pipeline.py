"""Orchestrates the full CV pipeline for a single video.

This module ties together the detector, tracker, pose estimator, feature
deriver, and segmenter.  The first real CV stage (frame extraction +
person detection) is now implemented; later stages still use stub
implementations.  The pipeline returns both a state dict (``state.json``)
and a detections dict (``detections.json``).
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

from libs.pipeline.contracts import (
    Detection,
    Detector,
    FeatureDeriver,
    Frame,
    PoseEstimator,
    Segmenter,
    Tracker,
)
from libs.pipeline.detector import StubDetector
from libs.pipeline.features import StubFeatureDeriver
from libs.pipeline.pose import StubPoseEstimator
from libs.pipeline.segments import StubSegmenter
from libs.pipeline.tracker import StubTracker
from libs.video.frames import FrameMeta

logger = logging.getLogger(__name__)


def run_pipeline(
    video_id: int | str,
    frames: list[FrameMeta] | None = None,
    *,
    detector: Detector | None = None,
    tracker: Tracker | None = None,
    pose_estimator: PoseEstimator | None = None,
    feature_deriver: FeatureDeriver | None = None,
    segmenter: Segmenter | None = None,
    sample_fps: float = 2.0,
) -> tuple[dict, dict]:
    """Run the full pipeline and return ``(state, detections)`` artifact dicts.

    Each stage defaults to its stub implementation.  Pass concrete instances
    to wire in real CV logic without changing this orchestration layer.

    Args:
        video_id: The ID of the video being processed.
        frames: Pre-extracted :class:`~libs.video.frames.FrameMeta` list.
            When provided each frame is loaded from disk and passed through
            the detector.  When ``None`` the detection stage is skipped and
            the detections artifact will contain an empty frame list.
        detector: Object detector to use (default: StubDetector).
        tracker: Multi-object tracker to use (default: StubTracker).
        pose_estimator: Pose estimator to use (default: StubPoseEstimator).
        feature_deriver: Feature deriver to use (default: StubFeatureDeriver).
        segmenter: Temporal segmenter to use (default: StubSegmenter).
        sample_fps: Sample rate used during frame extraction (informational
            only; stored in the detections artifact).

    Returns:
        A 2-tuple ``(state_dict, detections_dict)`` matching their respective
        artifact schemas.
    """
    _detector = detector or StubDetector()
    _tracker = tracker or StubTracker()
    _pose = pose_estimator or StubPoseEstimator()
    _features = feature_deriver or StubFeatureDeriver()
    _segmenter = segmenter or StubSegmenter()

    # ------------------------------------------------------------------
    # Detection stage – iterate extracted frames
    # ------------------------------------------------------------------
    detections_frames: list[dict] = []
    all_detections: list[Detection] = []

    for frame_meta in frames or []:
        frame_path = Path(frame_meta.path)
        if not frame_path.exists():
            logger.warning("Frame file missing, skipping: %s", frame_path)
            detections_frames.append(
                {
                    "frame_index": frame_meta.frame_index,
                    "timestamp_ms": frame_meta.timestamp_ms,
                    "path": frame_meta.path,
                    "detections": [],
                }
            )
            continue

        raw = frame_path.read_bytes()
        cv_frame = Frame(
            index=frame_meta.frame_index,
            timestamp_ms=frame_meta.timestamp_ms,
            data=raw,
        )

        frame_detections = _detector.detect(cv_frame)
        all_detections.extend(frame_detections)

        detections_frames.append(
            {
                "frame_index": frame_meta.frame_index,
                "timestamp_ms": frame_meta.timestamp_ms,
                "path": frame_meta.path,
                "detections": [
                    {
                        "class_label": d.class_label,
                        "bbox": asdict(d.bbox),
                    }
                    for d in frame_detections
                ],
            }
        )

    # ------------------------------------------------------------------
    # Later stages (stubs for now)
    # ------------------------------------------------------------------
    all_tracks: list = []
    all_poses: list = []
    features = _features.derive(all_tracks, all_poses)
    segments = _segmenter.segment(features)

    # ------------------------------------------------------------------
    # Build detections summary for state artifact
    # ------------------------------------------------------------------
    frame_count = len(detections_frames)
    frames_with_people = sum(1 for f in detections_frames if f["detections"])
    total_detections = len(all_detections)

    detections_artifact = {
        "video_id": str(video_id),
        "version": 1,
        "sample_fps": sample_fps,
        "frames": detections_frames,
    }

    state_artifact = {
        "video_id": str(video_id),
        "version": 2,
        "segments": [vars(s) for s in segments],
        "tracks": [],
        "features": [vars(f) for f in features],
        "detections_summary": {
            "frame_count": frame_count,
            "frames_with_people": frames_with_people,
            "total_detections": total_detections,
        },
        "notes": "first real CV stage: frame extraction and person detection",
    }

    return state_artifact, detections_artifact

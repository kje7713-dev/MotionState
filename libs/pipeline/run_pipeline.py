"""Orchestrates the full CV pipeline for a single video.

This module ties together the detector, tracker, pose estimator, feature
deriver, and segmenter.  Frame extraction, person detection, multi-frame
tracking, and pose estimation are now implemented; feature derivation and
segmentation still use stub implementations.  The pipeline returns four
artifacts: a state dict (``state.json``), a detections dict
(``detections.json``), a tracks dict (``tracks.json``), and a poses dict
(``poses.json``).
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
    PoseEstimate,
    PoseEstimator,
    Segmenter,
    Track,
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
) -> tuple[dict, dict, dict, dict]:
    """Run the full pipeline and return ``(state, detections, tracks, poses)`` dicts.

    Each stage defaults to its stub implementation.  Pass concrete instances
    to wire in real CV logic without changing this orchestration layer.

    Args:
        video_id: The ID of the video being processed.
        frames: Pre-extracted :class:`~libs.video.frames.FrameMeta` list.
            When provided each frame is loaded from disk and passed through
            the detector, tracker, and pose estimator.  When ``None`` all CV
            stages are skipped and the artifacts will contain empty collections.
        detector: Object detector to use (default: StubDetector).
        tracker: Multi-object tracker to use (default: StubTracker).
        pose_estimator: Pose estimator to use (default: StubPoseEstimator).
        feature_deriver: Feature deriver to use (default: StubFeatureDeriver).
        segmenter: Temporal segmenter to use (default: StubSegmenter).
        sample_fps: Sample rate used during frame extraction (informational
            only; stored in the detections artifact).

    Returns:
        A 4-tuple ``(state_dict, detections_dict, tracks_dict, poses_dict)``
        matching their respective artifact schemas.
    """
    _detector = detector or StubDetector()
    _tracker = tracker or StubTracker()
    _pose = pose_estimator or StubPoseEstimator()
    _features = feature_deriver or StubFeatureDeriver()
    _segmenter = segmenter or StubSegmenter()

    # Build a frame_index -> timestamp_ms lookup for the tracks artifact.
    timestamp_by_frame: dict[int, float] = {
        fm.frame_index: fm.timestamp_ms for fm in (frames or [])
    }

    # Detection + tracking + pose estimation stage - iterate extracted frames
    detections_frames: list[dict] = []
    all_detections: list[Detection] = []
    all_tracks: list[Track] = []
    all_poses: list[PoseEstimate] = []

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

        # Pass this frame's detections to the tracker.
        all_tracks = _tracker.update(frame_detections)

        # Run pose estimation for tracked humans in this frame.
        frame_poses = _pose.estimate(cv_frame, all_tracks)
        all_poses.extend(frame_poses)

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

    # Feature derivation and segmentation are still stubs.
    features = _features.derive(all_tracks, all_poses)
    segments = _segmenter.segment(features)

    # Build detections summary for state artifact
    frame_count = len(detections_frames)
    frames_with_people = sum(1 for f in detections_frames if f["detections"])
    total_detections = len(all_detections)

    # Build tracking summary
    tracked_frame_indices: set[int] = set()
    for track in all_tracks:
        for det in track.detections:
            tracked_frame_indices.add(det.frame_index)
    tracked_frame_count = len(tracked_frame_indices)
    avg_detections_per_frame = (
        total_detections / tracked_frame_count if tracked_frame_count > 0 else 0.0
    )

    # Build pose summary
    posed_track_ids: set[int] = {p.track_id for p in all_poses}
    total_keypoints = sum(len(p.keypoints) for p in all_poses)
    avg_keypoints_per_pose = total_keypoints / len(all_poses) if all_poses else 0.0

    # Build tracks artifact
    tracks_artifact = _build_tracks_artifact(video_id, all_tracks, timestamp_by_frame)

    # Build poses artifact
    poses_artifact = _build_poses_artifact(video_id, all_poses, timestamp_by_frame)

    detections_artifact = {
        "video_id": str(video_id),
        "version": 1,
        "sample_fps": sample_fps,
        "frames": detections_frames,
    }

    state_artifact = {
        "video_id": str(video_id),
        "version": 4,
        "segments": [vars(s) for s in segments],
        "tracks": tracks_artifact["tracks"],
        "features": [vars(f) for f in features],
        "detections_summary": {
            "frame_count": frame_count,
            "frames_with_people": frames_with_people,
            "total_detections": total_detections,
        },
        "tracking_summary": {
            "track_count": len(all_tracks),
            "tracked_frame_count": tracked_frame_count,
            "average_detections_per_frame": avg_detections_per_frame,
        },
        "pose_summary": {
            "pose_count": len(all_poses),
            "posed_track_count": len(posed_track_ids),
            "average_keypoints_per_pose": avg_keypoints_per_pose,
        },
        "notes": (
            "first real CV stages: frame extraction, person detection, tracking, pose estimation"
        ),
    }

    return state_artifact, detections_artifact, tracks_artifact, poses_artifact


def _build_tracks_artifact(
    video_id: int | str,
    tracks: list[Track],
    timestamp_by_frame: dict[int, float],
) -> dict:
    """Serialise *tracks* into the ``tracks.json`` artifact dict."""
    tracks_data = []
    for track in tracks:
        det_list = [
            {
                "frame_index": d.frame_index,
                "timestamp_ms": timestamp_by_frame.get(d.frame_index, 0.0),
                "class_label": d.class_label,
                "bbox": asdict(d.bbox),
            }
            for d in track.detections
        ]
        tracks_data.append({"track_id": track.track_id, "detections": det_list})

    return {
        "video_id": str(video_id),
        "version": 1,
        "track_count": len(tracks),
        "tracks": tracks_data,
    }


def _build_poses_artifact(
    video_id: int | str,
    poses: list[PoseEstimate],
    timestamp_by_frame: dict[int, float],
) -> dict:
    """Serialise *poses* into the ``poses.json`` artifact dict."""
    poses_data = [
        {
            "frame_index": p.frame_index,
            "timestamp_ms": timestamp_by_frame.get(p.frame_index, 0.0),
            "track_id": p.track_id,
            "keypoints": [
                {
                    "name": kp.name,
                    "x": kp.x,
                    "y": kp.y,
                    "confidence": kp.confidence,
                }
                for kp in p.keypoints
            ],
        }
        for p in poses
    ]

    return {
        "video_id": str(video_id),
        "version": 1,
        "pose_count": len(poses),
        "poses": poses_data,
    }

"""Orchestrates the full CV pipeline for a single video.

This module ties together the detector, tracker, pose estimator, feature
deriver, and segmenter.  Frame extraction, person detection, multi-frame
tracking, pose estimation, generic motion feature derivation, and temporal
segmentation are all implemented.  The pipeline returns six artifacts: a
state dict (``state.json``), a detections dict (``detections.json``), a
tracks dict (``tracks.json``), a poses dict (``poses.json``), a features
dict (``features.json``), and a segments dict (``segments.json``).
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
    MotionFeature,
    PoseEstimate,
    PoseEstimator,
    Segment,
    Segmenter,
    Track,
    Tracker,
)
from libs.pipeline.detector import StubDetector
from libs.pipeline.features_basic import BasicFeatureDeriver
from libs.pipeline.pose import StubPoseEstimator
from libs.pipeline.segments_basic import BasicSegmenter
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
) -> tuple[dict, dict, dict, dict, dict, dict]:
    """Run the full pipeline and return
    ``(state, detections, tracks, poses, features, segments)`` dicts.

    Each stage defaults to its concrete implementation.  Pass concrete instances
    to wire in real CV logic without changing this orchestration layer.

    Args:
        video_id: The ID of the video being processed.
        frames: Pre-extracted :class:`~libs.video.frames.FrameMeta` list.
            When provided each frame is loaded from disk and passed through
            the detector, tracker, pose estimator, and feature deriver.  When
            ``None`` all CV stages are skipped and the artifacts will contain
            empty collections.
        detector: Object detector to use (default: StubDetector).
        tracker: Multi-object tracker to use (default: StubTracker).
        pose_estimator: Pose estimator to use (default: StubPoseEstimator).
        feature_deriver: Feature deriver to use (default: BasicFeatureDeriver).
        segmenter: Temporal segmenter to use (default: BasicSegmenter).
        sample_fps: Sample rate used during frame extraction (informational
            only; stored in the detections artifact).

    Returns:
        A 6-tuple ``(state_dict, detections_dict, tracks_dict, poses_dict,
        features_dict, segments_dict)`` matching their respective artifact
        schemas.
    """
    _detector = detector or StubDetector()
    _tracker = tracker or StubTracker()
    _pose = pose_estimator or StubPoseEstimator()
    _features = feature_deriver or BasicFeatureDeriver()
    _segmenter = segmenter or BasicSegmenter()

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
        # Stamp each detection with this frame's timestamp so downstream
        # stages (e.g. feature deriver) can recover temporal information.
        for det in frame_detections:
            det.timestamp_ms = frame_meta.timestamp_ms
        all_detections.extend(frame_detections)

        # Pass this frame's detections to the tracker.
        all_tracks = _tracker.update(frame_detections)

        # Run pose estimation for tracked humans in this frame.
        frame_poses = _pose.estimate(cv_frame, all_tracks)
        # Stamp poses with this frame's timestamp.
        for pose in frame_poses:
            pose.timestamp_ms = frame_meta.timestamp_ms
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

    # Feature derivation and segmentation.
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

    # Build feature summary
    featured_track_ids: set[int] = {f.track_id for f in features}
    feature_names: list[str] = sorted({f.name for f in features})

    # Build segmentation summary
    segment_labels: list[str] = sorted({s.label for s in segments})
    total_segment_duration_ms = sum(s.end_ms - s.start_ms for s in segments)

    # Build tracks artifact
    tracks_artifact = _build_tracks_artifact(video_id, all_tracks, timestamp_by_frame)

    # Build poses artifact
    poses_artifact = _build_poses_artifact(video_id, all_poses, timestamp_by_frame)

    # Build features artifact
    features_artifact = _build_features_artifact(video_id, features)

    # Build segments artifact
    segments_artifact = _build_segments_artifact(video_id, segments)

    detections_artifact = {
        "video_id": str(video_id),
        "version": 1,
        "sample_fps": sample_fps,
        "frames": detections_frames,
    }

    state_artifact = {
        "video_id": str(video_id),
        "version": 6,
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
        "feature_summary": {
            "feature_count": len(features),
            "featured_track_count": len(featured_track_ids),
            "feature_names": feature_names,
        },
        "segmentation_summary": {
            "segment_count": len(segments),
            "segment_labels": segment_labels,
            "total_segment_duration_ms": total_segment_duration_ms,
        },
        "notes": (
            "first real CV stages: frame extraction, person detection, tracking, "
            "pose estimation, feature derivation, temporal segmentation"
        ),
    }

    return (
        state_artifact,
        detections_artifact,
        tracks_artifact,
        poses_artifact,
        features_artifact,
        segments_artifact,
    )


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


def _build_features_artifact(
    video_id: int | str,
    features: list[MotionFeature],
) -> dict:
    """Serialise *features* into the ``features.json`` artifact dict."""
    return {
        "video_id": str(video_id),
        "version": 1,
        "feature_count": len(features),
        "features": [
            {
                "track_id": f.track_id,
                "name": f.name,
                "start_ms": f.start_ms,
                "end_ms": f.end_ms,
                "value": f.value,
            }
            for f in features
        ],
    }


def _build_segments_artifact(
    video_id: int | str,
    segments: list[Segment],
) -> dict:
    """Serialise *segments* into the ``segments.json`` artifact dict."""
    return {
        "video_id": str(video_id),
        "version": 1,
        "segment_count": len(segments),
        "segments": [
            {
                "start_ms": s.start_ms,
                "end_ms": s.end_ms,
                "label": s.label,
                "confidence": s.confidence,
                "metadata": s.metadata,
            }
            for s in segments
        ],
    }

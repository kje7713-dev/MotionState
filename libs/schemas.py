"""Pydantic response models for MotionState artifact endpoints.

These models document the stable public shapes returned by the queryable API.
They are intentionally pragmatic: top-level fields are typed; deeply-nested
per-frame or per-keypoint structures are left as ``Any`` to avoid over-modelling
and to remain resilient to minor schema evolution within a version.

Schema version constants are defined here so route code and tests can reference
them without using magic numbers.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Schema version constants
# ---------------------------------------------------------------------------

STATE_SCHEMA_VERSION: int = 7
DETECTIONS_SCHEMA_VERSION: int = 1
TRACKS_SCHEMA_VERSION: int = 1
POSES_SCHEMA_VERSION: int = 1
FEATURES_SCHEMA_VERSION: int = 1
SEGMENTS_SCHEMA_VERSION: int = 1
TIMELINE_MANIFEST_SCHEMA_VERSION: int = 1


# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------


class ArtifactRecord(BaseModel):
    """A single artifact metadata row as returned by ``GET /videos/{id}/artifacts``."""

    id: int
    video_id: int
    type: str
    path: str
    metadata_json: dict[str, Any] | None
    created_at: Any  # datetime, serialised as ISO string in responses


# ---------------------------------------------------------------------------
# Artifact response models
# ---------------------------------------------------------------------------


class StateArtifact(BaseModel):
    """Parsed content of ``state.json``."""

    video_id: str
    version: int
    segments: list[Any]
    tracks: list[Any]
    features: list[Any]
    detections_summary: dict[str, Any]
    tracking_summary: dict[str, Any]
    pose_summary: dict[str, Any]
    feature_summary: dict[str, Any]
    segmentation_summary: dict[str, Any]
    clip_summary: dict[str, Any]
    manifest_path: str
    notes: str


class DetectionsArtifact(BaseModel):
    """Parsed content of ``detections.json``."""

    video_id: str
    version: int
    sample_fps: float
    frames: list[Any]


class TracksArtifact(BaseModel):
    """Parsed content of ``tracks.json``."""

    video_id: str
    version: int
    track_count: int
    tracks: list[Any]


class PosesArtifact(BaseModel):
    """Parsed content of ``poses.json``."""

    video_id: str
    version: int
    pose_count: int
    poses: list[Any]


class FeaturesArtifact(BaseModel):
    """Parsed content of ``features.json``."""

    video_id: str
    version: int
    feature_count: int
    features: list[Any]


class SegmentsArtifact(BaseModel):
    """Parsed content of ``segments.json``."""

    video_id: str
    version: int
    segment_count: int
    segments: list[Any]


class TimelineManifest(BaseModel):
    """Parsed content of ``timeline_manifest.json``."""

    video_id: str
    version: int
    duration_seconds: float
    artifacts: dict[str, str]
    timeline: list[Any]

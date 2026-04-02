"""Lightweight typed response models for the MotionState SDK."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class UploadResponse:
    """Returned by :meth:`~motionstate_client.MotionStateClient.upload_video`."""

    video_id: int
    job_id: int
    processing_run_id: int


@dataclass
class UploadInitResponse:
    """Returned by :meth:`~motionstate_client.MotionStateClient.upload_init`."""

    video_id: int
    upload_url: str | None
    storage_key: str


@dataclass
class VideoResponse:
    """Returned by :meth:`~motionstate_client.MotionStateClient.get_video`."""

    id: int
    original_filename: str
    status: str
    source_path: str | None
    normalized_path: str | None
    duration_seconds: float | None
    fps: float | None
    width: int | None
    height: int | None
    created_at: str
    updated_at: str


@dataclass
class ProcessingRun:
    """A single processing run record."""

    id: int
    status: str
    trigger_type: str
    pipeline_version: int | None
    created_at: str
    completed_at: str | None
    error: str | None


@dataclass
class ReprocessResponse:
    """Returned by :meth:`~motionstate_client.MotionStateClient.reprocess_video`."""

    video_id: int
    processing_run_id: int
    job_id: int


@dataclass
class ArtifactRecord:
    """A single artifact metadata row."""

    id: int
    video_id: int
    type: str
    path: str
    metadata_json: dict[str, Any] | None
    created_at: str


def _parse_run(data: dict[str, Any]) -> ProcessingRun:
    return ProcessingRun(
        id=data["id"],
        status=str(data.get("status", "")),
        trigger_type=str(data.get("trigger_type", "")),
        pipeline_version=data.get("pipeline_version"),
        created_at=str(data.get("created_at", "")),
        completed_at=str(data["completed_at"]) if data.get("completed_at") else None,
        error=data.get("error"),
    )


def _parse_artifact(data: dict[str, Any]) -> ArtifactRecord:
    return ArtifactRecord(
        id=data["id"],
        video_id=data["video_id"],
        type=data["type"],
        path=data["path"],
        metadata_json=data.get("metadata_json"),
        created_at=str(data.get("created_at", "")),
    )


def _parse_video(data: dict[str, Any]) -> VideoResponse:
    return VideoResponse(
        id=data["id"],
        original_filename=data.get("original_filename", ""),
        status=str(data.get("status", "")),
        source_path=data.get("source_path"),
        normalized_path=data.get("normalized_path"),
        duration_seconds=data.get("duration_seconds"),
        fps=data.get("fps"),
        width=data.get("width"),
        height=data.get("height"),
        created_at=str(data.get("created_at", "")),
        updated_at=str(data.get("updated_at", "")),
    )

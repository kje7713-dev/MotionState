"""SQLAlchemy ORM models for the motion-state pipeline."""

import enum
from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Enum, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class VideoStatus(enum.StrEnum):
    pending = "pending"
    processing = "processing"
    ready = "ready"
    error = "error"


class JobStatus(enum.StrEnum):
    queued = "queued"
    running = "running"
    done = "done"
    error = "error"


class JobType(enum.StrEnum):
    process_video = "process_video"


class RunStatus(enum.StrEnum):
    pending = "pending"
    running = "running"
    completed = "completed"
    error = "error"


class TriggerType(enum.StrEnum):
    initial = "initial"
    reprocess = "reprocess"


class Project(Base):
    """Ownership boundary for videos, runs, and API keys."""

    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    api_keys: Mapped[list["ApiKey"]] = relationship("ApiKey", back_populates="project")
    videos: Mapped[list["Video"]] = relationship("Video", back_populates="project")


class ApiKey(Base):
    """An API key belonging to a project.

    The raw secret is never stored; only a salted PBKDF2-HMAC-SHA256-derived value
    (typically hex-encoded) is kept. The full key is returned exactly once at
    creation time.
    """

    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(Integer, ForeignKey("projects.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    key_prefix: Mapped[str] = mapped_column(String(32), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    project: Mapped["Project"] = relationship("Project", back_populates="api_keys")


class Video(Base):
    """Represents an ingested video file and its processing state."""

    __tablename__ = "videos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("projects.id"), nullable=True, index=True
    )
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    status: Mapped[VideoStatus] = mapped_column(
        Enum(VideoStatus), default=VideoStatus.pending, nullable=False
    )
    source_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    normalized_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    project: Mapped["Project | None"] = relationship("Project", back_populates="videos")
    jobs: Mapped[list["Job"]] = relationship("Job", back_populates="video")
    artifacts: Mapped[list["Artifact"]] = relationship("Artifact", back_populates="video")
    processing_runs: Mapped[list["ProcessingRun"]] = relationship(
        "ProcessingRun", back_populates="video"
    )


class ProcessingRun(Base):
    """Represents a single processing attempt for a video.

    Each time a video is processed (or reprocessed) a new row is created so
    that artifact lineage is preserved across runs.  The ``status`` field
    tracks the lifecycle of the run; ``trigger_type`` records whether this was
    the initial ingest or an explicit reprocess request.
    """

    __tablename__ = "processing_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[int] = mapped_column(Integer, ForeignKey("videos.id"), nullable=False)
    status: Mapped[RunStatus] = mapped_column(
        Enum(RunStatus), default=RunStatus.pending, nullable=False
    )
    trigger_type: Mapped[TriggerType] = mapped_column(
        Enum(TriggerType), default=TriggerType.initial, nullable=False
    )
    pipeline_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    detector_backend: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tracker_backend: Mapped[str | None] = mapped_column(String(64), nullable=True)
    pose_backend: Mapped[str | None] = mapped_column(String(64), nullable=True)
    storage_backend: Mapped[str | None] = mapped_column(String(64), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    video: Mapped["Video"] = relationship("Video", back_populates="processing_runs")
    jobs: Mapped[list["Job"]] = relationship("Job", back_populates="processing_run")
    artifacts: Mapped[list["Artifact"]] = relationship(
        "Artifact", back_populates="processing_run"
    )


class Job(Base):
    """Represents a processing job associated with a video."""

    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[int] = mapped_column(Integer, ForeignKey("videos.id"), nullable=False)
    processing_run_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("processing_runs.id"), nullable=True
    )
    type: Mapped[JobType] = mapped_column(
        Enum(JobType), default=JobType.process_video, nullable=False
    )
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus), default=JobStatus.queued, nullable=False
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    video: Mapped["Video"] = relationship("Video", back_populates="jobs")
    processing_run: Mapped["ProcessingRun | None"] = relationship(
        "ProcessingRun", back_populates="jobs"
    )


class Artifact(Base):
    """Represents a structured output artifact produced for a video."""

    __tablename__ = "artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[int] = mapped_column(Integer, ForeignKey("videos.id"), nullable=False)
    processing_run_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("processing_runs.id"), nullable=True
    )
    type: Mapped[str] = mapped_column(String(128), nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    video: Mapped["Video"] = relationship("Video", back_populates="artifacts")
    processing_run: Mapped["ProcessingRun | None"] = relationship(
        "ProcessingRun", back_populates="artifacts"
    )

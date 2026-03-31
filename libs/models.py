"""SQLAlchemy ORM models for the motion-state pipeline."""

import enum
from datetime import datetime

from sqlalchemy import JSON, DateTime, Enum, Float, ForeignKey, Integer, String, Text, func
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


class Video(Base):
    """Represents an ingested video file and its processing state."""

    __tablename__ = "videos"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
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

    jobs: Mapped[list["Job"]] = relationship("Job", back_populates="video")
    artifacts: Mapped[list["Artifact"]] = relationship("Artifact", back_populates="video")


class Job(Base):
    """Represents a processing job associated with a video."""

    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[int] = mapped_column(Integer, ForeignKey("videos.id"), nullable=False)
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


class Artifact(Base):
    """Represents a structured output artifact produced for a video."""

    __tablename__ = "artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[int] = mapped_column(Integer, ForeignKey("videos.id"), nullable=False)
    type: Mapped[str] = mapped_column(String(128), nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    video: Mapped["Video"] = relationship("Video", back_populates="artifacts")

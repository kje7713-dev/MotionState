"""Typed contracts (abstract interfaces) for every CV pipeline stage.

Each interface is defined as an abstract base class with typed inputs/outputs.
Concrete implementations (real detectors, trackers, pose estimators, …) must
subclass the appropriate contract.  Stub implementations live alongside.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Shared primitive types
# ---------------------------------------------------------------------------


@dataclass
class Frame:
    """A single decoded video frame."""

    index: int  # zero-based frame number
    timestamp_ms: float  # position in the video in milliseconds
    data: bytes = field(repr=False)  # raw image bytes (e.g. BGR or RGB)


@dataclass
class BoundingBox:
    """Axis-aligned bounding box in pixel coordinates."""

    x: float
    y: float
    width: float
    height: float
    confidence: float = 1.0


@dataclass
class Detection:
    """A single object detection result."""

    frame_index: int
    bbox: BoundingBox
    class_id: int = 0
    class_label: str = "person"
    timestamp_ms: float = 0.0


@dataclass
class Track:
    """A single tracked person across frames."""

    track_id: int
    detections: list[Detection] = field(default_factory=list)


@dataclass
class Keypoint:
    """A single body keypoint in pixel coordinates."""

    name: str
    x: float
    y: float
    confidence: float = 1.0


@dataclass
class PoseEstimate:
    """Pose keypoints for a single person in a single frame."""

    frame_index: int
    track_id: int
    keypoints: list[Keypoint] = field(default_factory=list)
    timestamp_ms: float = 0.0


@dataclass
class MotionFeature:
    """A derived scalar or vector feature over a time window."""

    track_id: int
    name: str
    start_ms: float
    end_ms: float
    value: float | list[float] = 0.0


@dataclass
class Segment:
    """A time-indexed segment of meaningful motion/state."""

    start_ms: float
    end_ms: float
    label: str
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract interfaces
# ---------------------------------------------------------------------------


class Detector(ABC):
    """Detects persons (or other objects) in individual frames."""

    @abstractmethod
    def detect(self, frame: Frame) -> list[Detection]:
        """Return detections for *frame*."""
        ...


class Tracker(ABC):
    """Assigns consistent IDs to detections across frames."""

    @abstractmethod
    def update(self, detections: list[Detection]) -> list[Track]:
        """Consume new *detections* and return updated track list."""
        ...


class PoseEstimator(ABC):
    """Estimates 2-D body keypoints for each tracked person."""

    @abstractmethod
    def estimate(self, frame: Frame, tracks: list[Track]) -> list[PoseEstimate]:
        """Return pose estimates for all tracks visible in *frame*."""
        ...


class FeatureDeriver(ABC):
    """Derives scalar motion/state features from tracks and pose estimates."""

    @abstractmethod
    def derive(
        self,
        tracks: list[Track],
        poses: list[PoseEstimate],
    ) -> list[MotionFeature]:
        """Return derived features for the given tracks and poses."""
        ...


class Segmenter(ABC):
    """Segments a feature time-series into meaningful temporal intervals."""

    @abstractmethod
    def segment(self, features: list[MotionFeature]) -> list[Segment]:
        """Return a list of temporal segments derived from *features*."""
        ...

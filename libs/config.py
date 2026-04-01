"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All tuneable settings; values read from environment / .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database
    database_url: str = "postgresql+asyncpg://motionstate:motionstate@localhost:5432/motionstate"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Storage directories
    upload_dir: str = "./data/uploads"
    normalized_dir: str = "./data/normalized"
    artifacts_dir: str = "./data/artifacts"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False

    # Frame extraction
    frame_sample_fps: float = 2.0

    # Detector
    # Use "stub" for no-op detection (default) or "yolo" to enable YOLOv8.
    detector_backend: str = "stub"
    detector_model: str = "yolov8n.pt"

    # Tracker
    # Use "stub" for no-op (default) or "iou" for deterministic IOU-based tracking.
    tracker_backend: str = "stub"
    tracker_iou_threshold: float = 0.3
    tracker_max_age: int = 30

    # Pose estimator
    # Use "stub" for no-op (default) or "mediapipe" for MediaPipe BlazePose.
    pose_backend: str = "stub"
    pose_model: str = ""
    pose_min_confidence: float = 0.3


settings = Settings()

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

    # Storage backend
    # Use "local" (default, writes to local filesystem) or "s3" (AWS S3 / Cloudflare R2).
    storage_backend: str = "local"

    # S3 / R2 config (only used when storage_backend=s3)
    s3_bucket: str = ""
    s3_region: str = ""
    s3_endpoint_url: str = ""  # set to R2 endpoint for Cloudflare R2
    s3_access_key_id: str = ""
    s3_secret_access_key: str = ""
    signed_url_expiration_seconds: int = 3600

    # API key hashing
    # Server-side HMAC secret used to hash API keys before storing.
    # Set this to a strong random value in production (e.g. openssl rand -hex 32).
    api_key_hmac_secret: str = "change-me-in-production"


settings = Settings()

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


settings = Settings()

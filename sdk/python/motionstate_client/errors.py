"""SDK exception hierarchy for MotionState client errors."""

from __future__ import annotations


class MotionStateError(Exception):
    """Base class for all MotionState SDK errors."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class AuthError(MotionStateError):
    """Raised when the API key is missing, invalid, or revoked (HTTP 401/403)."""


class QuotaError(MotionStateError):
    """Raised when a project quota or rate limit is exceeded (HTTP 429)."""


class NotFoundError(MotionStateError):
    """Raised when a requested resource does not exist (HTTP 404)."""


class ServerError(MotionStateError):
    """Raised when the server returns an unexpected 5xx response."""


class PollingTimeout(MotionStateError):
    """Raised when ``wait_for_run_completion`` exceeds the timeout."""

    def __init__(self, video_id: int, run_id: int, timeout: float) -> None:
        super().__init__(
            f"Run {run_id} for video {video_id} did not complete within {timeout}s"
        )
        self.video_id = video_id
        self.run_id = run_id
        self.timeout = timeout

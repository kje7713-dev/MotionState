"""MotionState Python SDK client."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import httpx

from .errors import (
    AuthError,
    MotionStateError,
    NotFoundError,
    PollingTimeout,
    QuotaError,
    ServerError,
)
from .models import (
    ArtifactRecord,
    ProcessingRun,
    ReprocessResponse,
    UploadInitResponse,
    UploadResponse,
    VideoResponse,
    _parse_artifact,
    _parse_run,
    _parse_video,
)

# Terminal run statuses – polling stops when any of these is reached.
_TERMINAL_STATUSES = {"completed", "failed", "error"}


class MotionStateClient:
    """Synchronous HTTP client for the MotionState Pipeline API.

    Args:
        base_url: Root URL of the MotionState API, e.g. ``"http://localhost:8000"``.
        api_key: Project API key for authentication (sent as ``X-API-Key`` header).
        timeout: Per-request timeout in seconds (default 30).

    Example::

        client = MotionStateClient(base_url="http://localhost:8000", api_key="ms_...")
        resp = client.upload_video("my_video.mp4")
        run = client.wait_for_run_completion(resp.video_id, resp.processing_run_id)
        outputs = client.fetch_latest_outputs(resp.video_id)
    """

    def __init__(self, base_url: str, api_key: str, *, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._http = httpx.Client(
            base_url=self._base_url,
            headers={"X-API-Key": api_key},
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> MotionStateClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._http.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        """Execute an HTTP request and raise an SDK error on non-2xx status."""
        response = self._http.request(method, path, **kwargs)
        return self._raise_for_status(response)

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> Any:
        """Map HTTP errors to SDK exceptions and return parsed JSON on success."""
        if response.is_success:
            if response.status_code == 204:
                return None
            return response.json()
        status = response.status_code
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        if status in (401, 403):
            raise AuthError(str(detail), status_code=status)
        if status == 404:
            raise NotFoundError(str(detail), status_code=status)
        if status == 429:
            raise QuotaError(str(detail), status_code=status)
        if status >= 500:
            raise ServerError(str(detail), status_code=status)
        raise MotionStateError(str(detail), status_code=status)

    # ------------------------------------------------------------------
    # Video upload
    # ------------------------------------------------------------------

    def upload_video(self, file_path: str | Path) -> UploadResponse:
        """Upload a video file to the API and enqueue a processing job.

        This is the simple multipart-upload path.  For large files or
        production deployments where direct-to-storage uploads are preferred,
        use :meth:`upload_init` instead.

        Args:
            file_path: Local path to the video file.

        Returns:
            :class:`~motionstate_client.models.UploadResponse` with
            ``video_id``, ``job_id``, and ``processing_run_id``.
        """
        path = Path(file_path)
        with path.open("rb") as fh:
            data = self._request(
                "POST",
                "/videos",
                files={"file": (path.name, fh, "video/mp4")},
            )
        return UploadResponse(
            video_id=data["video_id"],
            job_id=data["job_id"],
            processing_run_id=data["processing_run_id"],
        )

    def upload_init(self, filename: str) -> UploadInitResponse:
        """Initialise a direct-to-storage upload and return a pre-signed URL.

        Call this to get an ``upload_url`` the client can PUT the video to
        directly (S3 / R2).  When the storage backend is *local*, ``upload_url``
        is ``None`` and you should use :meth:`upload_video` instead.

        Args:
            filename: Original filename including extension.

        Returns:
            :class:`~motionstate_client.models.UploadInitResponse` with
            ``video_id``, ``upload_url``, and ``storage_key``.
        """
        data = self._request("POST", "/videos/upload-init", json={"filename": filename})
        return UploadInitResponse(
            video_id=data["video_id"],
            upload_url=data.get("upload_url"),
            storage_key=data["storage_key"],
        )

    # ------------------------------------------------------------------
    # Video reads
    # ------------------------------------------------------------------

    def get_video(self, video_id: int) -> VideoResponse:
        """Return metadata for a single video.

        Args:
            video_id: Target video ID.

        Returns:
            :class:`~motionstate_client.models.VideoResponse`.

        Raises:
            :class:`~motionstate_client.errors.NotFoundError`: if the video
                does not exist or belongs to a different project.
        """
        data = self._request("GET", f"/videos/{video_id}")
        return _parse_video(data)

    def list_artifacts(self, video_id: int) -> list[ArtifactRecord]:
        """Return all artifact records for a video.

        Args:
            video_id: Target video ID.

        Returns:
            List of :class:`~motionstate_client.models.ArtifactRecord`.
        """
        data = self._request("GET", f"/videos/{video_id}/artifacts")
        return [_parse_artifact(a) for a in data]

    # ------------------------------------------------------------------
    # Artifact reads
    # ------------------------------------------------------------------

    def get_state(self, video_id: int, *, run_id: int | None = None) -> dict[str, Any]:
        """Return the parsed ``state.json`` artifact.

        Args:
            video_id: Target video ID.
            run_id: Pin to a specific run.  Defaults to the latest.
        """
        params = {"run_id": run_id} if run_id is not None else {}
        return self._request("GET", f"/videos/{video_id}/state", params=params)

    def get_detections(self, video_id: int, *, run_id: int | None = None) -> dict[str, Any]:
        """Return the parsed ``detections.json`` artifact."""
        params = {"run_id": run_id} if run_id is not None else {}
        return self._request("GET", f"/videos/{video_id}/detections", params=params)

    def get_tracks(self, video_id: int, *, run_id: int | None = None) -> dict[str, Any]:
        """Return the parsed ``tracks.json`` artifact."""
        params = {"run_id": run_id} if run_id is not None else {}
        return self._request("GET", f"/videos/{video_id}/tracks", params=params)

    def get_poses(self, video_id: int, *, run_id: int | None = None) -> dict[str, Any]:
        """Return the parsed ``poses.json`` artifact."""
        params = {"run_id": run_id} if run_id is not None else {}
        return self._request("GET", f"/videos/{video_id}/poses", params=params)

    def get_features(self, video_id: int, *, run_id: int | None = None) -> dict[str, Any]:
        """Return the parsed ``features.json`` artifact."""
        params = {"run_id": run_id} if run_id is not None else {}
        return self._request("GET", f"/videos/{video_id}/features", params=params)

    def get_segments(self, video_id: int, *, run_id: int | None = None) -> dict[str, Any]:
        """Return the parsed ``segments.json`` artifact."""
        params = {"run_id": run_id} if run_id is not None else {}
        return self._request("GET", f"/videos/{video_id}/segments", params=params)

    def get_timeline(self, video_id: int, *, run_id: int | None = None) -> dict[str, Any]:
        """Return the parsed ``timeline_manifest.json`` artifact."""
        params = {"run_id": run_id} if run_id is not None else {}
        return self._request("GET", f"/videos/{video_id}/timeline", params=params)

    # ------------------------------------------------------------------
    # Processing runs
    # ------------------------------------------------------------------

    def list_runs(self, video_id: int) -> list[ProcessingRun]:
        """Return all processing runs for a video, newest first.

        Args:
            video_id: Target video ID.

        Returns:
            List of :class:`~motionstate_client.models.ProcessingRun`.
        """
        data = self._request("GET", f"/videos/{video_id}/runs")
        return [_parse_run(r) for r in data]

    def reprocess_video(self, video_id: int) -> ReprocessResponse:
        """Enqueue a new reprocessing job for a video.

        Args:
            video_id: Target video ID.

        Returns:
            :class:`~motionstate_client.models.ReprocessResponse`.
        """
        data = self._request("POST", f"/videos/{video_id}/reprocess")
        return ReprocessResponse(
            video_id=data["video_id"],
            processing_run_id=data["processing_run_id"],
            job_id=data["job_id"],
        )

    # ------------------------------------------------------------------
    # Usage / quotas
    # ------------------------------------------------------------------

    def get_project_usage(self, project_id: int) -> dict[str, Any]:
        """Return the full usage summary for a project.

        Args:
            project_id: Target project ID.

        Returns:
            A dict with ``current_month``, ``alltime``, and ``storage_bytes``
            summary fields (structure matches the API response from
            ``GET /projects/{id}/usage``).
        """
        return self._request("GET", f"/projects/{project_id}/usage")

    # ------------------------------------------------------------------
    # High-level convenience helpers
    # ------------------------------------------------------------------

    def submit_video(self, file_path: str | Path) -> UploadResponse:
        """Alias for :meth:`upload_video`.

        Provided as a semantically clearer entry-point for the common
        *"I just want to submit a video"* workflow.

        Args:
            file_path: Local path to the video file.
        """
        return self.upload_video(file_path)

    def wait_for_run_completion(
        self,
        video_id: int,
        run_id: int,
        *,
        timeout: float = 300.0,
        poll_interval: float = 5.0,
    ) -> ProcessingRun:
        """Poll until the given run reaches a terminal status.

        Args:
            video_id: Video the run belongs to.
            run_id: Run to wait for.
            timeout: Maximum seconds to wait before raising
                :class:`~motionstate_client.errors.PollingTimeout`.
            poll_interval: Seconds between each status poll.

        Returns:
            The :class:`~motionstate_client.models.ProcessingRun` once it
            reaches a terminal state (``completed`` or ``failed``).

        Raises:
            :class:`~motionstate_client.errors.PollingTimeout`: if *timeout*
                seconds elapse before the run completes.
        """
        deadline = time.monotonic() + timeout
        while True:
            runs = self.list_runs(video_id)
            for run in runs:
                if run.id == run_id:
                    if run.status in _TERMINAL_STATUSES:
                        return run
                    break
            if time.monotonic() >= deadline:
                raise PollingTimeout(video_id, run_id, timeout)
            time.sleep(poll_interval)

    def fetch_latest_outputs(
        self, video_id: int, *, run_id: int | None = None
    ) -> dict[str, Any]:
        """Fetch all available pipeline outputs for a video in one call.

        Attempts to retrieve every artifact type.  Artifacts that are not yet
        available (404) are silently omitted from the result dict so callers
        do not need to handle partial pipeline runs.

        Args:
            video_id: Target video ID.
            run_id: Pin all reads to a specific run.  Defaults to latest.

        Returns:
            A dict with keys ``state``, ``detections``, ``tracks``, ``poses``,
            ``features``, ``segments``, ``timeline`` for each artifact that
            was successfully fetched.
        """
        outputs: dict[str, Any] = {}
        fetchers = {
            "state": self.get_state,
            "detections": self.get_detections,
            "tracks": self.get_tracks,
            "poses": self.get_poses,
            "features": self.get_features,
            "segments": self.get_segments,
            "timeline": self.get_timeline,
        }
        for key, fetcher in fetchers.items():
            try:
                outputs[key] = fetcher(video_id, run_id=run_id)
            except NotFoundError:
                pass
        return outputs

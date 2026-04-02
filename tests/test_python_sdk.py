"""Tests for MotionState Python SDK – request construction and response parsing."""

from __future__ import annotations

import json
import os

# ---------------------------------------------------------------------------
# Path setup: allow importing sdk without installing the package
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "sdk", "python"),
)

from motionstate_client import (
    ArtifactRecord,
    MotionStateClient,
    ProcessingRun,
    ReprocessResponse,
    UploadInitResponse,
    UploadResponse,
    VideoResponse,
)
from motionstate_client.errors import (
    PollingTimeout,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(status_code: int, body: object) -> httpx.Response:
    """Build a minimal httpx.Response for mocking."""
    content = json.dumps(body).encode()
    return httpx.Response(
        status_code=status_code,
        headers={"content-type": "application/json"},
        content=content,
    )


def _make_client(transport: httpx.MockTransport | None = None) -> MotionStateClient:
    client = MotionStateClient(base_url="http://test", api_key="ms_test_key")
    if transport is not None:
        client._http = httpx.Client(
            base_url="http://test",
            headers={"X-API-Key": "ms_test_key"},
            transport=transport,
        )
    return client


# ---------------------------------------------------------------------------
# Auth header
# ---------------------------------------------------------------------------


def test_auth_header_is_set():
    """The X-API-Key header must be present on every request."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _make_response(200, {"id": 1, "original_filename": "v.mp4", "status": "done",
                                    "source_path": None, "normalized_path": None,
                                    "duration_seconds": None, "fps": None,
                                    "width": None, "height": None,
                                    "created_at": "2024-01-01", "updated_at": "2024-01-01"})

    client = _make_client(httpx.MockTransport(handler))
    client.get_video(1)
    assert requests[0].headers["x-api-key"] == "ms_test_key"


# ---------------------------------------------------------------------------
# upload_video
# ---------------------------------------------------------------------------


def test_upload_video(tmp_path: Path):
    video = tmp_path / "test.mp4"
    video.write_bytes(b"fake video bytes")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/videos"
        return _make_response(201, {"video_id": 7, "job_id": 3, "processing_run_id": 5})

    client = _make_client(httpx.MockTransport(handler))
    resp = client.upload_video(video)
    assert isinstance(resp, UploadResponse)
    assert resp.video_id == 7
    assert resp.job_id == 3
    assert resp.processing_run_id == 5


def test_submit_video_is_alias_for_upload_video(tmp_path: Path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"data")

    def handler(_: httpx.Request) -> httpx.Response:
        return _make_response(201, {"video_id": 1, "job_id": 2, "processing_run_id": 3})

    client = _make_client(httpx.MockTransport(handler))
    resp = client.submit_video(video)
    assert resp.video_id == 1


# ---------------------------------------------------------------------------
# upload_init
# ---------------------------------------------------------------------------


def test_upload_init():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/videos/upload-init"
        body = json.loads(request.content)
        assert body["filename"] == "clip.mp4"
        return _make_response(201, {
            "video_id": 10,
            "upload_url": "https://s3.example.com/signed",
            "storage_key": "artifacts/10/source.mp4",
        })

    client = _make_client(httpx.MockTransport(handler))
    resp = client.upload_init("clip.mp4")
    assert isinstance(resp, UploadInitResponse)
    assert resp.video_id == 10
    assert resp.upload_url == "https://s3.example.com/signed"
    assert resp.storage_key == "artifacts/10/source.mp4"


def test_upload_init_local_backend_returns_no_url():
    def handler(_: httpx.Request) -> httpx.Response:
        return _make_response(201, {
            "video_id": 2,
            "upload_url": None,
            "storage_key": "artifacts/2/source.mp4",
        })

    client = _make_client(httpx.MockTransport(handler))
    resp = client.upload_init("video.mp4")
    assert resp.upload_url is None


# ---------------------------------------------------------------------------
# get_video
# ---------------------------------------------------------------------------


def test_get_video():
    payload = {
        "id": 5, "original_filename": "run.mp4", "status": "done",
        "source_path": "/tmp/x.mp4", "normalized_path": "/tmp/xn.mp4",
        "duration_seconds": 12.5, "fps": 30.0, "width": 1920, "height": 1080,
        "created_at": "2024-01-01T00:00:00", "updated_at": "2024-01-01T01:00:00",
    }

    def handler(_: httpx.Request) -> httpx.Response:
        return _make_response(200, payload)

    client = _make_client(httpx.MockTransport(handler))
    video = client.get_video(5)
    assert isinstance(video, VideoResponse)
    assert video.id == 5
    assert video.fps == 30.0
    assert video.width == 1920


# ---------------------------------------------------------------------------
# list_runs
# ---------------------------------------------------------------------------


def test_list_runs():
    payload = [
        {"id": 1, "status": "completed", "trigger_type": "initial",
         "pipeline_version": 2, "created_at": "2024-01-01", "completed_at": "2024-01-02",
         "error": None},
        {"id": 2, "status": "pending", "trigger_type": "reprocess",
         "pipeline_version": None, "created_at": "2024-01-03", "completed_at": None,
         "error": None},
    ]

    def handler(_: httpx.Request) -> httpx.Response:
        return _make_response(200, payload)

    client = _make_client(httpx.MockTransport(handler))
    runs = client.list_runs(1)
    assert len(runs) == 2
    assert isinstance(runs[0], ProcessingRun)
    assert runs[0].status == "completed"
    assert runs[1].completed_at is None


# ---------------------------------------------------------------------------
# reprocess_video
# ---------------------------------------------------------------------------


def test_reprocess_video():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert "/reprocess" in request.url.path
        return _make_response(201, {"video_id": 3, "processing_run_id": 9, "job_id": 11})

    client = _make_client(httpx.MockTransport(handler))
    resp = client.reprocess_video(3)
    assert isinstance(resp, ReprocessResponse)
    assert resp.processing_run_id == 9


# ---------------------------------------------------------------------------
# list_artifacts
# ---------------------------------------------------------------------------


def test_list_artifacts():
    payload = [
        {"id": 1, "video_id": 5, "type": "state", "path": "/a/state.json",
         "metadata_json": {"version": 7}, "created_at": "2024-01-01"},
    ]

    def handler(_: httpx.Request) -> httpx.Response:
        return _make_response(200, payload)

    client = _make_client(httpx.MockTransport(handler))
    arts = client.list_artifacts(5)
    assert len(arts) == 1
    assert isinstance(arts[0], ArtifactRecord)
    assert arts[0].type == "state"


# ---------------------------------------------------------------------------
# Artifact reads – run_id query param forwarding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method,path_suffix", [
    ("get_state", "state"),
    ("get_detections", "detections"),
    ("get_tracks", "tracks"),
    ("get_poses", "poses"),
    ("get_features", "features"),
    ("get_segments", "segments"),
    ("get_timeline", "timeline"),
])
def test_artifact_read_passes_run_id(method: str, path_suffix: str):
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return _make_response(200, {"video_id": "1", "version": 1})

    client = _make_client(httpx.MockTransport(handler))
    getattr(client, method)(1, run_id=42)
    assert f"/{path_suffix}" in captured[0].url.path
    assert "run_id=42" in str(captured[0].url)


def test_artifact_read_no_run_id_omits_param():
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return _make_response(200, {"video_id": "1", "version": 1})

    client = _make_client(httpx.MockTransport(handler))
    client.get_state(1)
    assert "run_id" not in str(captured[0].url)


# ---------------------------------------------------------------------------
# get_project_usage
# ---------------------------------------------------------------------------


def test_get_project_usage():
    payload = {"current_month": {"videos_uploaded": 3}, "alltime": {}, "storage_bytes": 1024}

    def handler(_: httpx.Request) -> httpx.Response:
        return _make_response(200, payload)

    client = _make_client(httpx.MockTransport(handler))
    usage = client.get_project_usage(1)
    assert usage["current_month"]["videos_uploaded"] == 3


# ---------------------------------------------------------------------------
# wait_for_run_completion
# ---------------------------------------------------------------------------


def test_wait_for_run_completion_returns_when_completed():
    call_count = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        status = "pending" if call_count < 3 else "completed"
        return _make_response(200, [
            {"id": 5, "status": status, "trigger_type": "initial",
             "pipeline_version": 1, "created_at": "2024-01-01",
             "completed_at": "2024-01-02" if status == "completed" else None,
             "error": None},
        ])

    client = _make_client(httpx.MockTransport(handler))
    run = client.wait_for_run_completion(1, 5, timeout=60, poll_interval=0)
    assert run.status == "completed"
    assert call_count == 3


def test_wait_for_run_completion_raises_on_timeout():
    def handler(_: httpx.Request) -> httpx.Response:
        return _make_response(200, [
            {"id": 5, "status": "running", "trigger_type": "initial",
             "pipeline_version": None, "created_at": "x", "completed_at": None, "error": None},
        ])

    client = _make_client(httpx.MockTransport(handler))
    with pytest.raises(PollingTimeout):
        client.wait_for_run_completion(1, 5, timeout=0, poll_interval=0)


# ---------------------------------------------------------------------------
# fetch_latest_outputs
# ---------------------------------------------------------------------------


def test_fetch_latest_outputs_skips_not_found():
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if "state" in request.url.path:
            return _make_response(200, {"video_id": "1", "version": 7})
        return _make_response(404, {"detail": "Not found"})

    client = _make_client(httpx.MockTransport(handler))
    outputs = client.fetch_latest_outputs(1)
    assert "state" in outputs
    # All other artifacts returned 404 and should be omitted.
    assert "detections" not in outputs


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


def test_client_context_manager(tmp_path: Path):
    with MotionStateClient(base_url="http://test", api_key="key") as client:
        assert client._http is not None
    # After exiting the context manager the underlying client should be closed.
    assert client._http.is_closed

"""Unit tests for the S3-compatible storage backend.

All S3 calls are mocked via unittest.mock – no real AWS / R2 credentials
are required to run these tests.  boto3 itself is also mocked at the module
level so these tests run even when the ``[storage]`` optional dependencies are
not installed (e.g. in the base dev CI install).
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure boto3 and botocore are importable even when the [storage] extras are
# not installed.  The mocks are placed in sys.modules *before* any import that
# would trigger them.
# ---------------------------------------------------------------------------
if "boto3" not in sys.modules:
    sys.modules["boto3"] = MagicMock()
if "botocore" not in sys.modules:
    _botocore_mock = MagicMock()
    sys.modules["botocore"] = _botocore_mock
    sys.modules["botocore.config"] = _botocore_mock.config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_s3_storage(bucket: str = "test-bucket"):
    """Return an S3Storage instance with a mocked boto3 client."""
    mock_client = MagicMock()
    with patch("boto3.client", return_value=mock_client):
        from libs.storage.s3 import S3Storage

        storage = S3Storage(
            bucket=bucket,
            region="us-east-1",
            endpoint_url="",
            access_key_id="test-key",
            secret_access_key="test-secret",
        )
    # Replace the internal client with our mock so we can inspect calls.
    storage._client = mock_client
    return storage, mock_client


# ---------------------------------------------------------------------------
# save / put_object
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_s3_save_calls_put_object():
    """save() calls put_object with the correct bucket, key and body."""
    storage, mock_client = _make_s3_storage()
    data = b"artifact content"
    key = "artifacts/1/state.json"

    returned_key = await storage.save(data, key)

    mock_client.put_object.assert_called_once_with(
        Bucket="test-bucket",
        Key=key,
        Body=data,
    )
    assert returned_key == key


@pytest.mark.asyncio
async def test_s3_save_returns_key():
    """save() returns the canonical S3 key (not a local path)."""
    storage, _ = _make_s3_storage()
    key = "artifacts/42/detections.json"
    returned = await storage.save(b"data", key)
    assert returned == key


# ---------------------------------------------------------------------------
# load / get_object
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_s3_load_calls_get_object_and_returns_bytes():
    """load() calls get_object and returns the response body bytes."""
    storage, mock_client = _make_s3_storage()
    expected_bytes = b"loaded content"

    # boto3 returns a streaming body; simulate with BytesIO.
    mock_body = MagicMock()
    mock_body.read.return_value = expected_bytes
    mock_client.get_object.return_value = {"Body": mock_body}

    result = await storage.load("artifacts/1/state.json")

    mock_client.get_object.assert_called_once_with(
        Bucket="test-bucket",
        Key="artifacts/1/state.json",
    )
    assert result == expected_bytes


# ---------------------------------------------------------------------------
# exists / head_object
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_s3_exists_true_when_head_object_succeeds():
    """exists() returns True when head_object does not raise."""
    storage, mock_client = _make_s3_storage()
    mock_client.head_object.return_value = {"ContentLength": 123}

    assert await storage.exists("artifacts/1/state.json") is True
    mock_client.head_object.assert_called_once_with(
        Bucket="test-bucket", Key="artifacts/1/state.json"
    )


@pytest.mark.asyncio
async def test_s3_exists_false_when_head_object_raises_404():
    """exists() returns False when head_object raises a 404 ClientError."""
    storage, mock_client = _make_s3_storage()
    exc = Exception("NoSuchKey")
    exc.response = {"Error": {"Code": "404"}}
    mock_client.head_object.side_effect = exc

    assert await storage.exists("artifacts/1/missing.json") is False


@pytest.mark.asyncio
async def test_s3_exists_reraises_non_404_errors():
    """exists() re-raises exceptions that are not 404 / NoSuchKey."""
    storage, mock_client = _make_s3_storage()
    exc = Exception("AccessDenied")
    exc.response = {"Error": {"Code": "403"}}
    mock_client.head_object.side_effect = exc

    with pytest.raises(Exception, match="AccessDenied"):
        await storage.exists("artifacts/1/state.json")


# ---------------------------------------------------------------------------
# full_path – should raise NotImplementedError
# ---------------------------------------------------------------------------


def test_s3_full_path_raises():
    """full_path() raises NotImplementedError for the S3 backend."""
    storage, _ = _make_s3_storage()
    with pytest.raises(NotImplementedError):
        storage.full_path("artifacts/1/state.json")


# ---------------------------------------------------------------------------
# generate_upload_url (pre-signed PUT)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_s3_generate_upload_url_calls_presigned():
    """generate_upload_url() calls generate_presigned_url with put_object."""
    storage, mock_client = _make_s3_storage()
    expected_url = "https://bucket.s3.example.com/videos/1/source.mp4?X-Amz-Signature=abc"
    mock_client.generate_presigned_url.return_value = expected_url

    url = await storage.generate_upload_url("videos/1/source.mp4", expires_in=3600)

    mock_client.generate_presigned_url.assert_called_once_with(
        "put_object",
        Params={"Bucket": "test-bucket", "Key": "videos/1/source.mp4"},
        ExpiresIn=3600,
    )
    assert url == expected_url


@pytest.mark.asyncio
async def test_s3_generate_upload_url_returns_string():
    """generate_upload_url() returns a non-empty string URL."""
    storage, mock_client = _make_s3_storage()
    mock_client.generate_presigned_url.return_value = "https://example.com/upload"

    url = await storage.generate_upload_url("videos/1/source.mp4", expires_in=300)
    assert isinstance(url, str)
    assert url


# ---------------------------------------------------------------------------
# generate_download_url (pre-signed GET)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_s3_generate_download_url_calls_presigned():
    """generate_download_url() calls generate_presigned_url with get_object."""
    storage, mock_client = _make_s3_storage()
    mock_client.generate_presigned_url.return_value = "https://example.com/dl"

    url = await storage.generate_download_url("artifacts/1/state.json", expires_in=600)

    mock_client.generate_presigned_url.assert_called_once_with(
        "get_object",
        Params={"Bucket": "test-bucket", "Key": "artifacts/1/state.json"},
        ExpiresIn=600,
    )
    assert url == "https://example.com/dl"


# ---------------------------------------------------------------------------
# canonical key helpers
# ---------------------------------------------------------------------------


def test_storage_source_video_key():
    """source_video_key() returns the expected canonical key."""
    from libs.storage import source_video_key

    assert source_video_key(1) == "videos/1/source.mp4"
    assert source_video_key(42, ext=".mov") == "videos/42/source.mov"
    assert source_video_key(7, ext="webm") == "videos/7/source.webm"


def test_storage_normalized_video_key():
    """normalized_video_key() returns the expected canonical key."""
    from libs.storage import normalized_video_key

    assert normalized_video_key(1) == "videos/1/normalized.mp4"
    assert normalized_video_key(99) == "videos/99/normalized.mp4"


def test_storage_artifact_key():
    """artifact_key() returns the expected canonical key."""
    from libs.storage import artifact_key

    assert artifact_key(1, "state.json") == "artifacts/1/state.json"
    assert artifact_key(5, "clips/segment_000_low_motion.mp4") == (
        "artifacts/5/clips/segment_000_low_motion.mp4"
    )


# ---------------------------------------------------------------------------
# get_storage factory
# ---------------------------------------------------------------------------


def test_get_storage_returns_local_by_default(tmp_path):
    """get_storage() returns a LocalStorage instance when storage_backend=local."""
    from libs.config import settings
    from libs.storage import get_storage
    from libs.storage.local import LocalStorage

    artifacts_dir = tmp_path / "data" / "artifacts"
    with (
        patch.object(settings, "storage_backend", "local"),
        patch.object(settings, "artifacts_dir", str(artifacts_dir)),
    ):
        backend = get_storage()
    assert isinstance(backend, LocalStorage)


def test_get_storage_returns_s3_when_configured(tmp_path):
    """get_storage() returns an S3Storage instance when storage_backend=s3."""
    from libs.config import settings
    from libs.storage import get_storage
    from libs.storage.s3 import S3Storage

    with (
        patch("boto3.client", return_value=MagicMock()),
        patch.object(settings, "storage_backend", "s3"),
        patch.object(settings, "s3_bucket", "my-bucket"),
        patch.object(settings, "s3_region", "us-east-1"),
        patch.object(settings, "s3_endpoint_url", ""),
        patch.object(settings, "s3_access_key_id", "key"),
        patch.object(settings, "s3_secret_access_key", "secret"),
    ):
        backend = get_storage()
    assert isinstance(backend, S3Storage)

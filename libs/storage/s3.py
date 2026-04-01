"""S3-compatible object storage backend (AWS S3, Cloudflare R2, MinIO, …)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from libs.storage.base import StorageBackend

if TYPE_CHECKING:
    import boto3 as _boto3_type  # noqa: F401 – type-checking only


class S3Storage(StorageBackend):
    """Stores objects in an S3-compatible bucket using ``boto3``.

    Compatible with AWS S3 and Cloudflare R2 (via ``endpoint_url``).

    Args:
        bucket: The S3 bucket name.
        region: AWS region (ignored by R2 but required for boto3 session).
        endpoint_url: Custom endpoint URL for S3-compatible services such as
            Cloudflare R2.  Leave empty to use the default AWS endpoint.
        access_key_id: AWS / R2 access key ID.
        secret_access_key: AWS / R2 secret access key.
    """

    def __init__(
        self,
        bucket: str,
        region: str = "",
        endpoint_url: str = "",
        access_key_id: str = "",
        secret_access_key: str = "",
    ) -> None:
        try:
            import boto3
        except ImportError as exc:
            raise ImportError(
                "boto3 is required for S3Storage. "
                "Install it with: pip install boto3  "
                "or: pip install -e \".[storage]\""
            ) from exc

        session_kwargs: dict = {}
        if access_key_id:
            session_kwargs["aws_access_key_id"] = access_key_id
        if secret_access_key:
            session_kwargs["aws_secret_access_key"] = secret_access_key
        if region:
            session_kwargs["region_name"] = region

        client_kwargs: dict = {}
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url

        self._bucket = bucket
        self._client = boto3.client("s3", **session_kwargs, **client_kwargs)

    # ------------------------------------------------------------------
    # StorageBackend implementation
    # ------------------------------------------------------------------

    async def save(self, data: bytes, relative_path: str) -> str:
        """Upload *data* to the bucket under *relative_path* and return the key.

        The returned value is the canonical S3 object key (``relative_path``),
        which should be stored in the ``Artifact.path`` DB column.
        """
        key = relative_path
        await asyncio.to_thread(
            self._client.put_object,
            Bucket=self._bucket,
            Key=key,
            Body=data,
        )
        return key

    async def load(self, path: str) -> bytes:
        """Download and return the raw bytes for the object at *path* (a key)."""
        response = await asyncio.to_thread(
            self._client.get_object,
            Bucket=self._bucket,
            Key=path,
        )
        body = response["Body"]
        return await asyncio.to_thread(body.read)

    async def exists(self, path: str) -> bool:
        """Return ``True`` if an object with key *path* exists in the bucket."""
        try:
            await asyncio.to_thread(
                self._client.head_object,
                Bucket=self._bucket,
                Key=path,
            )
            return True
        except Exception as exc:
            # botocore.exceptions.ClientError carries an HTTP response dict.
            # A 404 / NoSuchKey response means the object does not exist.
            # Any other error (permissions, network, etc.) is re-raised so it
            # is visible to the caller rather than silently returning False.
            http_response = getattr(exc, "response", None)
            if http_response is not None:
                code = http_response.get("Error", {}).get("Code", "")
                if code in ("404", "NoSuchKey"):
                    return False
            raise

    def full_path(self, path: str) -> Path:
        """Not supported for remote object storage.

        Raises:
            NotImplementedError: Always, because S3 objects have no local path.
        """
        raise NotImplementedError(
            "full_path() is not supported for S3Storage. "
            "Use load() to retrieve object contents instead."
        )

    async def generate_upload_url(self, key: str, expires_in: int) -> str:
        """Return a pre-signed PUT URL for direct client upload to *key*.

        Args:
            key: The S3 object key the client will upload to.
            expires_in: Seconds until the URL expires.

        Returns:
            A pre-signed HTTPS URL the client can PUT the file to directly.
        """
        url: str = await asyncio.to_thread(
            self._client.generate_presigned_url,
            "put_object",
            Params={"Bucket": self._bucket, "Key": key},
            ExpiresIn=expires_in,
        )
        return url

    async def generate_download_url(self, key: str, expires_in: int) -> str:
        """Return a pre-signed GET URL for reading *key*.

        Args:
            key: The S3 object key to generate a download URL for.
            expires_in: Seconds until the URL expires.

        Returns:
            A pre-signed HTTPS URL the client can GET to download the object.
        """
        url: str = await asyncio.to_thread(
            self._client.generate_presigned_url,
            "get_object",
            Params={"Bucket": self._bucket, "Key": key},
            ExpiresIn=expires_in,
        )
        return url

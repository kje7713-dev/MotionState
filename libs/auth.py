"""API key authentication helpers."""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.db import get_db
from libs.models import ApiKey, Project

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Key format: ms_live_<32 hex chars>
_KEY_PREFIX_LEN = 8  # "ms_live_" = 8 chars used as prefix display
_KEY_RANDOM_BYTES = 32
# PBKDF2 iterations for API key hashing.
# API keys are already high-entropy random strings (256 bits), so a low
# iteration count is used to keep auth fast while still using a proper KDF.
_PBKDF2_ITERATIONS = 1


def generate_api_key() -> tuple[str, str, str]:
    """Generate a new API key.

    Returns:
        (raw_key, prefix, key_hash) where raw_key is shown once, prefix is
        the short display prefix, and key_hash is the PBKDF2-SHA256 hex digest
        to store.
    """
    raw_random = secrets.token_hex(_KEY_RANDOM_BYTES)
    raw_key = f"ms_live_{raw_random}"
    prefix = raw_key[:_KEY_PREFIX_LEN + 4]  # "ms_live_" + first 4 random chars
    key_hash = hash_api_key(raw_key)
    return raw_key, prefix, key_hash


def hash_api_key(raw_key: str) -> str:
    """Return a PBKDF2-SHA256 hex digest of a raw API key.

    Uses ``settings.api_key_hmac_secret`` as the salt so that hashes in the
    database cannot be used to verify keys without the server secret.
    API keys contain 32 bytes of randomness so a single PBKDF2 iteration
    provides adequate security while keeping per-request verification fast.
    """
    from libs.config import settings

    return hashlib.pbkdf2_hmac(
        "sha256",
        raw_key.encode(),
        settings.api_key_hmac_secret.encode(),
        _PBKDF2_ITERATIONS,
    ).hex()


async def get_current_project(
    api_key_header: str | None = Security(_API_KEY_HEADER),
    db: AsyncSession = Depends(get_db),
) -> Project:
    """FastAPI dependency that resolves the current project from an API key.

    Accepts the key via the ``X-API-Key`` header.

    Raises:
        HTTPException 401: if the key is missing, invalid, or inactive.
    """
    if not api_key_header:
        raise HTTPException(status_code=401, detail="Missing API key")

    key_hash = hash_api_key(api_key_header)

    result = await db.execute(
        select(ApiKey).where(ApiKey.key_hash == key_hash)
    )
    api_key = result.scalars().first()

    if api_key is None:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not api_key.is_active:
        raise HTTPException(status_code=401, detail="API key is inactive")

    # Update last_used_at in-place; the session auto-commits after the request.
    # get_db() (libs/db.py) calls session.commit() after yielding, so this
    # write is persisted without an explicit extra commit here.
    api_key.last_used_at = datetime.now(UTC)

    project = await db.get(Project, api_key.project_id)
    if project is None:
        raise HTTPException(status_code=401, detail="Project not found")

    return project

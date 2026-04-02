"""Project management routes."""

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.auth import generate_api_key
from libs.db import get_db
from libs.models import ApiKey, Project
from libs.usage import monthly_totals, project_usage_summary

router = APIRouter()


@router.post("", status_code=201)
async def create_project(
    name: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Create a new project (tenancy boundary).

    Args:
        name: Human-readable project name.

    Returns:
        ``{"id": int, "name": str, "created_at": datetime}``
    """
    project = Project(name=name)
    db.add(project)
    await db.flush()

    return {
        "id": project.id,
        "name": project.name,
        "created_at": project.created_at,
    }


@router.get("/{project_id}")
async def get_project(
    project_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return metadata for a project.

    Raises:
        HTTPException 404: if the project does not exist.
    """
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    return {
        "id": project.id,
        "name": project.name,
        "created_at": project.created_at,
        "updated_at": project.updated_at,
    }


@router.post("/{project_id}/api-keys", status_code=201)
async def create_api_key(
    project_id: int,
    name: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Generate a new API key for a project.

    The full raw key is returned **once** in the response and never stored.
    Only the SHA-256 hash is persisted.

    Args:
        project_id: Target project ID.
        name: Human-readable label for this key (e.g. "production", "ci").

    Returns:
        ``{"id": int, "name": str, "key": str, "key_prefix": str, "created_at": datetime}``
        where ``key`` is the raw secret shown only at creation time.

    Raises:
        HTTPException 404: if the project does not exist.
    """
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    raw_key, prefix, key_hash = generate_api_key()

    api_key = ApiKey(
        project_id=project_id,
        name=name,
        key_prefix=prefix,
        key_hash=key_hash,
        is_active=True,
    )
    db.add(api_key)
    await db.flush()

    return {
        "id": api_key.id,
        "name": api_key.name,
        "key": raw_key,
        "key_prefix": api_key.key_prefix,
        "created_at": api_key.created_at,
    }


@router.get("/{project_id}/api-keys")
async def list_api_keys(
    project_id: int,
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    """List API keys for a project (without the raw secret).

    Raises:
        HTTPException 404: if the project does not exist.
    """
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    result = await db.execute(
        select(ApiKey).where(ApiKey.project_id == project_id).order_by(ApiKey.id)
    )
    keys = result.scalars().all()

    return [
        {
            "id": k.id,
            "name": k.name,
            "key_prefix": k.key_prefix,
            "is_active": k.is_active,
            "created_at": k.created_at,
            "last_used_at": k.last_used_at,
        }
        for k in keys
    ]


# ---------------------------------------------------------------------------
# Usage endpoints
# ---------------------------------------------------------------------------


@router.get("/{project_id}/usage")
async def get_usage(
    project_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return a full usage summary for a project.

    Includes current-month totals, all-time totals, and cumulative storage
    bytes written.

    Raises:
        HTTPException 404: if the project does not exist.
    """
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    return await project_usage_summary(db, project_id)


@router.get("/{project_id}/usage/current-month")
async def get_usage_current_month(
    project_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return current-calendar-month usage totals for a project.

    Returns a dict of ``{event_type: total_quantity}`` for the current UTC
    calendar month.

    Raises:
        HTTPException 404: if the project does not exist.
    """
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    now = datetime.now(UTC)
    totals = await monthly_totals(db, project_id)
    return {
        "project_id": project_id,
        "year": now.year,
        "month": now.month,
        "totals": totals,
    }


@router.get("/{project_id}/quotas")
async def get_quotas(
    project_id: int,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Return the current quota configuration for a project.

    All quota fields are ``null`` when no limit is set (i.e. unlimited).

    Raises:
        HTTPException 404: if the project does not exist.
    """
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    return {
        "project_id": project_id,
        "max_videos_per_month": project.max_videos_per_month,
        "max_video_seconds_per_month": project.max_video_seconds_per_month,
        "max_storage_bytes": project.max_storage_bytes,
        "max_api_reads_per_month": project.max_api_reads_per_month,
        "is_suspended": project.is_suspended,
    }

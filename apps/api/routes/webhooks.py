"""Webhook endpoint management routes (project-scoped)."""

from __future__ import annotations

import secrets

from fastapi import APIRouter, Body, HTTPException
from fastapi.params import Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.db import get_db
from libs.models import Project, WebhookEndpoint

router = APIRouter()

# Number of random bytes used to generate per-endpoint signing secrets.
_SECRET_BYTES = 32


def _generate_webhook_secret() -> str:
    """Return a cryptographically random hex secret for HMAC signing."""
    return secrets.token_hex(_SECRET_BYTES)


@router.post("/{project_id}/webhooks", status_code=201)
async def create_webhook(
    project_id: int,
    url: str = Body(..., embed=True),
    event_types: list[str] | None = Body(None, embed=True),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Register a new webhook endpoint for a project.

    Args:
        project_id: Target project.
        url: The HTTPS URL that will receive signed POST requests.
        event_types: Optional list of event type strings to subscribe to.
            When omitted (``null``) the endpoint receives **all** event types.
            Supported values: ``processing_run.created``,
            ``processing_run.running``, ``processing_run.completed``,
            ``processing_run.failed``.

    Returns:
        Webhook metadata including the generated ``secret`` (shown once).

    Raises:
        HTTPException 404: if the project does not exist.
    """
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    secret = _generate_webhook_secret()

    webhook = WebhookEndpoint(
        project_id=project_id,
        url=url,
        secret=secret,
        is_active=True,
        event_types_json=event_types,
    )
    db.add(webhook)
    await db.flush()

    return {
        "id": webhook.id,
        "project_id": webhook.project_id,
        "url": webhook.url,
        "secret": secret,
        "is_active": webhook.is_active,
        "event_types": webhook.event_types_json,
        "created_at": webhook.created_at,
    }


@router.get("/{project_id}/webhooks")
async def list_webhooks(
    project_id: int,
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    """List all webhook endpoints registered for a project.

    The signing secret is **not** included in list responses.

    Raises:
        HTTPException 404: if the project does not exist.
    """
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    result = await db.execute(
        select(WebhookEndpoint)
        .where(WebhookEndpoint.project_id == project_id)
        .order_by(WebhookEndpoint.id)
    )
    webhooks = result.scalars().all()

    return [
        {
            "id": w.id,
            "project_id": w.project_id,
            "url": w.url,
            "is_active": w.is_active,
            "event_types": w.event_types_json,
            "created_at": w.created_at,
            "updated_at": w.updated_at,
            "last_success_at": w.last_success_at,
            "last_failure_at": w.last_failure_at,
        }
        for w in webhooks
    ]


@router.patch("/{project_id}/webhooks/{webhook_id}")
async def update_webhook(
    project_id: int,
    webhook_id: int,
    url: str | None = Body(None, embed=True),
    is_active: bool | None = Body(None, embed=True),
    event_types: list[str] | None = Body(None, embed=True),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Update a webhook endpoint (partial update).

    Any field left as ``null`` in the request body is left unchanged.

    Raises:
        HTTPException 404: if the project or webhook does not exist or the
            webhook does not belong to the project.
    """
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    webhook = await db.get(WebhookEndpoint, webhook_id)
    if webhook is None or webhook.project_id != project_id:
        raise HTTPException(status_code=404, detail="Webhook not found")

    if url is not None:
        webhook.url = url
    if is_active is not None:
        webhook.is_active = is_active
    if event_types is not None:
        webhook.event_types_json = event_types

    await db.flush()

    return {
        "id": webhook.id,
        "project_id": webhook.project_id,
        "url": webhook.url,
        "is_active": webhook.is_active,
        "event_types": webhook.event_types_json,
        "updated_at": webhook.updated_at,
    }


@router.delete("/{project_id}/webhooks/{webhook_id}", status_code=204)
async def delete_webhook(
    project_id: int,
    webhook_id: int,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete a webhook endpoint.

    Raises:
        HTTPException 404: if the project or webhook does not exist or the
            webhook does not belong to the project.
    """
    project = await db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    webhook = await db.get(WebhookEndpoint, webhook_id)
    if webhook is None or webhook.project_id != project_id:
        raise HTTPException(status_code=404, detail="Webhook not found")

    await db.delete(webhook)

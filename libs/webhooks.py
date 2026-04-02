"""Webhook signing, payload building, and delivery enqueueing."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import uuid
from datetime import UTC, datetime

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from libs.events import RunEventType
from libs.models import WebhookEndpoint
from libs.queue import QUEUE_KEY, get_redis

logger = logging.getLogger(__name__)

# Maximum number of delivery attempts (initial + retries).
MAX_DELIVERY_ATTEMPTS = 4
# HTTP timeout in seconds for a single delivery attempt.
DELIVERY_TIMEOUT_SECONDS = 10.0


def sign_payload(secret: str, payload_bytes: bytes) -> str:
    """Return an HMAC-SHA256 hex digest of *payload_bytes* signed with *secret*.

    The signature is sent in the ``X-MotionState-Signature`` header so that
    receivers can verify authenticity by computing the same HMAC over the raw
    request body.
    """
    return hmac.new(secret.encode(), payload_bytes, hashlib.sha256).hexdigest()


def build_run_event_payload(
    event_type: RunEventType,
    *,
    project_id: int,
    video_id: int,
    processing_run_id: int,
    status: str,
    artifact_types: list[str] | None = None,
    error: str | None = None,
) -> dict:
    """Build a compact, domain-agnostic webhook payload dict.

    The payload is deterministically serialised (``sort_keys=True``) before
    signing so that signature verification is stable across implementations.

    Returns:
        A plain dict ready to be JSON-serialised and delivered.
    """
    payload: dict = {
        "event_id": str(uuid.uuid4()),
        "event_type": event_type,
        "occurred_at": datetime.now(UTC).isoformat(),
        "project_id": project_id,
        "video_id": video_id,
        "processing_run_id": processing_run_id,
        "status": status,
    }
    if artifact_types is not None:
        payload["artifact_types"] = artifact_types
    if error is not None:
        payload["error"] = error
    return payload


async def enqueue_run_event(
    db: AsyncSession,
    event_type: RunEventType,
    *,
    project_id: int,
    video_id: int,
    processing_run_id: int,
    status: str,
    artifact_types: list[str] | None = None,
    error: str | None = None,
) -> None:
    """Query active webhooks for *project_id* and enqueue a delivery job for each.

    Webhooks are filtered by event type subscription (``event_types_json``).
    When ``event_types_json`` is ``None`` the endpoint receives all event types.
    Inactive endpoints are skipped.

    This function is intentionally non-blocking: it only pushes messages to
    Redis and does not wait for HTTP delivery.

    Errors are logged but not re-raised so that webhook delivery never blocks
    or breaks the main processing flow.
    """
    try:
        result = await db.execute(
            select(WebhookEndpoint).where(
                WebhookEndpoint.project_id == project_id,
                WebhookEndpoint.is_active.is_(True),
            )
        )
        webhooks = result.scalars().all()
    except Exception:
        logger.exception("Failed to query webhooks for project %s", project_id)
        return

    payload = build_run_event_payload(
        event_type,
        project_id=project_id,
        video_id=video_id,
        processing_run_id=processing_run_id,
        status=status,
        artifact_types=artifact_types,
        error=error,
    )

    redis = get_redis()
    for webhook in webhooks:
        # Skip endpoints that don't subscribe to this event type.
        subscribed = webhook.event_types_json
        if subscribed is not None and event_type not in subscribed:
            continue

        message = json.dumps(
            {
                "job_id": 0,
                "type": "deliver_webhook",
                "payload": {
                    "webhook_id": webhook.id,
                    "url": webhook.url,
                    "secret": webhook.secret,
                    "event_payload": payload,
                    "retry_count": 0,
                },
            }
        )
        try:
            await redis.rpush(QUEUE_KEY, message)
        except Exception:
            logger.exception("Failed to enqueue webhook delivery for endpoint %s", webhook.id)


async def deliver_webhook(url: str, secret: str, payload_bytes: bytes) -> None:
    """POST *payload_bytes* to *url* with an HMAC-SHA256 signature header.

    Raises:
        httpx.HTTPStatusError: if the server responds with a 4xx/5xx status.
        httpx.RequestError: on network-level failures (timeout, DNS, etc.).
    """
    signature = sign_payload(secret, payload_bytes)
    async with httpx.AsyncClient(timeout=DELIVERY_TIMEOUT_SECONDS) as client:
        response = await client.post(
            url,
            content=payload_bytes,
            headers={
                "Content-Type": "application/json",
                "X-MotionState-Signature": signature,
            },
        )
        response.raise_for_status()

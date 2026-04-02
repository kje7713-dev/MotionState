"""deliver_webhook job handler.

Picks up webhook delivery messages from the queue and POSTs signed payloads
to the registered endpoint URL.  Credentials (URL, secret) are loaded from
the database at delivery time so that secrets are never stored in the queue.
Failed deliveries update ``last_failure_at`` and are re-enqueued with an
incremented ``retry_count`` until :data:`~libs.webhooks.MAX_DELIVERY_ATTEMPTS`
is reached.
"""

import json
import logging
from datetime import UTC, datetime

from libs.db import AsyncSessionLocal
from libs.models import UsageEventType, WebhookEndpoint
from libs.queue import QUEUE_KEY, get_redis
from libs.usage import emit as emit_usage
from libs.webhooks import MAX_DELIVERY_ATTEMPTS, deliver_webhook

logger = logging.getLogger(__name__)


async def handle_deliver_webhook(message: dict) -> None:
    """Deliver a single webhook payload, retrying on transient failures."""
    payload_data: dict = message.get("payload", {})
    webhook_id: int | None = payload_data.get("webhook_id")
    event_payload: dict = payload_data.get("event_payload", {})
    retry_count: int = payload_data.get("retry_count", 0)
    event_type: str = event_payload.get("event_type", "unknown")

    if webhook_id is None:
        logger.warning("deliver_webhook message is missing webhook_id; skipping")
        return

    payload_bytes = json.dumps(event_payload, sort_keys=True).encode()

    # Credentials are loaded from DB here; they are never stored in the queue.
    success, skip = await _load_and_deliver(webhook_id, payload_bytes)

    if skip:
        logger.info("Webhook endpoint %s not found or inactive; skipping", webhook_id)
        return

    if success:
        logger.info(
            "Webhook delivery succeeded: endpoint=%s event_type=%s attempt=%s",
            webhook_id,
            event_type,
            retry_count + 1,
        )
        await _update_webhook_timestamp(webhook_id, success=True)
        await _emit_webhook_usage(webhook_id)
    else:
        logger.warning(
            "Webhook delivery failed: endpoint=%s event_type=%s attempt=%s",
            webhook_id,
            event_type,
            retry_count + 1,
        )
        await _update_webhook_timestamp(webhook_id, success=False)

        next_retry_count = retry_count + 1
        if next_retry_count < MAX_DELIVERY_ATTEMPTS:
            await _reenqueue(webhook_id, event_payload, next_retry_count)
            logger.info(
                "Webhook re-enqueued: endpoint=%s attempt=%s/%s",
                webhook_id,
                next_retry_count + 1,
                MAX_DELIVERY_ATTEMPTS,
            )
        else:
            logger.warning(
                "Webhook delivery permanently failed after %s attempts: endpoint=%s",
                MAX_DELIVERY_ATTEMPTS,
                webhook_id,
            )


async def _load_and_deliver(webhook_id: int, payload_bytes: bytes) -> tuple[bool, bool]:
    """Load webhook credentials from DB and attempt HTTP delivery.

    Credentials (URL and secret) are only held within this function's scope
    and are never returned to or logged by the caller.

    Returns:
        ``(success, skip)`` where *skip* is True when the endpoint is not
        found or inactive (no retry needed).
    """
    try:
        async with AsyncSessionLocal() as db:
            webhook = await db.get(WebhookEndpoint, webhook_id)
            if webhook is None or not webhook.is_active:
                return False, True
            await deliver_webhook(webhook.url, webhook.secret, payload_bytes)
            return True, False
    except Exception:
        return False, False


async def _update_webhook_timestamp(webhook_id: int, *, success: bool) -> None:
    """Update last_success_at or last_failure_at on the webhook endpoint row."""
    try:
        async with AsyncSessionLocal() as db:
            webhook = await db.get(WebhookEndpoint, webhook_id)
            if webhook is not None:
                now = datetime.now(UTC)
                if success:
                    webhook.last_success_at = now
                else:
                    webhook.last_failure_at = now
                await db.commit()
    except Exception:
        logger.exception("Failed to update webhook timestamp for endpoint %s", webhook_id)


async def _reenqueue(webhook_id: int, event_payload: dict, retry_count: int) -> None:
    """Push a retry delivery message back onto the queue."""
    message = json.dumps(
        {
            "job_id": 0,
            "type": "deliver_webhook",
            "payload": {
                "webhook_id": webhook_id,
                "event_payload": event_payload,
                "retry_count": retry_count,
            },
        }
    )
    try:
        await get_redis().rpush(QUEUE_KEY, message)
    except Exception:
        logger.exception("Failed to re-enqueue webhook delivery retry")


async def _emit_webhook_usage(webhook_id: int) -> None:
    """Record a webhook_deliveries usage event for the project that owns the endpoint."""
    try:
        async with AsyncSessionLocal() as db:
            webhook = await db.get(WebhookEndpoint, webhook_id)
            if webhook is None:
                return
            await emit_usage(
                db,
                project_id=webhook.project_id,
                event_type=UsageEventType.webhook_deliveries,
                quantity=1,
                metadata={"webhook_id": webhook_id},
            )
            await db.commit()
    except Exception:
        logger.exception("Failed to emit webhook usage for endpoint %s", webhook_id)

"""deliver_webhook job handler.

Picks up webhook delivery messages from the queue and POSTs signed payloads
to the registered endpoint URL.  Failed deliveries update ``last_failure_at``
and are re-enqueued with an incremented ``retry_count`` until
:data:`~libs.webhooks.MAX_DELIVERY_ATTEMPTS` is reached.
"""

import json
import logging
from datetime import UTC, datetime

from libs.db import AsyncSessionLocal
from libs.models import WebhookEndpoint
from libs.queue import QUEUE_KEY, get_redis
from libs.webhooks import MAX_DELIVERY_ATTEMPTS, deliver_webhook

logger = logging.getLogger(__name__)


async def handle_deliver_webhook(message: dict) -> None:
    """Deliver a single webhook payload, retrying on transient failures."""
    payload_data: dict = message.get("payload", {})
    webhook_id: int | None = payload_data.get("webhook_id")
    url: str = payload_data["url"]
    secret: str = payload_data["secret"]
    event_payload: dict = payload_data["event_payload"]
    retry_count: int = payload_data.get("retry_count", 0)

    payload_bytes = json.dumps(event_payload, sort_keys=True).encode()

    try:
        await deliver_webhook(url, secret, payload_bytes)
        logger.info(
            "Webhook delivery succeeded: endpoint=%s event_type=%s attempt=%s",
            webhook_id,
            event_payload.get("event_type"),
            retry_count + 1,
        )
        # Record success timestamp.
        if webhook_id is not None:
            await _update_webhook_timestamp(webhook_id, success=True)
    except Exception as exc:
        logger.warning(
            "Webhook delivery failed: endpoint=%s event_type=%s attempt=%s error=%s",
            webhook_id,
            event_payload.get("event_type"),
            retry_count + 1,
            exc,
        )
        # Record failure timestamp.
        if webhook_id is not None:
            await _update_webhook_timestamp(webhook_id, success=False)

        # Re-enqueue if retries remain.
        next_retry_count = retry_count + 1
        if next_retry_count < MAX_DELIVERY_ATTEMPTS:
            await _reenqueue(payload_data, next_retry_count)
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


async def _reenqueue(payload_data: dict, retry_count: int) -> None:
    """Push a retry delivery message back onto the queue."""
    new_payload = {**payload_data, "retry_count": retry_count}
    message = json.dumps(
        {
            "job_id": 0,
            "type": "deliver_webhook",
            "payload": new_payload,
        }
    )
    try:
        await get_redis().rpush(QUEUE_KEY, message)
    except Exception:
        logger.exception("Failed to re-enqueue webhook delivery retry")

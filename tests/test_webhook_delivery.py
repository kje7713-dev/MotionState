"""Tests for webhook delivery logic.

Covers:
- successful delivery updates last_success_at
- failed delivery updates last_failure_at
- failed delivery is re-enqueued with retry_count+1
- delivery stops after MAX_DELIVERY_ATTEMPTS
- inactive webhook is skipped in enqueue_run_event
- deliver_webhook posts correct headers
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from libs.events import RunEventType
from libs.webhooks import MAX_DELIVERY_ATTEMPTS, deliver_webhook, sign_payload

# ---------------------------------------------------------------------------
# deliver_webhook – HTTP delivery
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deliver_webhook_posts_with_signature():
    """deliver_webhook includes X-MotionState-Signature header."""
    import httpx

    secret = "test-secret"
    payload = b'{"event_type":"processing_run.completed"}'
    expected_sig = sign_payload(secret, payload)

    captured_headers = {}

    async def _mock_post(url, *, content, headers, **kwargs):
        captured_headers.update(headers)
        response = MagicMock(spec=httpx.Response)
        response.raise_for_status = MagicMock()
        return response

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(side_effect=_mock_post)

    with patch("libs.webhooks.httpx.AsyncClient", return_value=mock_client):
        await deliver_webhook("https://example.com/hook", secret, payload)

    assert captured_headers.get("X-MotionState-Signature") == expected_sig
    assert captured_headers.get("Content-Type") == "application/json"


@pytest.mark.asyncio
async def test_deliver_webhook_raises_on_http_error():
    """deliver_webhook raises on a 4xx/5xx response."""
    import httpx

    async def _mock_post(url, *, content, headers, **kwargs):
        response = MagicMock(spec=httpx.Response)
        response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError("500", request=MagicMock(), response=MagicMock())
        )
        return response

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(side_effect=_mock_post)

    with patch("libs.webhooks.httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(httpx.HTTPStatusError):
            await deliver_webhook("https://example.com/hook", "secret", b'{}')


# ---------------------------------------------------------------------------
# handle_deliver_webhook – job handler
# ---------------------------------------------------------------------------


def _make_delivery_message(
    webhook_id: int = 1,
    url: str = "https://example.com/hook",
    secret: str = "secret",
    event_type: str = "processing_run.completed",
    retry_count: int = 0,
) -> dict:
    return {
        "job_id": 0,
        "type": "deliver_webhook",
        "payload": {
            "webhook_id": webhook_id,
            "url": url,
            "secret": secret,
            "event_payload": {
                "event_id": "abc123",
                "event_type": event_type,
                "occurred_at": "2024-01-01T00:00:00+00:00",
                "project_id": 1,
                "video_id": 1,
                "processing_run_id": 1,
                "status": "completed",
            },
            "retry_count": retry_count,
        },
    }


@pytest.mark.asyncio
async def test_successful_delivery_updates_last_success_at():
    """Successful delivery sets last_success_at on the webhook row."""
    from apps.worker.jobs.deliver_webhook import handle_deliver_webhook

    fake_webhook = MagicMock()
    fake_webhook.last_success_at = None
    fake_webhook.last_failure_at = None

    mock_db = AsyncMock()
    mock_db.get = AsyncMock(return_value=fake_webhook)
    mock_db.commit = AsyncMock()
    mock_db.__aenter__ = AsyncMock(return_value=mock_db)
    mock_db.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("apps.worker.jobs.deliver_webhook.deliver_webhook", new=AsyncMock()),
        patch("apps.worker.jobs.deliver_webhook.AsyncSessionLocal", return_value=mock_db),
    ):
        await handle_deliver_webhook(_make_delivery_message())

    assert fake_webhook.last_success_at is not None


@pytest.mark.asyncio
async def test_failed_delivery_updates_last_failure_at():
    """Failed delivery sets last_failure_at on the webhook row."""
    import httpx

    from apps.worker.jobs.deliver_webhook import handle_deliver_webhook

    fake_webhook = MagicMock()
    fake_webhook.last_success_at = None
    fake_webhook.last_failure_at = None

    mock_db = AsyncMock()
    mock_db.get = AsyncMock(return_value=fake_webhook)
    mock_db.commit = AsyncMock()
    mock_db.__aenter__ = AsyncMock(return_value=mock_db)
    mock_db.__aexit__ = AsyncMock(return_value=False)

    mock_redis = AsyncMock()
    mock_redis.rpush = AsyncMock()

    with (
        patch(
            "apps.worker.jobs.deliver_webhook.deliver_webhook",
            new=AsyncMock(side_effect=httpx.RequestError("timeout")),
        ),
        patch("apps.worker.jobs.deliver_webhook.AsyncSessionLocal", return_value=mock_db),
        patch("apps.worker.jobs.deliver_webhook.get_redis", return_value=mock_redis),
    ):
        await handle_deliver_webhook(_make_delivery_message(retry_count=0))

    assert fake_webhook.last_failure_at is not None


@pytest.mark.asyncio
async def test_failed_delivery_reenqueues_with_incremented_retry():
    """Failed delivery re-enqueues with retry_count+1."""
    import httpx

    from apps.worker.jobs.deliver_webhook import handle_deliver_webhook

    fake_webhook = MagicMock()
    mock_db = AsyncMock()
    mock_db.get = AsyncMock(return_value=fake_webhook)
    mock_db.commit = AsyncMock()
    mock_db.__aenter__ = AsyncMock(return_value=mock_db)
    mock_db.__aexit__ = AsyncMock(return_value=False)

    mock_redis = AsyncMock()
    mock_redis.rpush = AsyncMock()

    with (
        patch(
            "apps.worker.jobs.deliver_webhook.deliver_webhook",
            new=AsyncMock(side_effect=httpx.RequestError("timeout")),
        ),
        patch("apps.worker.jobs.deliver_webhook.AsyncSessionLocal", return_value=mock_db),
        patch("apps.worker.jobs.deliver_webhook.get_redis", return_value=mock_redis),
    ):
        await handle_deliver_webhook(_make_delivery_message(retry_count=0))

    assert mock_redis.rpush.called
    call_args = mock_redis.rpush.call_args
    pushed_message = json.loads(call_args[0][1])
    assert pushed_message["payload"]["retry_count"] == 1


@pytest.mark.asyncio
async def test_delivery_stops_after_max_attempts():
    """No re-enqueue when retry_count has reached MAX_DELIVERY_ATTEMPTS - 1."""
    import httpx

    from apps.worker.jobs.deliver_webhook import handle_deliver_webhook

    fake_webhook = MagicMock()
    mock_db = AsyncMock()
    mock_db.get = AsyncMock(return_value=fake_webhook)
    mock_db.commit = AsyncMock()
    mock_db.__aenter__ = AsyncMock(return_value=mock_db)
    mock_db.__aexit__ = AsyncMock(return_value=False)

    mock_redis = AsyncMock()
    mock_redis.rpush = AsyncMock()

    with (
        patch(
            "apps.worker.jobs.deliver_webhook.deliver_webhook",
            new=AsyncMock(side_effect=httpx.RequestError("timeout")),
        ),
        patch("apps.worker.jobs.deliver_webhook.AsyncSessionLocal", return_value=mock_db),
        patch("apps.worker.jobs.deliver_webhook.get_redis", return_value=mock_redis),
    ):
        # Simulate the final allowed attempt (MAX - 1 means next would be MAX).
        await handle_deliver_webhook(
            _make_delivery_message(retry_count=MAX_DELIVERY_ATTEMPTS - 1)
        )

    # Should NOT re-enqueue since we've exhausted all attempts.
    assert not mock_redis.rpush.called


# ---------------------------------------------------------------------------
# enqueue_run_event – filtering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inactive_webhook_is_skipped():
    """enqueue_run_event does not enqueue a delivery for inactive webhooks."""
    from libs.webhooks import enqueue_run_event

    # Mock DB returns no active webhooks (is_active filter in the query).
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = []  # no active webhooks
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars

    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(return_value=mock_result)

    mock_redis = AsyncMock()
    mock_redis.rpush = AsyncMock()

    with patch("libs.webhooks.get_redis", return_value=mock_redis):
        await enqueue_run_event(
            mock_db,
            RunEventType.completed,
            project_id=1,
            video_id=1,
            processing_run_id=1,
            status="completed",
        )

    assert not mock_redis.rpush.called


@pytest.mark.asyncio
async def test_event_type_filter_skips_unsubscribed():
    """enqueue_run_event skips endpoints not subscribed to the event type."""
    from libs.models import WebhookEndpoint
    from libs.webhooks import enqueue_run_event

    # Webhook only subscribed to failed events.
    fake_webhook = MagicMock(spec=WebhookEndpoint)
    fake_webhook.id = 1
    fake_webhook.url = "https://example.com/hook"
    fake_webhook.secret = "secret"
    fake_webhook.event_types_json = ["processing_run.failed"]

    mock_scalars = MagicMock()
    mock_scalars.all.return_value = [fake_webhook]
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars

    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(return_value=mock_result)

    mock_redis = AsyncMock()
    mock_redis.rpush = AsyncMock()

    with patch("libs.webhooks.get_redis", return_value=mock_redis):
        # Emit a "completed" event – webhook only wants "failed".
        await enqueue_run_event(
            mock_db,
            RunEventType.completed,
            project_id=1,
            video_id=1,
            processing_run_id=1,
            status="completed",
        )

    assert not mock_redis.rpush.called


@pytest.mark.asyncio
async def test_null_event_types_receives_all_events():
    """Webhook with event_types_json=None receives all event types."""
    from libs.models import WebhookEndpoint
    from libs.webhooks import enqueue_run_event

    fake_webhook = MagicMock(spec=WebhookEndpoint)
    fake_webhook.id = 1
    fake_webhook.url = "https://example.com/hook"
    fake_webhook.secret = "secret"
    fake_webhook.event_types_json = None  # No filter → all events.

    mock_scalars = MagicMock()
    mock_scalars.all.return_value = [fake_webhook]
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars

    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(return_value=mock_result)

    mock_redis = AsyncMock()
    mock_redis.rpush = AsyncMock()

    with patch("libs.webhooks.get_redis", return_value=mock_redis):
        await enqueue_run_event(
            mock_db,
            RunEventType.running,
            project_id=1,
            video_id=1,
            processing_run_id=1,
            status="running",
        )

    assert mock_redis.rpush.called

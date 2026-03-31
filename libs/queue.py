"""Redis-backed job queue helpers."""

import json

import redis.asyncio as aioredis

from libs.config import settings

_pool: aioredis.Redis | None = None

QUEUE_KEY = "motionstate:jobs"


def get_redis() -> aioredis.Redis:
    """Return (and lazily create) the shared Redis connection pool."""
    global _pool
    if _pool is None:
        _pool = aioredis.from_url(settings.redis_url, decode_responses=True)
    return _pool


async def enqueue(job_id: int, job_type: str, payload: dict) -> None:
    """Push a job message onto the queue."""
    message = json.dumps({"job_id": job_id, "type": job_type, "payload": payload})
    await get_redis().rpush(QUEUE_KEY, message)


async def dequeue(timeout: int = 5) -> dict | None:
    """Block until a job is available and return the parsed message.

    Returns None on timeout.
    """
    result = await get_redis().blpop(QUEUE_KEY, timeout=timeout)
    if result is None:
        return None
    _, raw = result
    return json.loads(raw)

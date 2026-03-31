"""Worker entry point – polls Redis and dispatches jobs."""

import asyncio
import logging

from apps.worker.jobs.process_video import handle_process_video
from libs.queue import dequeue

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HANDLERS = {
    "process_video": handle_process_video,
}


async def run_worker() -> None:
    """Blocking event loop that continuously polls the queue."""
    logger.info("Worker started – waiting for jobs…")
    while True:
        message = await dequeue(timeout=5)
        if message is None:
            continue

        job_type = message.get("type")
        handler = HANDLERS.get(job_type)
        if handler is None:
            logger.warning("Unknown job type: %s", job_type)
            continue

        logger.info("Handling job %s (type=%s)", message.get("job_id"), job_type)
        try:
            await handler(message)
        except Exception:
            logger.exception("Unhandled error processing job %s", message.get("job_id"))


if __name__ == "__main__":
    asyncio.run(run_worker())

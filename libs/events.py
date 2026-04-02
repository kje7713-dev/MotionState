"""Run lifecycle event type definitions."""

import enum


class RunEventType(enum.StrEnum):
    """Supported processing-run lifecycle event types."""

    created = "processing_run.created"
    running = "processing_run.running"
    completed = "processing_run.completed"
    failed = "processing_run.failed"

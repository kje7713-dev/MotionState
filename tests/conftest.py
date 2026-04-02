"""Shared test helpers and fixtures for MotionState tests."""

from __future__ import annotations

from unittest.mock import MagicMock


def make_fake_project(project_id: int = 1):
    """Return a MagicMock that looks like a Project ORM row."""
    from libs.models import Project

    p = MagicMock(spec=Project)
    p.id = project_id
    p.name = "test-project"
    # Quota fields default to unlimited / not suspended.
    p.is_suspended = False
    p.max_videos_per_month = None
    p.max_video_seconds_per_month = None
    p.max_storage_bytes = None
    p.max_api_reads_per_month = None
    return p


def make_auth_override(project_id: int = 1):
    """Return an async callable that can be used as a get_current_project override."""
    fake_project = make_fake_project(project_id)

    async def _override():
        return fake_project

    return _override

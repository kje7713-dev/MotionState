"""Shared test helpers and fixtures for MotionState tests."""

from __future__ import annotations

from unittest.mock import MagicMock


def make_fake_project(project_id: int = 1):
    """Return a MagicMock that looks like a Project ORM row."""
    from libs.models import Project

    p = MagicMock(spec=Project)
    p.id = project_id
    p.name = "test-project"
    return p


def make_auth_override(project_id: int = 1):
    """Return an async callable that can be used as a get_current_project override."""
    fake_project = make_fake_project(project_id)

    async def _override():
        return fake_project

    return _override

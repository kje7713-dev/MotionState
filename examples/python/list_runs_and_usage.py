#!/usr/bin/env python3
"""List all processing runs and current-month usage for a video / project.

Usage::

    export MOTIONSTATE_API_KEY=ms_your_key_here
    export MOTIONSTATE_BASE_URL=http://localhost:8000
    python list_runs_and_usage.py <video_id> <project_id>
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "sdk", "python"))

from motionstate_client import MotionStateClient

BASE_URL = os.environ.get("MOTIONSTATE_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("MOTIONSTATE_API_KEY", "")


def main() -> None:
    if not API_KEY:
        print("ERROR: set MOTIONSTATE_API_KEY environment variable")
        sys.exit(1)
    if len(sys.argv) < 3:
        print("Usage: list_runs_and_usage.py <video_id> <project_id>")
        sys.exit(1)

    video_id = int(sys.argv[1])
    project_id = int(sys.argv[2])

    with MotionStateClient(base_url=BASE_URL, api_key=API_KEY) as client:
        # Show processing runs.
        runs = client.list_runs(video_id)
        print(f"Processing runs for video {video_id} ({len(runs)} total):")
        for run in runs:
            completed = run.completed_at or "—"
            error = f"  error: {run.error}" if run.error else ""
            print(
                f"  run {run.id}: {run.status:12s}  started={run.created_at}"
                f"  completed={completed}{error}"
            )

        # Show usage summary.
        usage = client.get_project_usage(project_id)
        print(f"\nUsage summary for project {project_id}:")
        current = usage.get("current_month", {})
        for event_type, total in current.items():
            print(f"  {event_type}: {total}")

        storage = usage.get("storage_bytes", 0)
        print(f"  storage_bytes_written: {storage}")


if __name__ == "__main__":
    main()

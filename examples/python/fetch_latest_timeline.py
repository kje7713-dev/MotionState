#!/usr/bin/env python3
"""Fetch and display the latest timeline manifest for an existing video.

Usage::

    export MOTIONSTATE_API_KEY=ms_your_key_here
    export MOTIONSTATE_BASE_URL=http://localhost:8000
    python fetch_latest_timeline.py <video_id>
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "sdk", "python"))

from motionstate_client import MotionStateClient, NotFoundError

BASE_URL = os.environ.get("MOTIONSTATE_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("MOTIONSTATE_API_KEY", "")


def main() -> None:
    if not API_KEY:
        print("ERROR: set MOTIONSTATE_API_KEY environment variable")
        sys.exit(1)
    if len(sys.argv) < 2:
        print("Usage: fetch_latest_timeline.py <video_id>")
        sys.exit(1)

    video_id = int(sys.argv[1])
    run_id = int(sys.argv[2]) if len(sys.argv) > 2 else None

    with MotionStateClient(base_url=BASE_URL, api_key=API_KEY) as client:
        video = client.get_video(video_id)
        print(f"Video {video_id}: {video.original_filename}  status={video.status}")

        try:
            timeline = client.get_timeline(video_id, run_id=run_id)
        except NotFoundError:
            print("Timeline manifest not yet available (processing may still be running).")
            sys.exit(1)

        print(f"\nTimeline manifest (schema v{timeline.get('version', '?')}):")
        print(json.dumps(timeline, indent=2, default=str))


if __name__ == "__main__":
    main()

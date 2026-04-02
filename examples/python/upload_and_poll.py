#!/usr/bin/env python3
"""Upload a video and poll until processing completes, then print a summary.

Usage::

    export MOTIONSTATE_API_KEY=ms_your_key_here
    export MOTIONSTATE_BASE_URL=http://localhost:8000  # optional, defaults shown
    python upload_and_poll.py path/to/video.mp4
"""

from __future__ import annotations

import os
import sys

# Allow running from the repo root without installing the SDK package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "sdk", "python"))

from motionstate_client import MotionStateClient, PollingTimeout

BASE_URL = os.environ.get("MOTIONSTATE_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("MOTIONSTATE_API_KEY", "")


def main() -> None:
    if not API_KEY:
        print("ERROR: set MOTIONSTATE_API_KEY environment variable")
        sys.exit(1)
    if len(sys.argv) < 2:
        print("Usage: upload_and_poll.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]

    with MotionStateClient(base_url=BASE_URL, api_key=API_KEY) as client:
        # 1. Upload the video file.
        print(f"Uploading {video_path} …")
        upload = client.submit_video(video_path)
        print(f"  video_id={upload.video_id}  run_id={upload.processing_run_id}")

        # 2. Wait for the processing run to finish.
        print("Waiting for processing to complete …")
        try:
            run = client.wait_for_run_completion(
                upload.video_id,
                upload.processing_run_id,
                timeout=600,
                poll_interval=5,
            )
        except PollingTimeout as exc:
            print(f"Timed out: {exc}")
            sys.exit(1)

        print(f"Run {run.id} finished with status: {run.status}")
        if run.error:
            print(f"  error: {run.error}")
            sys.exit(1)

        # 3. Fetch all available outputs.
        print("Fetching pipeline outputs …")
        outputs = client.fetch_latest_outputs(upload.video_id, run_id=run.id)
        for artifact_type, artifact in outputs.items():
            version = artifact.get("version", "?")
            print(f"  {artifact_type}: schema v{version}")

        state = outputs.get("state")
        if state:
            t = state.get("tracking_summary", {})
            print(f"\nTracking summary: {t.get('track_count', 0)} tracks")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Initiate a signed upload, PUT the file directly, then poll for completion.

This example demonstrates the production-oriented upload flow where the video
bytes bypass the API server and go straight to object storage (S3/R2).

For local backends the ``upload_url`` will be ``None``; the script falls back
to a normal multipart upload automatically.

Usage::

    export MOTIONSTATE_API_KEY=ms_your_key_here
    export MOTIONSTATE_BASE_URL=http://localhost:8000
    python signed_upload_then_poll.py path/to/video.mp4
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "sdk", "python"))

import httpx
from motionstate_client import MotionStateClient, PollingTimeout

BASE_URL = os.environ.get("MOTIONSTATE_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("MOTIONSTATE_API_KEY", "")


def _put_to_signed_url(upload_url: str, file_path: str) -> None:
    """Upload file bytes directly to a pre-signed storage URL."""
    with open(file_path, "rb") as fh:
        data = fh.read()
    resp = httpx.put(upload_url, content=data, headers={"Content-Type": "video/mp4"})
    resp.raise_for_status()


def main() -> None:
    if not API_KEY:
        print("ERROR: set MOTIONSTATE_API_KEY environment variable")
        sys.exit(1)
    if len(sys.argv) < 2:
        print("Usage: signed_upload_then_poll.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    filename = os.path.basename(video_path)

    with MotionStateClient(base_url=BASE_URL, api_key=API_KEY) as client:
        # 1. Initialise the upload and get a pre-signed URL.
        print(f"Initialising upload for {filename} …")
        init = client.upload_init(filename)
        print(f"  video_id={init.video_id}  storage_key={init.storage_key}")

        if init.upload_url:
            # 2a. PUT the file directly to object storage.
            print("Uploading to signed URL …")
            _put_to_signed_url(init.upload_url, video_path)
            print("  Upload complete.")

            # 3. Trigger processing via reprocess endpoint.
            print("Triggering processing run …")
            reprocess = client.reprocess_video(init.video_id)
            run_id = reprocess.processing_run_id
        else:
            # 2b. Local backend — fall back to direct multipart upload.
            print("No signed URL available (local backend). Using direct upload …")
            upload = client.upload_video(video_path)
            run_id = upload.processing_run_id

        # 4. Wait for completion.
        print(f"Waiting for run {run_id} to complete …")
        try:
            run = client.wait_for_run_completion(
                init.video_id,
                run_id,
                timeout=600,
                poll_interval=5,
            )
        except PollingTimeout as exc:
            print(f"Timed out: {exc}")
            sys.exit(1)

        print(f"Run finished: status={run.status}")
        if run.error:
            print(f"Error: {run.error}")
            sys.exit(1)

        # 5. Fetch outputs.
        outputs = client.fetch_latest_outputs(init.video_id, run_id=run.id)
        print(f"\nAvailable outputs: {', '.join(outputs.keys())}")


if __name__ == "__main__":
    main()

"""FFmpeg helpers for video normalization and metadata probing."""

import json
import subprocess
from pathlib import Path
from typing import TypedDict


class VideoMeta(TypedDict):
    duration_seconds: float
    fps: float
    width: int
    height: int


def normalize_video(input_path: str | Path, output_path: str | Path) -> None:
    """Re-encode *input_path* to a consistent H.264/AAC MP4 at *output_path*.

    Uses libx264 with CRF 23 and yuv420p pixel format so the output is
    compatible with most players and downstream CV libraries.

    Raises:
        subprocess.CalledProcessError: if ffmpeg exits with a non-zero status.
        FileNotFoundError: if ffmpeg is not installed.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(output_path),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )


def probe_video(path: str | Path) -> VideoMeta:
    """Return basic metadata for the video at *path* via ffprobe.

    Raises:
        subprocess.CalledProcessError: if ffprobe exits with a non-zero status.
        ValueError: if the probe output cannot be parsed.
        FileNotFoundError: if ffprobe is not installed.
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )

    data = json.loads(result.stdout)

    # Pick the first video stream.
    video_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
        None,
    )
    if video_stream is None:
        raise ValueError(f"No video stream found in {path}")

    # FPS may be expressed as a fraction string like "30000/1001".
    fps_raw = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate", "0/1")
    num, den = (float(x) for x in fps_raw.split("/"))
    fps = num / den if den else 0.0

    duration = float(data.get("format", {}).get("duration", 0))

    return VideoMeta(
        duration_seconds=duration,
        fps=fps,
        width=int(video_stream.get("width", 0)),
        height=int(video_stream.get("height", 0)),
    )

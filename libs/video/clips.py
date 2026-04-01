"""Clip generation helper for temporal segments.

Given a normalized video path and a list of segment dicts (with ``start_ms``,
``end_ms``, and ``label`` fields) this module extracts one MP4 clip per
segment using FFmpeg.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def generate_clip(
    video_path: str | Path,
    start_ms: float,
    end_ms: float,
    output_path: str | Path,
) -> None:
    """Extract a single clip from *video_path* between *start_ms* and *end_ms*.

    The clip is written to *output_path* as an MP4 using stream-copy (no
    re-encoding) so the operation is fast.

    Args:
        video_path: Path to the source (normalized) video.
        start_ms: Clip start position in milliseconds.
        end_ms: Clip end position in milliseconds.
        output_path: Destination MP4 file path.

    Raises:
        subprocess.CalledProcessError: if ffmpeg exits with a non-zero status.
        FileNotFoundError: if ffmpeg is not installed.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    start_sec = start_ms / 1000.0
    duration_sec = max((end_ms - start_ms) / 1000.0, 0.0)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start_sec),
        "-i", str(video_path),
        "-t", str(duration_sec),
        "-c", "copy",
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


def generate_clips(
    video_path: str | Path,
    segments: list[dict],
    output_dir: str | Path,
) -> list[dict]:
    """Generate one MP4 clip per segment and return clip metadata.

    Clips are written under *output_dir* with deterministic filenames of the
    form ``segment_{index:03d}_{label}.mp4`` (e.g.
    ``segment_000_low_motion.mp4``).

    Args:
        video_path: Path to the normalized source video.
        segments: List of segment dicts; each must contain ``start_ms``,
            ``end_ms``, and ``label``.
        output_dir: Directory to write clip files into.

    Returns:
        List of clip info dicts, one per segment, each containing:
        ``segment_index``, ``label``, ``start_ms``, ``end_ms``, and ``path``.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    clips: list[dict] = []
    for i, seg in enumerate(segments):
        label = seg["label"]
        filename = f"segment_{i:03d}_{label}.mp4"
        clip_path = out / filename
        generate_clip(video_path, seg["start_ms"], seg["end_ms"], clip_path)
        clips.append(
            {
                "segment_index": i,
                "label": label,
                "start_ms": seg["start_ms"],
                "end_ms": seg["end_ms"],
                "path": str(clip_path),
            }
        )

    return clips

"""Frame extraction helper using FFmpeg.

Extracts individual JPEG frames from a video at a configurable sample rate
and writes them to an output directory.  Returns structured metadata for
each extracted frame.
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FrameMeta:
    """Metadata for a single extracted frame."""

    frame_index: int
    timestamp_ms: float
    path: str


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    sample_fps: float = 2.0,
) -> list[FrameMeta]:
    """Extract frames from *video_path* at *sample_fps* and write JPEGs to *output_dir*.

    Args:
        video_path: Path to the source (normalized) video file.
        output_dir: Directory under which frame JPEGs will be written.
        sample_fps: Number of frames to sample per second of video.

    Returns:
        A list of :class:`FrameMeta` objects, one per extracted frame, in
        ascending order of *frame_index*.

    Raises:
        subprocess.CalledProcessError: if ffmpeg exits with a non-zero status.
        FileNotFoundError: if ffmpeg is not installed or *video_path* is missing.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pattern for output JPEG files: frame_000000.jpg, frame_000001.jpg, …
    output_pattern = str(out_dir / "frame_%06d.jpg")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vf", f"fps={sample_fps}",
        "-q:v", "2",  # high-quality JPEG
        output_pattern,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )

    # Collect written frame files in sorted order and build metadata.
    frame_files = sorted(out_dir.glob("frame_*.jpg"))
    frames: list[FrameMeta] = []
    interval_ms = 1000.0 / sample_fps

    for idx, frame_path in enumerate(frame_files):
        frames.append(
            FrameMeta(
                frame_index=idx,
                timestamp_ms=round(idx * interval_ms, 3),
                path=str(frame_path),
            )
        )

    return frames

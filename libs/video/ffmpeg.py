"""FFmpeg helpers for video normalization and metadata probing."""

import json
import subprocess
from pathlib import Path
from typing import TypedDict

# ---------------------------------------------------------------------------
# Normalization constants – keep these conservative so small workers survive.
# ---------------------------------------------------------------------------

# Maximum output resolution bounding box (portrait and landscape both fit in).
_MAX_OUTPUT_WIDTH = 1280
_MAX_OUTPUT_HEIGHT = 720

# Frame-rate cap.  Phone video can shoot at 60 fps; 30 is plenty for CV work.
_MAX_OUTPUT_FPS = 30

# Thread limit – bounds per-worker RSS during encode.
_FFMPEG_THREADS = 2

# Preset / CRF: superfast + slightly higher CRF = significantly cheaper encode
# at the cost of a marginally larger file, which is an acceptable trade-off.
_FFMPEG_PRESET = "superfast"
_FFMPEG_CRF = 28


class VideoMeta(TypedDict):
    duration_seconds: float
    fps: float
    width: int
    height: int


class StreamInfo(TypedDict):
    """Minimal stream description returned by :func:`probe_media_streams`."""

    has_video: bool
    has_audio: bool
    video_codec: str | None
    audio_codec: str | None
    container_format: str | None


class NormalizationError(Exception):
    """Raised when ffmpeg normalization fails.

    Carries the raw ``returncode`` and the captured ``stderr`` text so callers
    can surface a useful diagnostic without keeping megabytes of output around.
    """

    def __init__(self, message: str, returncode: int, stderr: str) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


def probe_media_streams(path: str | Path) -> StreamInfo:
    """Inspect *path* with ffprobe and return a lightweight :class:`StreamInfo`.

    Unlike :func:`probe_video`, this function does not require a video stream –
    it simply describes what is present.  Use it to validate the source and
    decide whether normalisation makes sense *before* launching a potentially
    expensive ffmpeg encode.  The result is intentionally media-content-driven:
    the file extension is irrelevant to ffprobe.

    Raises:
        subprocess.CalledProcessError: if ffprobe exits non-zero.
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
    streams = data.get("streams", [])
    fmt = data.get("format", {})

    video_stream = next(
        (s for s in streams if s.get("codec_type") == "video"), None
    )
    audio_stream = next(
        (s for s in streams if s.get("codec_type") == "audio"), None
    )

    return StreamInfo(
        has_video=video_stream is not None,
        has_audio=audio_stream is not None,
        video_codec=video_stream.get("codec_name") if video_stream else None,
        audio_codec=audio_stream.get("codec_name") if audio_stream else None,
        container_format=fmt.get("format_name"),
    )


def normalize_video(input_path: str | Path, output_path: str | Path) -> None:
    """Re-encode *input_path* to a consistent H.264/AAC MP4 at *output_path*.

    Uses deliberately conservative settings so it runs reliably on constrained
    workers.  The goal is *robust and cheap*, not broadcast-quality output:

    * Output is capped at :data:`_MAX_OUTPUT_WIDTH` × :data:`_MAX_OUTPUT_HEIGHT`
      (portrait and landscape videos both fit within the bounding box while
      preserving their original aspect ratio).
    * Frame rate is capped at :data:`_MAX_OUTPUT_FPS`.
    * Thread count is capped at :data:`_FFMPEG_THREADS`.
    * The ``superfast`` preset reduces encode CPU and memory pressure at the
      cost of a slightly larger output file – an acceptable trade-off here.
    * Audio is mapped *optionally*: if the source contains no audio track
      (silent clips, time-lapses, etc.) the output will simply omit audio
      rather than erroring out.
    * Only errors are logged by ffmpeg (``-loglevel error``) to prevent
      buffering megabytes of per-frame progress lines in ``capture_output``.

    Raises:
        NormalizationError: if ffmpeg exits with a non-zero status.  The
            exception message contains the trimmed stderr (last 2 KB) so it
            can be stored in a job-error field without ballooning storage.
        FileNotFoundError: if ffmpeg is not installed.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Scale filter: shrink to the bounding box while preserving aspect ratio.
    # force_original_aspect_ratio=decrease only ever *downscales*; smaller
    # inputs pass through unchanged.  The trailing pad step ensures both
    # dimensions are even, which is required for the yuv420p pixel format.
    scale_filter = (
        f"scale={_MAX_OUTPUT_WIDTH}:{_MAX_OUTPUT_HEIGHT}"
        ":force_original_aspect_ratio=decrease"
        ",pad=ceil(iw/2)*2:ceil(ih/2)*2"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        # Cap threads to contain per-worker RSS.
        "-threads", str(_FFMPEG_THREADS),
        # Select the first video stream (required) and first audio stream
        # (optional – the '?' suppresses the error when audio is absent,
        # which is common for silent clips and some time-lapses).
        "-map", "0:v:0",
        "-map", "0:a:0?",
        # Video encoding
        "-c:v", "libx264",
        "-crf", str(_FFMPEG_CRF),
        "-preset", _FFMPEG_PRESET,
        "-pix_fmt", "yuv420p",
        "-r", str(_MAX_OUTPUT_FPS),
        "-vf", scale_filter,
        # Audio: mono 64 k is more than enough for CV downstream work.
        "-c:a", "aac",
        "-ac", "1",
        "-b:a", "64k",
        "-movflags", "+faststart",
        # Only log errors to avoid buffering megabytes of progress lines.
        "-loglevel", "error",
        str(output_path),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        # Keep only the last 2 KB of stderr – enough to diagnose, small enough
        # to store in a job error field without blowing up the database.
        stderr_snippet = (result.stderr or "").strip()[-2000:]
        msg = f"ffmpeg exited {result.returncode}"
        if stderr_snippet:
            msg = f"{msg}: {stderr_snippet}"
        raise NormalizationError(msg, result.returncode, result.stderr or "")


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

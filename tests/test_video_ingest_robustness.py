"""Tests for video ingest robustness: weird filenames, mobile formats, and
normalization memory/error-reporting improvements.

These tests verify that:
- Weird mobile-like filenames/extensions are handled without crashing.
- probe_media_streams() detects streams from file content, not filename.
- NormalizationError carries useful returncode + stderr context.
- The worker's normalization path raises clear errors for bad inputs.
- The worker logs and re-raises NormalizationError with full stderr context.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from libs.video.ffmpeg import (
    _FFMPEG_CRF,
    _FFMPEG_PRESET,
    _FFMPEG_THREADS,
    _MAX_OUTPUT_FPS,
    _MAX_OUTPUT_HEIGHT,
    _MAX_OUTPUT_WIDTH,
    NormalizationError,
    StreamInfo,
    normalize_video,
    probe_media_streams,
)

# ---------------------------------------------------------------------------
# NormalizationError
# ---------------------------------------------------------------------------


def test_normalization_error_carries_returncode():
    err = NormalizationError("bad encode", returncode=1, stderr="some error text")
    assert err.returncode == 1


def test_normalization_error_carries_stderr():
    err = NormalizationError("bad encode", returncode=1, stderr="some error text")
    assert err.stderr == "some error text"


def test_normalization_error_message_is_str():
    err = NormalizationError("ffmpeg exited 1: error detail", returncode=1, stderr="error detail")
    assert str(err) == "ffmpeg exited 1: error detail"


def test_normalization_error_is_exception():
    err = NormalizationError("oops", returncode=2, stderr="")
    assert isinstance(err, Exception)


# ---------------------------------------------------------------------------
# normalize_video – error reporting
# ---------------------------------------------------------------------------


def test_normalize_video_raises_normalization_error_on_nonzero_exit(tmp_path):
    """normalize_video raises NormalizationError (not CalledProcessError) on failure."""
    fake_result = MagicMock()
    fake_result.returncode = 1
    fake_result.stderr = "Invalid data found when processing input"
    fake_result.stdout = ""

    with patch("libs.video.ffmpeg.subprocess.run", return_value=fake_result):
        with pytest.raises(NormalizationError) as exc_info:
            normalize_video(tmp_path / "input.mov", tmp_path / "out.mp4")

    assert exc_info.value.returncode == 1
    assert "Invalid data" in exc_info.value.stderr


def test_normalize_video_error_message_includes_stderr(tmp_path):
    """NormalizationError message includes a snippet of ffmpeg stderr."""
    fake_result = MagicMock()
    fake_result.returncode = 1
    fake_result.stderr = "No such file or directory"
    fake_result.stdout = ""

    with patch("libs.video.ffmpeg.subprocess.run", return_value=fake_result):
        with pytest.raises(NormalizationError) as exc_info:
            normalize_video(tmp_path / "x.mp4", tmp_path / "out.mp4")

    assert "No such file" in str(exc_info.value)


def test_normalize_video_error_trims_long_stderr(tmp_path):
    """NormalizationError message is limited to the last 2 KB of stderr."""
    long_stderr = "x" * 5000
    fake_result = MagicMock()
    fake_result.returncode = 1
    fake_result.stderr = long_stderr
    fake_result.stdout = ""

    with patch("libs.video.ffmpeg.subprocess.run", return_value=fake_result):
        with pytest.raises(NormalizationError) as exc_info:
            normalize_video(tmp_path / "a.mp4", tmp_path / "out.mp4")

    # The message should be trimmed (not contain all 5000 'x' chars).
    assert len(str(exc_info.value)) < 4000


def test_normalize_video_succeeds_on_zero_exit(tmp_path):
    """normalize_video does not raise when ffmpeg exits 0."""
    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stderr = ""
    fake_result.stdout = ""

    with patch("libs.video.ffmpeg.subprocess.run", return_value=fake_result):
        # Should not raise
        normalize_video(tmp_path / "input.mp4", tmp_path / "out.mp4")


# ---------------------------------------------------------------------------
# normalize_video – command construction (cheap/safe defaults)
# ---------------------------------------------------------------------------


def _capture_cmd(tmp_path: Path, input_name: str = "input.mp4") -> list[str]:
    """Run normalize_video with a mocked subprocess and return the ffmpeg cmd."""
    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stderr = ""
    fake_result.stdout = ""

    captured: list[list[str]] = []

    def _fake_run(cmd, **kwargs):
        captured.append(cmd)
        return fake_result

    with patch("libs.video.ffmpeg.subprocess.run", side_effect=_fake_run):
        normalize_video(tmp_path / input_name, tmp_path / "out.mp4")

    return captured[0]


def test_normalize_video_uses_superfast_preset(tmp_path):
    cmd = _capture_cmd(tmp_path)
    assert _FFMPEG_PRESET in cmd


def test_normalize_video_uses_configured_crf(tmp_path):
    cmd = _capture_cmd(tmp_path)
    crf_idx = cmd.index("-crf")
    assert cmd[crf_idx + 1] == str(_FFMPEG_CRF)


def test_normalize_video_caps_threads(tmp_path):
    cmd = _capture_cmd(tmp_path)
    threads_idx = cmd.index("-threads")
    assert int(cmd[threads_idx + 1]) <= _FFMPEG_THREADS


def test_normalize_video_caps_fps(tmp_path):
    cmd = _capture_cmd(tmp_path)
    r_idx = cmd.index("-r")
    assert int(cmd[r_idx + 1]) <= _MAX_OUTPUT_FPS


def test_normalize_video_includes_scale_filter(tmp_path):
    cmd = _capture_cmd(tmp_path)
    vf_idx = cmd.index("-vf")
    scale_arg = cmd[vf_idx + 1]
    assert str(_MAX_OUTPUT_WIDTH) in scale_arg
    assert str(_MAX_OUTPUT_HEIGHT) in scale_arg


def test_normalize_video_uses_loglevel_error(tmp_path):
    cmd = _capture_cmd(tmp_path)
    assert "-loglevel" in cmd
    loglevel_idx = cmd.index("-loglevel")
    assert cmd[loglevel_idx + 1] == "error"


def test_normalize_video_maps_audio_optionally(tmp_path):
    """The ffmpeg command must include an optional audio map to handle no-audio inputs."""
    cmd = _capture_cmd(tmp_path)
    # Find all -map arguments
    maps = [cmd[i + 1] for i, v in enumerate(cmd) if v == "-map"]
    # Audio map must have '?' suffix (optional)
    assert any("a" in m and "?" in m for m in maps), f"No optional audio map found in: {maps}"


def test_normalize_video_accepts_mov_extension(tmp_path):
    """normalize_video accepts a .MOV input path without raising."""
    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stderr = ""
    fake_result.stdout = ""

    with patch("libs.video.ffmpeg.subprocess.run", return_value=fake_result):
        normalize_video(tmp_path / "IMG_0001.MOV", tmp_path / "out.mp4")


def test_normalize_video_accepts_tmp_extension(tmp_path):
    """normalize_video accepts a .tmp input (weird temp filename)."""
    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stderr = ""
    fake_result.stdout = ""

    with patch("libs.video.ffmpeg.subprocess.run", return_value=fake_result):
        normalize_video(tmp_path / "upload_abc123.tmp", tmp_path / "out.mp4")


def test_normalize_video_accepts_no_extension(tmp_path):
    """normalize_video accepts a path with no extension."""
    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stderr = ""
    fake_result.stdout = ""

    with patch("libs.video.ffmpeg.subprocess.run", return_value=fake_result):
        normalize_video(tmp_path / "video_file", tmp_path / "out.mp4")


# ---------------------------------------------------------------------------
# probe_media_streams
# ---------------------------------------------------------------------------


def _make_ffprobe_output(
    *,
    has_video: bool = True,
    has_audio: bool = True,
    video_codec: str = "h264",
    audio_codec: str = "aac",
    format_name: str = "mov,mp4,m4a,3gp,3g2,mj2",
) -> str:
    import json

    streams = []
    if has_video:
        streams.append({"codec_type": "video", "codec_name": video_codec})
    if has_audio:
        streams.append({"codec_type": "audio", "codec_name": audio_codec})
    return json.dumps({"streams": streams, "format": {"format_name": format_name}})


def _run_probe(ffprobe_output: str, returncode: int = 0) -> StreamInfo:
    fake_result = MagicMock()
    fake_result.returncode = returncode
    fake_result.stdout = ffprobe_output
    fake_result.stderr = ""

    with patch("libs.video.ffmpeg.subprocess.run", return_value=fake_result):
        return probe_media_streams("/fake/path.mov")


def test_probe_media_streams_detects_video_stream():
    info = _run_probe(_make_ffprobe_output(has_video=True))
    assert info["has_video"] is True


def test_probe_media_streams_detects_audio_stream():
    info = _run_probe(_make_ffprobe_output(has_audio=True))
    assert info["has_audio"] is True


def test_probe_media_streams_returns_video_codec():
    info = _run_probe(_make_ffprobe_output(video_codec="hevc"))
    assert info["video_codec"] == "hevc"


def test_probe_media_streams_returns_audio_codec():
    info = _run_probe(_make_ffprobe_output(audio_codec="aac"))
    assert info["audio_codec"] == "aac"


def test_probe_media_streams_returns_container_format():
    info = _run_probe(_make_ffprobe_output(format_name="mov,mp4,m4a,3gp,3g2,mj2"))
    assert "mov" in info["container_format"]


def test_probe_media_streams_no_video_stream():
    info = _run_probe(_make_ffprobe_output(has_video=False, has_audio=True))
    assert info["has_video"] is False
    assert info["video_codec"] is None


def test_probe_media_streams_no_audio_stream():
    info = _run_probe(_make_ffprobe_output(has_video=True, has_audio=False))
    assert info["has_audio"] is False
    assert info["audio_codec"] is None


def test_probe_media_streams_raises_on_nonzero_exit():
    fake_result = MagicMock()
    fake_result.returncode = 1
    fake_result.stdout = ""
    fake_result.stderr = "No such file"

    with patch("libs.video.ffmpeg.subprocess.run", return_value=fake_result):
        with pytest.raises(subprocess.CalledProcessError):
            probe_media_streams("/nonexistent/file.mov")


def test_probe_media_streams_independent_of_extension():
    """probe_media_streams works the same regardless of file extension."""
    # Same ffprobe output, different file extensions → same result.
    output = _make_ffprobe_output(video_codec="hevc", audio_codec="aac")

    results = []
    for _ext in [".MOV", ".mov", ".mp4", ".tmp", ".bin", ""]:
        results.append(_run_probe(output))

    video_codecs = {r["video_codec"] for r in results}
    assert video_codecs == {"hevc"}, "All extensions should yield same video_codec"


def test_probe_media_streams_detects_iphone_hevc():
    """Typical iPhone MOV (HEVC video + AAC audio) is correctly identified."""
    output = _make_ffprobe_output(
        video_codec="hevc",
        audio_codec="aac",
        format_name="mov,mp4,m4a,3gp,3g2,mj2",
    )
    info = _run_probe(output)
    assert info["has_video"] is True
    assert info["video_codec"] == "hevc"
    assert info["has_audio"] is True


# ---------------------------------------------------------------------------
# Worker – pre-normalization probe integration
# ---------------------------------------------------------------------------


def _make_worker_db(tmp_path: Path, *, source_name: str = "input.mp4"):
    """Return mock DB + fake video/job for worker tests."""
    fake_video = MagicMock()
    fake_video.id = 1
    fake_video.source_path = str(tmp_path / source_name)
    fake_video.status = None
    fake_video.project_id = 1

    (tmp_path / source_name).write_bytes(b"fake video bytes")

    fake_job = MagicMock()
    fake_job.id = 1
    fake_job.status = None
    fake_job.error = None
    fake_job.processing_run_id = 1

    fake_run = MagicMock()
    fake_run.id = 1
    fake_run.status = None
    fake_run.error = None
    fake_run.started_at = None
    fake_run.completed_at = None
    fake_run.pipeline_version = None

    def _get(model, pk):
        name = model.__name__
        if name == "Job":
            return fake_job
        if name == "ProcessingRun":
            return fake_run
        return fake_video

    mock_db = AsyncMock()
    mock_db.get = AsyncMock(side_effect=_get)
    mock_db.add = MagicMock()
    mock_db.commit = AsyncMock()
    mock_db.__aenter__ = AsyncMock(return_value=mock_db)
    mock_db.__aexit__ = AsyncMock(return_value=False)

    return mock_db, fake_video, fake_job, fake_run


def _patch_worker_settings(mock_settings, tmp_path):
    mock_settings.normalized_dir = str(tmp_path / "normalized")
    mock_settings.artifacts_dir = str(tmp_path / "artifacts")
    mock_settings.frame_sample_fps = 2.0
    mock_settings.detector_backend = "stub"
    mock_settings.detector_model = "yolov8n.pt"
    mock_settings.tracker_backend = "stub"
    mock_settings.tracker_iou_threshold = 0.3
    mock_settings.tracker_max_age = 30
    mock_settings.pose_backend = "stub"
    mock_settings.pose_min_confidence = 0.3
    mock_settings.storage_backend = "local"


def _make_fake_pipeline_return():
    state = {
        "video_id": "1", "version": 7, "segments": [], "tracks": [], "features": [],
        "detections_summary": {"frame_count": 0, "frames_with_people": 0, "total_detections": 0},
        "tracking_summary": {"track_count": 0, "tracked_frame_count": 0,
                             "average_detections_per_frame": 0.0},
        "pose_summary": {"pose_count": 0, "posed_track_count": 0,
                         "average_keypoints_per_pose": 0.0},
        "feature_summary": {"feature_count": 0, "featured_track_count": 0, "feature_names": []},
        "segmentation_summary": {"segment_count": 0, "segment_labels": [],
                                 "total_segment_duration_ms": 0},
        "clip_summary": {"clip_count": 0, "total_clip_duration_ms": 0},
        "manifest_path": "",
        "notes": "",
    }
    detections = {"video_id": "1", "version": 1, "sample_fps": 2.0, "frames": []}
    tracks = {"video_id": "1", "version": 1, "track_count": 0, "tracks": []}
    poses = {"video_id": "1", "version": 1, "pose_count": 0, "poses": []}
    features = {"video_id": "1", "version": 1, "feature_count": 0, "features": []}
    segments = {"video_id": "1", "version": 1, "segment_count": 0, "segments": []}
    return state, detections, tracks, poses, features, segments


@pytest.mark.asyncio
async def test_worker_raises_clear_error_when_no_video_stream(tmp_path):
    """Worker raises ValueError with clear message when source has no video stream."""
    mock_db, _, fake_job, _ = _make_worker_db(tmp_path)
    no_video_info = StreamInfo(
        has_video=False, has_audio=True,
        video_codec=None, audio_codec="aac",
        container_format="mov,mp4,m4a,3gp,3g2,mj2",
    )

    with (
        patch("apps.worker.jobs.process_video.AsyncSessionLocal", return_value=mock_db),
        patch(
            "apps.worker.jobs.process_video.probe_media_streams",
            return_value=no_video_info,
        ),
        patch("apps.worker.jobs.process_video.normalize_video"),
        patch("apps.worker.jobs.process_video.settings") as mock_settings,
    ):
        _patch_worker_settings(mock_settings, tmp_path)
        from apps.worker.jobs.process_video import handle_process_video

        with pytest.raises(ValueError, match="no video stream"):
            await handle_process_video({"job_id": 1, "payload": {"video_id": 1}})


@pytest.mark.asyncio
async def test_worker_surfaces_normalization_error(tmp_path):
    """Worker re-raises NormalizationError so caller can see returncode context."""
    mock_db, _, fake_job, _ = _make_worker_db(tmp_path)
    valid_info = StreamInfo(
        has_video=True, has_audio=True,
        video_codec="h264", audio_codec="aac",
        container_format="mov,mp4,m4a,3gp,3g2,mj2",
    )

    with (
        patch("apps.worker.jobs.process_video.AsyncSessionLocal", return_value=mock_db),
        patch(
            "apps.worker.jobs.process_video.probe_media_streams",
            return_value=valid_info,
        ),
        patch(
            "apps.worker.jobs.process_video.normalize_video",
            side_effect=NormalizationError("ffmpeg exited 1: bad input", 1, "bad input"),
        ),
        patch("apps.worker.jobs.process_video.settings") as mock_settings,
    ):
        _patch_worker_settings(mock_settings, tmp_path)
        from apps.worker.jobs.process_video import handle_process_video

        with pytest.raises(NormalizationError):
            await handle_process_video({"job_id": 1, "payload": {"video_id": 1}})


@pytest.mark.asyncio
async def test_worker_sets_job_error_on_normalization_failure(tmp_path):
    """When normalization fails, job.error is set to the NormalizationError message."""
    mock_db, _, fake_job, _ = _make_worker_db(tmp_path)
    valid_info = StreamInfo(
        has_video=True, has_audio=False,
        video_codec="hevc", audio_codec=None,
        container_format="mov,mp4,m4a,3gp,3g2,mj2",
    )

    with (
        patch("apps.worker.jobs.process_video.AsyncSessionLocal", return_value=mock_db),
        patch(
            "apps.worker.jobs.process_video.probe_media_streams",
            return_value=valid_info,
        ),
        patch(
            "apps.worker.jobs.process_video.normalize_video",
            side_effect=NormalizationError("ffmpeg exited 1: codec error", 1, "codec error"),
        ),
        patch("apps.worker.jobs.process_video.settings") as mock_settings,
    ):
        _patch_worker_settings(mock_settings, tmp_path)
        from apps.worker.jobs.process_video import handle_process_video

        with pytest.raises(NormalizationError):
            await handle_process_video({"job_id": 1, "payload": {"video_id": 1}})

    assert "ffmpeg exited 1" in str(fake_job.error)


@pytest.mark.asyncio
async def test_worker_processes_mov_source_filename(tmp_path):
    """Worker succeeds when the source path has a .MOV extension (iPhone video)."""
    mock_db, _, _, _ = _make_worker_db(tmp_path, source_name="IMG_0001.MOV")
    valid_info = StreamInfo(
        has_video=True, has_audio=True,
        video_codec="hevc", audio_codec="aac",
        container_format="mov,mp4,m4a,3gp,3g2,mj2",
    )

    with (
        patch("apps.worker.jobs.process_video.AsyncSessionLocal", return_value=mock_db),
        patch(
            "apps.worker.jobs.process_video.probe_media_streams",
            return_value=valid_info,
        ),
        patch("apps.worker.jobs.process_video.normalize_video"),
        patch(
            "apps.worker.jobs.process_video.probe_video",
            return_value={"duration_seconds": 3.0, "fps": 30.0, "width": 720, "height": 1280},
        ),
        patch("apps.worker.jobs.process_video.extract_frames", return_value=[]),
        patch("apps.worker.jobs.process_video.generate_clips", return_value=[]),
        patch(
            "apps.worker.jobs.process_video.run_pipeline",
            return_value=_make_fake_pipeline_return(),
        ),
        patch("apps.worker.jobs.process_video.settings") as mock_settings,
    ):
        _patch_worker_settings(mock_settings, tmp_path)
        from apps.worker.jobs.process_video import handle_process_video

        # Should not raise
        await handle_process_video({"job_id": 1, "payload": {"video_id": 1}})


@pytest.mark.asyncio
async def test_worker_processes_tmp_source_filename(tmp_path):
    """Worker succeeds when source has a .tmp extension (share-sheet temp filename)."""
    mock_db, _, _, _ = _make_worker_db(tmp_path, source_name="upload_abc123.tmp")
    valid_info = StreamInfo(
        has_video=True, has_audio=True,
        video_codec="h264", audio_codec="aac",
        container_format="mov,mp4,m4a,3gp,3g2,mj2",
    )

    with (
        patch("apps.worker.jobs.process_video.AsyncSessionLocal", return_value=mock_db),
        patch(
            "apps.worker.jobs.process_video.probe_media_streams",
            return_value=valid_info,
        ),
        patch("apps.worker.jobs.process_video.normalize_video"),
        patch(
            "apps.worker.jobs.process_video.probe_video",
            return_value={"duration_seconds": 2.0, "fps": 30.0, "width": 1280, "height": 720},
        ),
        patch("apps.worker.jobs.process_video.extract_frames", return_value=[]),
        patch("apps.worker.jobs.process_video.generate_clips", return_value=[]),
        patch(
            "apps.worker.jobs.process_video.run_pipeline",
            return_value=_make_fake_pipeline_return(),
        ),
        patch("apps.worker.jobs.process_video.settings") as mock_settings,
    ):
        _patch_worker_settings(mock_settings, tmp_path)
        from apps.worker.jobs.process_video import handle_process_video

        await handle_process_video({"job_id": 1, "payload": {"video_id": 1}})

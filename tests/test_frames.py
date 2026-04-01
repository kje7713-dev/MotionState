"""Focused tests for libs/video/frames.py."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from libs.video.frames import FrameMeta, extract_frames


def _write_fake_frames(directory: Path, count: int) -> None:
    """Write *count* minimal JPEG files to *directory*, named frame_NNNNNN.jpg."""
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (directory / f"frame_{i:06d}.jpg").write_bytes(b"\xff\xd8\xff\xe0")


class TestExtractFramesMetadata:
    """extract_frames returns correctly ordered FrameMeta."""

    def test_returns_frame_meta_instances(self, tmp_path):
        frames_dir = tmp_path / "frames"
        _write_fake_frames(frames_dir, 2)
        with patch("libs.video.frames.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = extract_frames("/video.mp4", frames_dir, sample_fps=1.0)
        assert all(isinstance(f, FrameMeta) for f in result)

    def test_returns_correct_count(self, tmp_path):
        frames_dir = tmp_path / "frames"
        _write_fake_frames(frames_dir, 5)
        with patch("libs.video.frames.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = extract_frames("/video.mp4", frames_dir, sample_fps=2.0)
        assert len(result) == 5

    def test_frame_index_is_zero_based(self, tmp_path):
        frames_dir = tmp_path / "frames"
        _write_fake_frames(frames_dir, 3)
        with patch("libs.video.frames.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = extract_frames("/video.mp4", frames_dir, sample_fps=1.0)
        assert [f.frame_index for f in result] == [0, 1, 2]

    def test_frames_returned_in_ascending_order(self, tmp_path):
        frames_dir = tmp_path / "frames"
        _write_fake_frames(frames_dir, 4)
        with patch("libs.video.frames.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = extract_frames("/video.mp4", frames_dir, sample_fps=2.0)
        indices = [f.frame_index for f in result]
        assert indices == sorted(indices)

    def test_path_points_into_output_dir(self, tmp_path):
        frames_dir = tmp_path / "frames"
        _write_fake_frames(frames_dir, 2)
        with patch("libs.video.frames.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = extract_frames("/video.mp4", frames_dir, sample_fps=1.0)
        for meta in result:
            assert Path(meta.path).parent == frames_dir


class TestExtractFramesTimestamps:
    """Timestamp computation matches expected intervals."""

    def test_first_frame_timestamp_is_zero(self, tmp_path):
        frames_dir = tmp_path / "frames"
        _write_fake_frames(frames_dir, 1)
        with patch("libs.video.frames.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = extract_frames("/video.mp4", frames_dir, sample_fps=2.0)
        assert result[0].timestamp_ms == pytest.approx(0.0)

    def test_timestamp_interval_at_2fps(self, tmp_path):
        """At 2 FPS the interval is 500 ms."""
        frames_dir = tmp_path / "frames"
        _write_fake_frames(frames_dir, 3)
        with patch("libs.video.frames.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = extract_frames("/video.mp4", frames_dir, sample_fps=2.0)
        assert result[0].timestamp_ms == pytest.approx(0.0)
        assert result[1].timestamp_ms == pytest.approx(500.0)
        assert result[2].timestamp_ms == pytest.approx(1000.0)

    def test_timestamp_interval_at_1fps(self, tmp_path):
        """At 1 FPS the interval is 1000 ms."""
        frames_dir = tmp_path / "frames"
        _write_fake_frames(frames_dir, 3)
        with patch("libs.video.frames.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = extract_frames("/video.mp4", frames_dir, sample_fps=1.0)
        assert result[1].timestamp_ms == pytest.approx(1000.0)
        assert result[2].timestamp_ms == pytest.approx(2000.0)

    def test_timestamp_interval_at_5fps(self, tmp_path):
        """At 5 FPS the interval is 200 ms."""
        frames_dir = tmp_path / "frames"
        _write_fake_frames(frames_dir, 2)
        with patch("libs.video.frames.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = extract_frames("/video.mp4", frames_dir, sample_fps=5.0)
        assert result[1].timestamp_ms == pytest.approx(200.0)


class TestExtractFramesFileSystem:
    """extract_frames interacts with the filesystem correctly."""

    def test_creates_output_directory(self, tmp_path):
        new_dir = tmp_path / "nested" / "frames"
        assert not new_dir.exists()
        with patch("libs.video.frames.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            extract_frames("/video.mp4", new_dir, sample_fps=1.0)
        assert new_dir.exists()

    def test_returns_empty_list_when_no_frames_written(self, tmp_path):
        """Returns [] when ffmpeg exits cleanly but writes no frames."""
        frames_dir = tmp_path / "frames"
        with patch("libs.video.frames.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = extract_frames("/video.mp4", frames_dir, sample_fps=2.0)
        assert result == []

    def test_raises_on_ffmpeg_failure(self, tmp_path):
        """CalledProcessError is raised when ffmpeg exits non-zero."""
        frames_dir = tmp_path / "frames"
        with patch("libs.video.frames.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
            with pytest.raises(subprocess.CalledProcessError):
                extract_frames("/video.mp4", frames_dir, sample_fps=2.0)

"""Tests for libs/video/clips.py – clip generation helper."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# generate_clip
# ---------------------------------------------------------------------------


class TestGenerateClip:
    def test_calls_ffmpeg_with_correct_seek_and_duration(self, tmp_path):
        """generate_clip builds the expected ffmpeg command."""
        from libs.video.clips import generate_clip

        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")
        out = tmp_path / "clip.mp4"

        with patch("libs.video.clips.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            generate_clip(video, start_ms=1000.0, end_ms=3500.0, output_path=out)

        cmd = mock_run.call_args[0][0]
        assert "ffmpeg" in cmd
        assert "-ss" in cmd
        assert "1.0" in cmd  # start_sec = 1000 / 1000
        assert "-t" in cmd
        assert "2.5" in cmd  # duration_sec = (3500 - 1000) / 1000
        assert str(out) in cmd

    def test_raises_on_ffmpeg_failure(self, tmp_path):
        """generate_clip raises CalledProcessError if ffmpeg exits non-zero."""
        from libs.video.clips import generate_clip

        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")
        out = tmp_path / "clip.mp4"

        with patch("libs.video.clips.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
            with __import__("pytest").raises(subprocess.CalledProcessError):
                generate_clip(video, start_ms=0.0, end_ms=1000.0, output_path=out)

    def test_creates_parent_directory(self, tmp_path):
        """generate_clip creates the output directory if it does not exist."""
        from libs.video.clips import generate_clip

        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")
        out = tmp_path / "clips" / "sub" / "clip.mp4"

        with patch("libs.video.clips.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            generate_clip(video, start_ms=0.0, end_ms=500.0, output_path=out)

        assert out.parent.exists()

    def test_zero_duration_segment_is_handled(self, tmp_path):
        """generate_clip does not crash when start_ms == end_ms."""
        from libs.video.clips import generate_clip

        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")
        out = tmp_path / "clip.mp4"

        with patch("libs.video.clips.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            generate_clip(video, start_ms=500.0, end_ms=500.0, output_path=out)

        cmd = mock_run.call_args[0][0]
        assert "-t" in cmd
        t_idx = cmd.index("-t")
        assert float(cmd[t_idx + 1]) == 0.0


# ---------------------------------------------------------------------------
# generate_clips
# ---------------------------------------------------------------------------


class TestGenerateClips:
    def _make_segments(self) -> list[dict]:
        return [
            {"start_ms": 0.0, "end_ms": 2500.0, "label": "low_motion", "confidence": 0.88},
            {"start_ms": 2500.0, "end_ms": 5000.0, "label": "active_motion", "confidence": 0.92},
        ]

    def test_returns_one_clip_per_segment(self, tmp_path):
        """generate_clips returns as many clip dicts as there are segments."""
        from libs.video.clips import generate_clips

        with patch("libs.video.clips.generate_clip"):
            clips = generate_clips(tmp_path / "v.mp4", self._make_segments(), tmp_path / "clips")

        assert len(clips) == 2

    def test_filenames_are_deterministic(self, tmp_path):
        """Clip filenames follow segment_{index:03d}_{label}.mp4 convention."""
        from libs.video.clips import generate_clips

        with patch("libs.video.clips.generate_clip"):
            clips = generate_clips(tmp_path / "v.mp4", self._make_segments(), tmp_path / "clips")

        assert Path(clips[0]["path"]).name == "segment_000_low_motion.mp4"
        assert Path(clips[1]["path"]).name == "segment_001_active_motion.mp4"

    def test_clip_metadata_has_required_fields(self, tmp_path):
        """Each returned clip dict contains segment_index, label, start_ms, end_ms, path."""
        from libs.video.clips import generate_clips

        with patch("libs.video.clips.generate_clip"):
            clips = generate_clips(tmp_path / "v.mp4", self._make_segments(), tmp_path / "clips")

        for clip in clips:
            assert "segment_index" in clip
            assert "label" in clip
            assert "start_ms" in clip
            assert "end_ms" in clip
            assert "path" in clip

    def test_segment_index_matches_position(self, tmp_path):
        """segment_index in each clip dict matches its position in the list."""
        from libs.video.clips import generate_clips

        with patch("libs.video.clips.generate_clip"):
            clips = generate_clips(tmp_path / "v.mp4", self._make_segments(), tmp_path / "clips")

        assert clips[0]["segment_index"] == 0
        assert clips[1]["segment_index"] == 1

    def test_empty_segments_returns_empty_list(self, tmp_path):
        """generate_clips with no segments returns an empty list."""
        from libs.video.clips import generate_clips

        with patch("libs.video.clips.generate_clip"):
            clips = generate_clips(tmp_path / "v.mp4", [], tmp_path / "clips")

        assert clips == []

    def test_creates_output_directory(self, tmp_path):
        """generate_clips creates the output directory if it does not exist."""
        from libs.video.clips import generate_clips

        clips_dir = tmp_path / "new_dir" / "clips"
        assert not clips_dir.exists()

        with patch("libs.video.clips.generate_clip"):
            generate_clips(tmp_path / "v.mp4", [], clips_dir)

        assert clips_dir.exists()

    def test_start_ms_end_ms_preserved_in_output(self, tmp_path):
        """start_ms and end_ms from the segment are preserved in the clip dict."""
        from libs.video.clips import generate_clips

        with patch("libs.video.clips.generate_clip"):
            clips = generate_clips(tmp_path / "v.mp4", self._make_segments(), tmp_path / "clips")

        assert clips[0]["start_ms"] == 0.0
        assert clips[0]["end_ms"] == 2500.0
        assert clips[1]["start_ms"] == 2500.0
        assert clips[1]["end_ms"] == 5000.0

    def test_calls_generate_clip_for_each_segment(self, tmp_path):
        """generate_clips invokes generate_clip once per segment."""
        from libs.video.clips import generate_clips

        with patch("libs.video.clips.generate_clip") as mock_gc:
            generate_clips(tmp_path / "v.mp4", self._make_segments(), tmp_path / "clips")

        assert mock_gc.call_count == 2

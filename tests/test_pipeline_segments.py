"""Pipeline-level tests for temporal segmentation.

These tests exercise the full run_pipeline path with respect to segmentation:
segments artifact structure, state artifact segmentation_summary, and
integration with the BasicSegmenter.
"""


from libs.pipeline.contracts import (
    BoundingBox,
    Detection,
    Detector,
    Frame,
    Keypoint,
    PoseEstimate,
    PoseEstimator,
    Track,
)
from libs.pipeline.run_pipeline import run_pipeline
from libs.pipeline.segments_basic import BasicSegmenter
from libs.video.frames import FrameMeta

# ---------------------------------------------------------------------------
# Minimal mock detector and pose estimator for controlled feature generation
# ---------------------------------------------------------------------------


class _SinglePersonDetector(Detector):
    """Always returns one person detection per frame."""

    def detect(self, frame: Frame) -> list[Detection]:
        return [
            Detection(
                frame_index=frame.index,
                bbox=BoundingBox(x=10.0, y=10.0, width=50.0, height=100.0, confidence=0.9),
                class_label="person",
            )
        ]


class _StillPersonPose(PoseEstimator):
    """Returns a minimal pose with a single keypoint (still person)."""

    def estimate(self, frame: Frame, tracks: list[Track]) -> list[PoseEstimate]:
        return [
            PoseEstimate(
                frame_index=frame.index,
                track_id=t.track_id,
                keypoints=[Keypoint("nose", 50.0, 30.0, 1.0)],
            )
            for t in tracks
        ]


def _make_frame(tmp_path, index: int, timestamp_ms: float = 0.0) -> FrameMeta:
    path = tmp_path / f"frame_{index:06d}.jpg"
    path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
    return FrameMeta(frame_index=index, timestamp_ms=timestamp_ms, path=str(path))


# ---------------------------------------------------------------------------
# Segments artifact structure
# ---------------------------------------------------------------------------


class TestSegmentsArtifactStructure:
    def test_run_pipeline_returns_six_tuple(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        result = run_pipeline("v", frames=[frame], detector=_SinglePersonDetector())
        assert len(result) == 6

    def test_segments_artifact_has_required_keys(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        *_, segments = run_pipeline("v", frames=[frame], detector=_SinglePersonDetector())
        assert "video_id" in segments
        assert "version" in segments
        assert "segment_count" in segments
        assert "segments" in segments

    def test_segments_artifact_video_id_matches(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        *_, segments = run_pipeline("42", frames=[frame], detector=_SinglePersonDetector())
        assert segments["video_id"] == "42"

    def test_segments_artifact_version_is_1(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        *_, segments = run_pipeline("v", frames=[frame], detector=_SinglePersonDetector())
        assert segments["version"] == 1

    def test_segments_artifact_segment_count_matches_list_length(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        *_, segments = run_pipeline("v", frames=[frame], detector=_SinglePersonDetector())
        assert segments["segment_count"] == len(segments["segments"])

    def test_segment_item_has_required_fields(self, tmp_path):
        """Each segment dict must include start_ms, end_ms, label, confidence, metadata."""
        frames = [
            _make_frame(tmp_path, 0, timestamp_ms=0.0),
            _make_frame(tmp_path, 1, timestamp_ms=500.0),
        ]
        *_, segments = run_pipeline(
            "v",
            frames=frames,
            detector=_SinglePersonDetector(),
            pose_estimator=_StillPersonPose(),
        )
        for seg in segments["segments"]:
            assert "start_ms" in seg
            assert "end_ms" in seg
            assert "label" in seg
            assert "confidence" in seg
            assert "metadata" in seg

    def test_empty_frames_gives_empty_segments_list(self):
        *_, segments = run_pipeline("v", frames=[])
        assert segments["segments"] == []
        assert segments["segment_count"] == 0


# ---------------------------------------------------------------------------
# State artifact segmentation_summary
# ---------------------------------------------------------------------------


class TestStateSegmentationSummary:
    def test_state_version_is_six(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        state, *_ = run_pipeline("v", frames=[frame], detector=_SinglePersonDetector())
        assert state["version"] == 7

    def test_state_has_segmentation_summary(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        state, *_ = run_pipeline("v", frames=[frame], detector=_SinglePersonDetector())
        assert "segmentation_summary" in state

    def test_segmentation_summary_has_required_fields(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        state, *_ = run_pipeline("v", frames=[frame], detector=_SinglePersonDetector())
        summary = state["segmentation_summary"]
        assert "segment_count" in summary
        assert "segment_labels" in summary
        assert "total_segment_duration_ms" in summary

    def test_segmentation_summary_segment_count_non_negative(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        state, *_ = run_pipeline("v", frames=[frame], detector=_SinglePersonDetector())
        assert state["segmentation_summary"]["segment_count"] >= 0

    def test_segmentation_summary_segment_labels_is_list(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        state, *_ = run_pipeline("v", frames=[frame], detector=_SinglePersonDetector())
        assert isinstance(state["segmentation_summary"]["segment_labels"], list)

    def test_segmentation_summary_total_duration_non_negative(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        state, *_ = run_pipeline("v", frames=[frame], detector=_SinglePersonDetector())
        assert state["segmentation_summary"]["total_segment_duration_ms"] >= 0

    def test_state_notes_mention_temporal_segmentation(self, tmp_path):
        frame = _make_frame(tmp_path, 0)
        state, *_ = run_pipeline("v", frames=[frame], detector=_SinglePersonDetector())
        assert "temporal segmentation" in state["notes"]

    def test_empty_frames_segmentation_summary_zeroed(self):
        state, *_ = run_pipeline("v", frames=[])
        summary = state["segmentation_summary"]
        assert summary["segment_count"] == 0
        assert summary["segment_labels"] == []
        assert summary["total_segment_duration_ms"] == 0


# ---------------------------------------------------------------------------
# BasicSegmenter integration via run_pipeline
# ---------------------------------------------------------------------------


class TestBasicSegmenterIntegration:
    def test_custom_segmenter_used_when_provided(self, tmp_path):
        """Passing a custom segmenter instance is honoured by run_pipeline."""
        from libs.pipeline.contracts import MotionFeature, Segment, Segmenter

        class _CountingSegmenter(Segmenter):
            called = False

            def segment(self, features: list[MotionFeature]) -> list[Segment]:
                _CountingSegmenter.called = True
                return []

        frame = _make_frame(tmp_path, 0)
        run_pipeline("v", frames=[frame], segmenter=_CountingSegmenter())
        assert _CountingSegmenter.called

    def test_segments_labels_are_domain_agnostic(self, tmp_path):
        """Labels produced by the default segmenter contain no domain words."""
        domain_words = {"bjj", "guard", "pass", "scramble", "stance", "kick"}
        frames = [_make_frame(tmp_path, i, timestamp_ms=float(i * 500)) for i in range(6)]
        *_, segments = run_pipeline(
            "v",
            frames=frames,
            detector=_SinglePersonDetector(),
            pose_estimator=_StillPersonPose(),
        )
        for seg in segments["segments"]:
            label = seg["label"].lower()
            for word in domain_words:
                assert word not in label

    def test_valid_labels_only(self, tmp_path):
        """All segment labels must be from the allowed generic set."""
        valid_labels = {"low_motion", "active_motion", "transition_window", "sparse_data"}
        frames = [_make_frame(tmp_path, i, timestamp_ms=float(i * 500)) for i in range(4)]
        *_, segments = run_pipeline(
            "v",
            frames=frames,
            detector=_SinglePersonDetector(),
            pose_estimator=_StillPersonPose(),
        )
        for seg in segments["segments"]:
            assert seg["label"] in valid_labels

    def test_pipeline_with_explicit_basic_segmenter(self, tmp_path):
        """Explicitly passing BasicSegmenter produces the same result as the default."""
        frames = [_make_frame(tmp_path, i, timestamp_ms=float(i * 500)) for i in range(4)]

        result_default = run_pipeline(
            "v",
            frames=frames,
            detector=_SinglePersonDetector(),
            pose_estimator=_StillPersonPose(),
        )
        result_explicit = run_pipeline(
            "v",
            frames=frames,
            detector=_SinglePersonDetector(),
            pose_estimator=_StillPersonPose(),
            segmenter=BasicSegmenter(),
        )

        # Both should produce the same number of segments.
        assert result_default[5]["segment_count"] == result_explicit[5]["segment_count"]

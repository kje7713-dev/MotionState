"""Tests for usage event emission and aggregation (libs/usage.py).

Covers:
- emit() appends a UsageEvent row with correct fields
- monthly_totals() sums only events within the target month
- alltime_totals() sums all events regardless of date
- latest_storage_bytes() aggregates only storage_bytes_written events
- project_usage_summary() returns a structured dict
- storage bytes are populated for artifact writes in the worker
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from libs.models import UsageEvent, UsageEventType

# Minimal valid StreamInfo to satisfy the pre-normalization probe in the worker.
_VALID_SRC_INFO = {
    "has_video": True,
    "has_audio": True,
    "video_codec": "h264",
    "audio_codec": "aac",
    "container_format": "mov,mp4,m4a,3gp,3g2,mj2",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(rows: list | None = None):
    """Return a mock AsyncSession suitable for usage helpers."""
    session = AsyncMock()

    added_objects: list = []
    session.add = MagicMock(side_effect=added_objects.append)
    session.flush = AsyncMock()
    session.commit = AsyncMock()

    # Build a mock result that returns the given rows for aggregate queries.
    mock_result = MagicMock()
    mock_result.all.return_value = rows or []
    mock_result.scalar.return_value = None
    session.execute = AsyncMock(return_value=mock_result)

    return session, added_objects


# ---------------------------------------------------------------------------
# emit()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emit_adds_usage_event_row():
    """emit() appends a UsageEvent row to the session."""
    from libs.usage import emit

    session, added = _make_session()
    await emit(
        session,
        project_id=1,
        event_type=UsageEventType.videos_uploaded,
        quantity=1,
    )
    assert len(added) == 1
    assert isinstance(added[0], UsageEvent)


@pytest.mark.asyncio
async def test_emit_sets_correct_event_type():
    """emit() stores the correct event_type string."""
    from libs.usage import emit

    session, added = _make_session()
    await emit(
        session,
        project_id=2,
        event_type=UsageEventType.frames_extracted,
        quantity=42,
    )
    assert added[0].event_type == "frames_extracted"


@pytest.mark.asyncio
async def test_emit_sets_quantity():
    """emit() stores the supplied quantity."""
    from libs.usage import emit

    session, added = _make_session()
    await emit(
        session,
        project_id=1,
        event_type=UsageEventType.storage_bytes_written,
        quantity=1024,
    )
    assert added[0].quantity == 1024


@pytest.mark.asyncio
async def test_emit_sets_unit_from_lookup():
    """emit() derives the unit from the event type."""
    from libs.usage import emit

    session, added = _make_session()
    await emit(
        session,
        project_id=1,
        event_type=UsageEventType.video_seconds_processed,
        quantity=30,
    )
    assert added[0].unit == "seconds"


@pytest.mark.asyncio
async def test_emit_sets_storage_unit():
    """emit() uses 'bytes' as the unit for storage_bytes_written."""
    from libs.usage import emit

    session, added = _make_session()
    await emit(
        session,
        project_id=1,
        event_type=UsageEventType.storage_bytes_written,
        quantity=2048,
    )
    assert added[0].unit == "bytes"


@pytest.mark.asyncio
async def test_emit_stores_processing_run_id():
    """emit() links the event to a processing_run_id when supplied."""
    from libs.usage import emit

    session, added = _make_session()
    await emit(
        session,
        project_id=1,
        event_type=UsageEventType.clips_generated,
        quantity=3,
        processing_run_id=99,
    )
    assert added[0].processing_run_id == 99


@pytest.mark.asyncio
async def test_emit_stores_metadata():
    """emit() stores arbitrary metadata as metadata_json."""
    from libs.usage import emit

    session, added = _make_session()
    meta = {"video_id": 5, "note": "test"}
    await emit(
        session,
        project_id=1,
        event_type=UsageEventType.api_reads,
        quantity=1,
        metadata=meta,
    )
    assert added[0].metadata_json == meta


@pytest.mark.asyncio
async def test_emit_swallows_exceptions():
    """emit() does not raise when the DB operation fails."""
    from libs.usage import emit

    session = AsyncMock()
    session.add = MagicMock(side_effect=RuntimeError("db down"))
    session.flush = AsyncMock()

    # Should not raise.
    await emit(session, project_id=1, event_type=UsageEventType.api_reads, quantity=1)


# ---------------------------------------------------------------------------
# monthly_totals()
# ---------------------------------------------------------------------------


class _AggRow:
    def __init__(self, event_type: str, total: int):
        self.event_type = event_type
        self.total = total


@pytest.mark.asyncio
async def test_monthly_totals_returns_dict():
    """monthly_totals() returns a dict keyed by event_type."""
    from libs.usage import monthly_totals

    rows = [
        _AggRow("videos_uploaded", 5),
        _AggRow("frames_extracted", 100),
    ]
    mock_result = MagicMock()
    mock_result.all.return_value = rows
    session = AsyncMock()
    session.execute = AsyncMock(return_value=mock_result)

    result = await monthly_totals(session, project_id=1)
    assert result == {"videos_uploaded": 5, "frames_extracted": 100}


@pytest.mark.asyncio
async def test_monthly_totals_empty_when_no_events():
    """monthly_totals() returns {} when there are no events."""
    from libs.usage import monthly_totals

    mock_result = MagicMock()
    mock_result.all.return_value = []
    session = AsyncMock()
    session.execute = AsyncMock(return_value=mock_result)

    result = await monthly_totals(session, project_id=1)
    assert result == {}


# ---------------------------------------------------------------------------
# alltime_totals()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_alltime_totals_returns_all_event_types():
    """alltime_totals() returns totals for every event type present."""
    from libs.usage import alltime_totals

    rows = [
        _AggRow("videos_uploaded", 10),
        _AggRow("storage_bytes_written", 99999),
        _AggRow("api_reads", 250),
    ]
    mock_result = MagicMock()
    mock_result.all.return_value = rows
    session = AsyncMock()
    session.execute = AsyncMock(return_value=mock_result)

    result = await alltime_totals(session, project_id=1)
    assert result["storage_bytes_written"] == 99999
    assert result["api_reads"] == 250


# ---------------------------------------------------------------------------
# latest_storage_bytes()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_latest_storage_bytes_returns_sum():
    """latest_storage_bytes() returns the aggregated storage total."""
    from libs.usage import latest_storage_bytes

    mock_result = MagicMock()
    mock_result.scalar.return_value = 512000
    session = AsyncMock()
    session.execute = AsyncMock(return_value=mock_result)

    total = await latest_storage_bytes(session, project_id=1)
    assert total == 512000


@pytest.mark.asyncio
async def test_latest_storage_bytes_returns_zero_when_none():
    """latest_storage_bytes() returns 0 when there are no storage events."""
    from libs.usage import latest_storage_bytes

    mock_result = MagicMock()
    mock_result.scalar.return_value = None
    session = AsyncMock()
    session.execute = AsyncMock(return_value=mock_result)

    total = await latest_storage_bytes(session, project_id=1)
    assert total == 0


# ---------------------------------------------------------------------------
# project_usage_summary()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_project_usage_summary_shape():
    """project_usage_summary() returns a dict with expected top-level keys."""
    from libs.usage import project_usage_summary

    # We'll mock monthly_totals / alltime_totals / latest_storage_bytes.
    with (
        patch("libs.usage.monthly_totals", new=AsyncMock(return_value={"videos_uploaded": 2})),
        patch("libs.usage.alltime_totals", new=AsyncMock(return_value={"videos_uploaded": 10})),
        patch("libs.usage.latest_storage_bytes", new=AsyncMock(return_value=1024)),
    ):
        session = AsyncMock()
        summary = await project_usage_summary(session, project_id=1)

    assert summary["project_id"] == 1
    assert "current_month" in summary
    assert "alltime" in summary
    assert "storage_bytes_total" in summary
    assert summary["storage_bytes_total"] == 1024


@pytest.mark.asyncio
async def test_project_usage_summary_current_month_has_year_month():
    """project_usage_summary().current_month includes year and month."""
    from libs.usage import project_usage_summary

    with (
        patch("libs.usage.monthly_totals", new=AsyncMock(return_value={})),
        patch("libs.usage.alltime_totals", new=AsyncMock(return_value={})),
        patch("libs.usage.latest_storage_bytes", new=AsyncMock(return_value=0)),
    ):
        session = AsyncMock()
        summary = await project_usage_summary(session, project_id=3)

    now = datetime.now(UTC)
    assert summary["current_month"]["year"] == now.year
    assert summary["current_month"]["month"] == now.month


# ---------------------------------------------------------------------------
# Storage bytes populated in worker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_video_emits_storage_bytes(tmp_path):
    """handle_process_video emits a storage_bytes_written usage event."""
    import sys
    from unittest.mock import AsyncMock, MagicMock, patch

    # Stub heavy imports so the worker can be imported in a test environment.
    for mod in ["ffmpeg", "cv2"]:
        if mod not in sys.modules:
            sys.modules[mod] = MagicMock()

    from libs.models import Job, ProcessingRun, Video

    fake_video = MagicMock()
    fake_video.id = 1
    fake_video.project_id = 1
    fake_video.source_path = str(tmp_path / "source.mp4")
    (tmp_path / "source.mp4").write_bytes(b"fakevideocontent")

    fake_job = MagicMock()
    fake_job.id = 10
    fake_job.processing_run_id = 20

    fake_run = MagicMock()
    fake_run.id = 20

    emitted_events: list[dict] = []

    async def _fake_emit(
        db, *, project_id, event_type, quantity, processing_run_id=None, metadata=None
    ):
        emitted_events.append({"event_type": str(event_type), "quantity": quantity})

    db_session = AsyncMock()

    async def _fake_get(model, pk):

        if model is Job:
            return fake_job
        if model is Video:
            return fake_video
        if model is ProcessingRun:
            return fake_run
        return None

    db_session.get = AsyncMock(side_effect=_fake_get)
    db_session.add = MagicMock()
    db_session.flush = AsyncMock()
    db_session.commit = AsyncMock()

    ctx_manager = AsyncMock()
    ctx_manager.__aenter__ = AsyncMock(return_value=db_session)
    ctx_manager.__aexit__ = AsyncMock(return_value=False)

    fake_meta = {"duration_seconds": 10.0, "fps": 30.0, "width": 1920, "height": 1080}
    fake_state = {
        "version": 7,
        "clip_summary": {"clip_count": 0, "total_clip_duration_ms": 0},
        "manifest_path": "",
    }

    with (
        patch("libs.db.AsyncSessionLocal", return_value=ctx_manager),
        patch("apps.worker.jobs.process_video.AsyncSessionLocal", return_value=ctx_manager),
        patch("apps.worker.jobs.process_video.normalize_video", return_value=None),
        patch("apps.worker.jobs.process_video.probe_media_streams", return_value=_VALID_SRC_INFO),
        patch("apps.worker.jobs.process_video.probe_video", return_value=fake_meta),
        patch("apps.worker.jobs.process_video.extract_frames", return_value=["f1", "f2", "f3"]),
        patch("apps.worker.jobs.process_video.run_pipeline", return_value=(
            fake_state,
            {"version": 1, "detections": []},
            {"version": 1, "tracks": [], "track_count": 0},
            {"version": 1, "poses": [], "pose_count": 0},
            {"version": 1, "features": [], "feature_count": 0},
            {"version": 1, "segments": [], "segment_count": 0},
        )),
        patch("apps.worker.jobs.process_video.generate_clips", return_value=[]),
        patch(
            "apps.worker.jobs.process_video._save_json",
            new=AsyncMock(return_value=("/fake/path", 100)),
        ),
        patch("apps.worker.jobs.process_video.emit_usage", new=_fake_emit),
        patch("apps.worker.jobs.process_video.enqueue_run_event", new=AsyncMock()),
        patch("apps.worker.jobs.process_video.settings") as mock_settings,
    ):
        mock_settings.detector_backend = "stub"
        mock_settings.tracker_backend = "stub"
        mock_settings.pose_backend = "stub"
        mock_settings.storage_backend = "local"
        mock_settings.frame_sample_fps = 1.0
        mock_settings.normalized_dir = str(tmp_path / "norm")
        mock_settings.artifacts_dir = str(tmp_path / "artifacts")
        # Make the source file "exist" check pass via a real file.
        fake_video.source_path = str(tmp_path / "source.mp4")

        from apps.worker.jobs.process_video import handle_process_video

        await handle_process_video({"job_id": 10, "payload": {"video_id": 1}})

    storage_events = [e for e in emitted_events if e["event_type"] == "storage_bytes_written"]
    assert len(storage_events) >= 1, "Expected at least one storage_bytes_written event"
    assert storage_events[0]["quantity"] > 0

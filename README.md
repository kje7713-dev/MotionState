# MotionState — Video Motion Pipeline

> Turn video into structured human motion data.

## What this repo does

1. **Video ingest** – accepts uploaded video files via a REST API, stores them locally (S3/R2-ready interface).
2. **Normalization** – re-encodes uploads to a consistent format/resolution using FFmpeg.
3. **Queued processing** – enqueues a `process_video` job in Redis; a background worker picks it up.
4. **Frame extraction** – samples frames from the normalized video at a configurable rate (default: 2 FPS) and writes JPEG files under `data/artifacts/{video_id}/frames/`.
5. **Person detection** – runs a configurable person detector (stub by default; YOLOv8 when enabled) on each extracted frame.
6. **Multi-frame tracking** – assigns persistent track IDs across frames using a deterministic IOU-based tracker (stub by default; IOU tracker when `TRACKER_BACKEND=iou`).
7. **Structured artifacts** – writes three time-indexed JSON artifacts per video: `state.json` (summary + pipeline output), `detections.json` (per-frame bounding boxes), and `tracks.json` (persistent track histories).

## What is NOT in scope (yet)

- No sport-specific ontology (no BJJ, no tennis, …)
- No coaching logic or scoring engine
- No real-time guarantees
- No frontend / product UI
- No pose estimation
- No motion/state feature derivation
- No segmentation

## Architecture

```
┌──────────┐   POST /videos   ┌───────────┐   Redis queue   ┌────────────┐
│  Client  │ ──────────────►  │  FastAPI  │ ──────────────► │   Worker   │
└──────────┘                  │   (API)   │                  │  (Python)  │
                              └─────┬─────┘                  └─────┬──────┘
                                    │                               │
                              Postgres (metadata)     FFmpeg + detector + tracker
                                    │                               │
                              ┌─────▼─────────────────────────────▼──────┐
                              │            Local filesystem               │
                              │  data/uploads/  data/normalized/          │
                              │  data/artifacts/{video_id}/               │
                              │    frames/  state.json                    │
                              │    detections.json  tracks.json           │
                              └───────────────────────────────────────────┘
```

## Quick start

```bash
cp .env.example .env
make dev          # docker compose up --build
```

Visit `http://localhost:8000/health` — should return `{"status":"ok"}`.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| POST | `/videos` | Upload a video file |
| GET | `/videos/{video_id}` | Get video status |
| GET | `/videos/{video_id}/artifacts` | List artifact records for a video |
| GET | `/jobs/{job_id}` | Get job status |

## Development

```bash
# Base utility install (no heavy CV deps)
pip install -e ".[dev]"

# Full install with vision extras (enables YOLOv8 detector path)
pip install -e ".[dev,vision]"

make test      # run pytest
make lint      # ruff check
make format    # ruff format
```

## Install modes

| Mode | Command | When to use |
|------|---------|-------------|
| Base (default) | `pip install -e .` | API, worker with stub detector/tracker, tests — no heavy deps |
| Vision extras | `pip install -e ".[vision]"` | Enable the real YOLOv8 detector path |

The default install works without any CV packages.  Only install the `vision`
extras if you intend to run `DETECTOR_BACKEND=yolo`.

The IOU tracker (`TRACKER_BACKEND=iou`) requires no additional dependencies —
it ships as part of the base install.

## Current capability

- **Video normalization** – re-encodes uploads to a consistent format/resolution via FFmpeg
- **Frame extraction** – samples JPEG frames at a configurable rate (default: 2 FPS)
- **Person detection artifact** – runs a configurable detector (stub by default; YOLOv8 when `vision` extras are installed and `DETECTOR_BACKEND=yolo`) and writes `detections.json`
- **Multi-frame tracking** – assigns persistent track IDs across frames using deterministic IOU matching (`TRACKER_BACKEND=iou`); writes `tracks.json`
- **State artifact** – writes `state.json` with per-video detection summary counts and tracking summary

## Current limitations

- **Pose estimation not implemented** – no 2-D keypoint extraction
- **Feature derivation still stubbed** – motion/state feature derivation returns empty results
- **Segmentation still stubbed** – temporal segmentation returns empty results
- **Domain ontology intentionally absent** – no sport-specific labels or scoring logic

## Configuration

Key settings (see `.env.example` for the full list):

| Variable | Default | Description |
|----------|---------|-------------|
| `FRAME_SAMPLE_FPS` | `2.0` | Frames to extract per second of video |
| `DETECTOR_BACKEND` | `stub` | `stub` (no-op) or `yolo` (YOLOv8) |
| `DETECTOR_MODEL` | `yolov8n.pt` | YOLO model name or path (only when `yolo` backend) |
| `TRACKER_BACKEND` | `stub` | `stub` (no-op) or `iou` (deterministic IOU tracker) |
| `TRACKER_IOU_THRESHOLD` | `0.3` | Minimum IOU to associate a detection with an existing track |
| `TRACKER_MAX_AGE` | `30` | Frames a track can go undetected before being dropped |

To enable YOLOv8 detection:
```bash
pip install -e ".[vision]"   # or: pip install ultralytics pillow numpy
DETECTOR_BACKEND=yolo make dev
```

To enable IOU-based tracking (no extra dependencies needed):
```bash
TRACKER_BACKEND=iou make dev
```

If `DETECTOR_BACKEND=yolo` is set but the `vision` extras are **not** installed,
the worker logs a warning and falls back to the stub detector automatically — no crash.

## Artifact schemas

### `state.json`

```json
{
  "video_id": "123",
  "version": 3,
  "segments": [],
  "tracks": [
    {
      "track_id": 1,
      "detections": []
    }
  ],
  "features": [],
  "detections_summary": {
    "frame_count": 120,
    "frames_with_people": 97,
    "total_detections": 181
  },
  "tracking_summary": {
    "track_count": 2,
    "tracked_frame_count": 97,
    "average_detections_per_frame": 1.87
  },
  "notes": "first real CV stages: frame extraction, person detection, tracking"
}
```

### `detections.json`

```json
{
  "video_id": "123",
  "version": 1,
  "sample_fps": 2,
  "frames": [
    {
      "frame_index": 0,
      "timestamp_ms": 0,
      "path": "data/artifacts/123/frames/frame_000000.jpg",
      "detections": [
        {
          "class_label": "person",
          "bbox": {
            "x": 10,
            "y": 20,
            "width": 100,
            "height": 200,
            "confidence": 0.93
          }
        }
      ]
    }
  ]
}
```

### `tracks.json`

```json
{
  "video_id": "123",
  "version": 1,
  "track_count": 2,
  "tracks": [
    {
      "track_id": 1,
      "detections": [
        {
          "frame_index": 0,
          "timestamp_ms": 0,
          "class_label": "person",
          "bbox": {
            "x": 10,
            "y": 20,
            "width": 100,
            "height": 200,
            "confidence": 0.93
          }
        }
      ]
    }
  ]
}
```

## Next steps

1. Pose estimation (2-D keypoints per tracked person)
2. Motion/state feature derivation from tracks + landmarks
3. Temporal segmentation
4. Schema hardening for queryable state output
5. Optional ontology layers on top

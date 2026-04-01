# MotionState — Video Motion Pipeline

> Turn video into structured human motion data.

## What this repo does

1. **Video ingest** – accepts uploaded video files via a REST API, stores them locally (S3/R2-ready interface).
2. **Normalization** – re-encodes uploads to a consistent format/resolution using FFmpeg.
3. **Queued processing** – enqueues a `process_video` job in Redis; a background worker picks it up.
4. **Frame extraction** – samples frames from the normalized video at a configurable rate (default: 2 FPS) and writes JPEG files under `data/artifacts/{video_id}/frames/`.
5. **Person detection** – runs a configurable person detector (stub by default; YOLOv8 when enabled) on each extracted frame.
6. **Multi-frame tracking** – assigns persistent track IDs across frames using a deterministic IOU-based tracker (stub by default; IOU tracker when `TRACKER_BACKEND=iou`).
7. **Pose estimation** – estimates 2-D body keypoints for each tracked person per frame (stub by default; MediaPipe BlazePose when `POSE_BACKEND=mediapipe`).
8. **Generic motion feature derivation** – derives scalar geometric and temporal features from tracked pose data (torso angle, joint angles, widths, centroid velocity, etc.); writes `features.json`.
9. **Structured artifacts** – writes five time-indexed JSON artifacts per video: `state.json` (summary + pipeline output), `detections.json` (per-frame bounding boxes), `tracks.json` (persistent track histories), `poses.json` (per-frame body keypoints), and `features.json` (derived motion features).

## What is NOT in scope (yet)

- No sport-specific ontology (no BJJ, no tennis, …)
- No coaching logic or scoring engine
- No real-time guarantees
- No frontend / product UI
- No temporal segmentation

## Architecture

```
┌──────────┐   POST /videos   ┌───────────┐   Redis queue   ┌────────────┐
│  Client  │ ──────────────►  │  FastAPI  │ ──────────────► │   Worker   │
└──────────┘                  │   (API)   │                  │  (Python)  │
                              └─────┬─────┘                  └─────┬──────┘
                                    │                               │
                              Postgres (metadata)     FFmpeg + detector + tracker
                                    │                     + pose estimator
                              ┌─────▼──────────────────+ feature deriver──────┐
                              │            Local filesystem                    │
                              │  data/uploads/  data/normalized/               │
                              │  data/artifacts/{video_id}/                    │
                              │    frames/  state.json                         │
                              │    detections.json  tracks.json                │
                              │    poses.json  features.json                   │
                              └────────────────────────────────────────────────┘
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

# Full install with pose extras (enables MediaPipe pose estimator path)
pip install -e ".[dev,pose]"

make test      # run pytest
make lint      # ruff check
make format    # ruff format
```

## Install modes

| Mode | Command | When to use |
|------|---------|-------------|
| Base (default) | `pip install -e .` | API, worker with stub detector/tracker/pose, tests — no heavy deps |
| Vision extras | `pip install -e ".[vision]"` | Enable the real YOLOv8 detector path |
| Pose extras | `pip install -e ".[pose]"` | Enable the real MediaPipe pose estimation path |

The default install works without any CV packages.  Only install the `vision`
extras if you intend to run `DETECTOR_BACKEND=yolo`, and the `pose` extras if
you intend to run `POSE_BACKEND=mediapipe`.

The IOU tracker (`TRACKER_BACKEND=iou`) requires no additional dependencies —
it ships as part of the base install.

## Current capability

- **Video normalization** – re-encodes uploads to a consistent format/resolution via FFmpeg
- **Frame extraction** – samples JPEG frames at a configurable rate (default: 2 FPS)
- **Person detection artifact** – runs a configurable detector (stub by default; YOLOv8 when `vision` extras are installed and `DETECTOR_BACKEND=yolo`) and writes `detections.json`
- **Multi-frame tracking** – assigns persistent track IDs across frames using deterministic IOU matching (`TRACKER_BACKEND=iou`); writes `tracks.json`
- **Pose estimation** – estimates 2-D body keypoints per tracked person per frame (stub by default; MediaPipe BlazePose when `pose` extras are installed and `POSE_BACKEND=mediapipe`); writes `poses.json`
- **Generic motion feature derivation** – derives domain-agnostic scalar features from tracked pose landmarks; writes `features.json`. Included features:
  - `torso_angle` – lean of the torso relative to vertical (degrees)
  - `shoulder_width` – pixel distance between left and right shoulders
  - `hip_width` – pixel distance between left and right hips
  - `left_elbow_angle` / `right_elbow_angle` – joint angle at each elbow (degrees)
  - `left_knee_angle` / `right_knee_angle` – joint angle at each knee (degrees)
  - `keypoint_visibility_count` – number of high-confidence body landmarks
  - `bbox_area` – bounding-box area in pixels²
  - `centroid_velocity` – keypoint-centroid displacement per millisecond between consecutive frames
- **State artifact** – writes `state.json` with per-video detection summary, tracking summary, pose summary, and feature summary

## Current limitations

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
| `POSE_BACKEND` | `stub` | `stub` (no-op) or `mediapipe` (MediaPipe BlazePose) |
| `POSE_MIN_CONFIDENCE` | `0.3` | Minimum landmark visibility score to include a keypoint |

To enable YOLOv8 detection:
```bash
pip install -e ".[vision]"   # or: pip install ultralytics pillow numpy
DETECTOR_BACKEND=yolo make dev
```

To enable IOU-based tracking (no extra dependencies needed):
```bash
TRACKER_BACKEND=iou make dev
```

To enable MediaPipe pose estimation:
```bash
pip install -e ".[pose]"   # or: pip install mediapipe pillow numpy
POSE_BACKEND=mediapipe make dev
```

If `DETECTOR_BACKEND=yolo` is set but the `vision` extras are **not** installed,
the worker logs a warning and falls back to the stub detector automatically — no crash.

## Artifact schemas

### `state.json`

```json
{
  "video_id": "123",
  "version": 5,
  "segments": [],
  "tracks": [
    {
      "track_id": 1,
      "detections": []
    }
  ],
  "features": [
    {
      "track_id": 1,
      "name": "torso_angle",
      "start_ms": 5000,
      "end_ms": 5000,
      "value": 37.2
    }
  ],
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
  "pose_summary": {
    "pose_count": 24,
    "posed_track_count": 2,
    "average_keypoints_per_pose": 17.0
  },
  "feature_summary": {
    "feature_count": 48,
    "featured_track_count": 2,
    "feature_names": [
      "torso_angle",
      "left_elbow_angle",
      "right_elbow_angle",
      "centroid_velocity"
    ]
  },
  "notes": "first real CV stages: frame extraction, person detection, tracking, pose estimation, feature derivation"
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

### `poses.json`

```json
{
  "video_id": "123",
  "version": 1,
  "pose_count": 24,
  "poses": [
    {
      "frame_index": 10,
      "timestamp_ms": 5000,
      "track_id": 1,
      "keypoints": [
        { "name": "nose", "x": 120.0, "y": 80.0, "confidence": 0.92 },
        { "name": "left_shoulder", "x": 100.0, "y": 140.0, "confidence": 0.88 }
      ]
    }
  ]
}
```

### `features.json`

```json
{
  "video_id": "123",
  "version": 1,
  "feature_count": 48,
  "features": [
    {
      "track_id": 1,
      "name": "torso_angle",
      "start_ms": 5000,
      "end_ms": 5000,
      "value": 37.2
    },
    {
      "track_id": 1,
      "name": "centroid_velocity",
      "start_ms": 5000,
      "end_ms": 5500,
      "value": 18.4
    }
  ]
}
```

## Next steps

1. Temporal segmentation of motion feature time-series
2. Schema hardening for queryable state output
3. Optional ontology layers on top

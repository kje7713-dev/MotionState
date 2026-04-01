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
9. **Generic temporal segmentation** – divides the feature time-series into labelled windows using motion-intensity thresholding and adjacent-window merging; writes `segments.json`.
10. **Clip generation** – extracts an MP4 clip for each temporal segment from the normalized video; writes files under `data/artifacts/{video_id}/clips/`.
11. **Timeline manifest** – writes `timeline_manifest.json`, a downstream-friendly summary that ties all artifacts together with a per-segment timeline containing clip paths and related artifact references.
12. **Structured artifacts** – writes eight time-indexed artifacts per video: `state.json` (summary + pipeline output), `detections.json` (per-frame bounding boxes), `tracks.json` (persistent track histories), `poses.json` (per-frame body keypoints), `features.json` (derived motion features), `segments.json` (temporal segments), clip files, and `timeline_manifest.json`.

## What is NOT in scope (yet)

- No sport-specific ontology (no BJJ, no tennis, …)
- No coaching logic or scoring engine
- No real-time guarantees
- No frontend / product UI
- No domain-specific segment interpretation
- No ontology / semantic labelling
- No multi-tenant authentication

## Architecture

```
┌──────────┐   POST /videos         ┌───────────┐   Redis queue   ┌────────────┐
│  Client  │ ───────────────────►   │  FastAPI  │ ──────────────► │   Worker   │
│          │   POST /videos/        │   (API)   │                  │  (Python)  │
│          │       upload-init  ◄── └─────┬─────┘                  └─────┬──────┘
│          │   PUT <signed-url>            │                               │
│          │ ─────────────────►    Postgres (metadata)     FFmpeg + CV pipeline
└──────────┘    (S3 / R2)                  │                 + storage backend
                                    ┌─────▼──────────────────────────────────┐
                                    │     Swappable storage backend          │
                                    │                                        │
                                    │  local (default):                      │
                                    │    data/artifacts/{video_id}/          │
                                    │      frames/  state.json               │
                                    │      detections.json  tracks.json      │
                                    │      poses.json  features.json         │
                                    │      segments.json  clips/             │
                                    │      timeline_manifest.json            │
                                    │                                        │
                                    │  S3 / R2:                              │
                                    │    artifacts/{video_id}/state.json     │
                                    │    artifacts/{video_id}/clips/…        │
                                    └────────────────────────────────────────┘
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
| POST | `/videos` | Upload a video file (simple multipart; dev/local path) |
| POST | `/videos/upload-init` | Prepare a direct-to-storage upload; returns signed upload URL |
| GET | `/videos/{video_id}` | Get video status |
| GET | `/videos/{video_id}/artifacts` | List artifact records for a video |
| GET | `/videos/{video_id}/runs` | List all processing runs for a video (newest first) |
| POST | `/videos/{video_id}/reprocess` | Create a new processing run and enqueue reprocessing |
| GET | `/videos/{video_id}/timeline` | Return the timeline manifest (latest run, or `?run_id=N`) |
| GET | `/videos/{video_id}/state` | Return the `state.json` artifact (latest run, or `?run_id=N`) |
| GET | `/videos/{video_id}/detections` | Return the `detections.json` artifact (latest run, or `?run_id=N`) |
| GET | `/videos/{video_id}/tracks` | Return the `tracks.json` artifact (latest run, or `?run_id=N`) |
| GET | `/videos/{video_id}/poses` | Return the `poses.json` artifact (latest run, or `?run_id=N`) |
| GET | `/videos/{video_id}/features` | Return the `features.json` artifact (latest run, or `?run_id=N`) |
| GET | `/videos/{video_id}/segments` | Return the `segments.json` artifact (latest run, or `?run_id=N`) |
| GET | `/jobs/{job_id}` | Get job status |

## Development

```bash
# Base utility install (no heavy CV deps)
pip install -e ".[dev]"

# Full install with vision extras (enables YOLOv8 detector path)
pip install -e ".[dev,vision]"

# Full install with pose extras (enables MediaPipe pose estimator path)
pip install -e ".[dev,pose]"

# Full install with object storage extras (enables S3 / R2 backend)
pip install -e ".[dev,storage]"

make test      # run pytest
make lint      # ruff check
make format    # ruff format
make smoke     # run only smoke tests (requires ffmpeg)
```

## Storage backends

MotionState uses a swappable storage abstraction so the same artifact pipeline
works on a local dev machine and in a cloud deployment.

### Local backend (default)

The default backend writes all artifacts directly to the local filesystem under
the directory configured by `ARTIFACTS_DIR` (default: `./data/artifacts`).
No additional dependencies are needed.

```bash
STORAGE_BACKEND=local   # the default; no extra config required
```

### S3 / Cloudflare R2 backend

Set `STORAGE_BACKEND=s3` and supply bucket credentials.  The backend uses
[boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) and
is compatible with AWS S3 and Cloudflare R2 (via `S3_ENDPOINT_URL`).

```bash
pip install -e ".[storage]"   # or: pip install boto3

STORAGE_BACKEND=s3
S3_BUCKET=my-motionstate-bucket
S3_REGION=auto                # use "auto" for R2; or an AWS region such as "us-east-1"
S3_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com   # R2 only; omit for AWS
S3_ACCESS_KEY_ID=<key-id>
S3_SECRET_ACCESS_KEY=<secret>
SIGNED_URL_EXPIRATION_SECONDS=3600
```

#### Canonical object keys

All artifacts use a stable, human-readable key layout in the bucket:

| Artifact | Key |
|----------|-----|
| Uploaded source video | `videos/{video_id}/source.mp4` |
| Normalized video | `videos/{video_id}/normalized.mp4` |
| State summary | `artifacts/{video_id}/state.json` |
| Detections | `artifacts/{video_id}/detections.json` |
| Tracks | `artifacts/{video_id}/tracks.json` |
| Poses | `artifacts/{video_id}/poses.json` |
| Features | `artifacts/{video_id}/features.json` |
| Segments | `artifacts/{video_id}/segments.json` |
| Timeline manifest | `artifacts/{video_id}/timeline_manifest.json` |
| Segment clips | `artifacts/{video_id}/clips/segment_NNN_<label>.mp4` |

The `Artifact.path` DB column stores the canonical key so artifact reads are
routed through the same storage backend regardless of which backend is active.

### Direct upload flow (`POST /videos/upload-init`)

For production deployments the client should upload the source video directly
to object storage instead of streaming through the API server.

1. Client calls `POST /videos/upload-init` with `{"filename": "game.mp4"}`.
2. API creates a pending `Video` row, generates a canonical storage key, and
   returns a pre-signed `PUT` URL (S3/R2 only; `null` for local backend).
3. Client uploads the file directly to the signed URL.
4. Client enqueues / triggers processing through the regular job queue.

```json
POST /videos/upload-init
{"filename": "game.mp4"}

→ 201
{
  "video_id": 42,
  "upload_url": "https://bucket.r2.cloudflarestorage.com/videos/42/source.mp4?…",
  "storage_key": "videos/42/source.mp4"
}
```

`upload_url` is `null` for the local backend — fall back to `POST /videos`
(multipart upload through the API server) for local development.

### What remains out of scope for storage

- Pre-signed *download* URLs are not yet exposed as an API endpoint (the
  `S3Storage.generate_download_url()` method exists but is not wired to a route).
- CDN / caching configuration for object storage is not managed by this repo.
- Lifecycle / retention policies for the S3 bucket are out of scope.

## Processing runs

Every time a video is processed (or reprocessed) the system creates a
**`ProcessingRun`** row that acts as the lineage anchor for that execution.
Jobs and artifacts are linked to the run that produced them.

### One video → many runs

```
video (id=42)
  └── ProcessingRun id=1  (trigger=initial,   status=completed)
        ├── Job id=7        (type=process_video, status=done)
        └── Artifacts: state, detections, tracks, poses, features, segments, clips…
  └── ProcessingRun id=2  (trigger=reprocess,  status=completed)
        ├── Job id=15       (type=process_video, status=done)
        └── Artifacts: state, detections, tracks, poses, features, segments, clips…
```

### Default read behaviour

All artifact read endpoints (`/state`, `/detections`, `/tracks`, etc.) resolve
to the **latest completed run** for that video when no `run_id` is provided:

```
GET /videos/42/state           → artifacts from the most recently completed run
GET /videos/42/state?run_id=1  → artifacts from ProcessingRun id=1 specifically
```

Failed runs are excluded from the "latest successful run" resolution, so a
failed reprocessing attempt never overwrites what downstream consumers see.

### Listing runs

```
GET /videos/{video_id}/runs
```

Returns all runs for a video, newest first:

```json
[
  {
    "id": 2,
    "status": "completed",
    "trigger_type": "reprocess",
    "pipeline_version": "7",
    "created_at": "2024-01-02T00:00:00Z",
    "completed_at": "2024-01-02T00:01:00Z",
    "error": null
  },
  {
    "id": 1,
    "status": "completed",
    "trigger_type": "initial",
    "pipeline_version": "7",
    "created_at": "2024-01-01T00:00:00Z",
    "completed_at": "2024-01-01T00:01:00Z",
    "error": null
  }
]
```

### Triggering a reprocess

```
POST /videos/{video_id}/reprocess
```

Creates a **new** `ProcessingRun` and enqueues a new `process_video` job.
Previous runs and their artifacts are preserved unchanged.

```json
{
  "video_id": 42,
  "processing_run_id": 2,
  "job_id": 15
}
```

## Smoke test

The smoke test proves that the full pipeline still works as a connected system
after any change.  It runs a real video through every stage using stub CV
backends (no heavy optional dependencies) and validates that all expected
artifacts are produced.

**What the smoke test proves:**

- A small fixture video can be successfully normalized and decoded by FFmpeg
- Frames are extracted and passed through the CV pipeline (stub backends)
- All expected artifact files are written:
  `state.json`, `detections.json`, `tracks.json`, `poses.json`,
  `features.json`, `segments.json`, `timeline_manifest.json`, `clips/`
- `state.json` contains the expected top-level summary keys
- `timeline_manifest.json` contains the expected top-level keys
- Artifact rows are created for all artifact types
- The `GET /videos/{id}/state` and `GET /videos/{id}/timeline` API endpoints
  return correct responses against actually generated artifact files

**What the smoke test does NOT prove:**

- CV model quality or detection accuracy (stub backends return empty results)
- Production backend behaviour (YOLOv8, MediaPipe, ByteTrack)
- Domain-specific motion correctness

Run the smoke test locally (requires `ffmpeg`):

```bash
make smoke
```

The CI `smoke-test` job runs the same path automatically on every push.  It
installs `ffmpeg` via `apt` and runs `pytest -m smoke` against the committed
fixture video at `tests/fixtures/fixture.mp4`.

## Install modes

| Mode | Command | When to use |
|------|---------|-------------|
| Base (default) | `pip install -e .` | API, worker with stub detector/tracker/pose, tests — no heavy deps |
| Vision extras | `pip install -e ".[vision]"` | Enable the real YOLOv8 detector path |
| Pose extras | `pip install -e ".[pose]"` | Enable the real MediaPipe pose estimation path |
| Storage extras | `pip install -e ".[storage]"` | Enable the S3 / R2 storage backend (`STORAGE_BACKEND=s3`) |

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
- **Generic temporal segmentation** – divides the motion feature time-series into fixed-size windows, classifies each window with a domain-agnostic label, and merges adjacent same-label windows; writes `segments.json`. Generic labels:
  - `low_motion` – low centroid velocity; person is relatively still
  - `active_motion` – high centroid velocity; person is actively moving
  - `transition_window` – velocity between low and active thresholds; ambiguous motion
  - `sparse_data` – too few features in the window to classify reliably
- **Clip generation** – extracts one MP4 clip per segment from the normalized video using FFmpeg; clips are written under `data/artifacts/{video_id}/clips/` with deterministic filenames (e.g. `segment_000_low_motion.mp4`)
- **Timeline manifest** – writes `timeline_manifest.json` tying all pipeline artifacts together with a per-segment timeline; each entry includes `segment_index`, `start_ms`, `end_ms`, `label`, `confidence`, `clip_path`, and `related_artifacts` references
- **State artifact** – writes `state.json` with per-video detection summary, tracking summary, pose summary, feature summary, segmentation summary, clip summary, and manifest path

## Current limitations

- **Domain ontology intentionally absent** – no sport-specific labels, no scoring, no coaching logic

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
| `STORAGE_BACKEND` | `local` | `local` (filesystem) or `s3` (AWS S3 / Cloudflare R2) |
| `S3_BUCKET` | `` | S3/R2 bucket name (only when `STORAGE_BACKEND=s3`) |
| `S3_REGION` | `` | AWS region or `auto` for R2 (only when `STORAGE_BACKEND=s3`) |
| `S3_ENDPOINT_URL` | `` | Custom endpoint for R2 or MinIO (only when `STORAGE_BACKEND=s3`) |
| `S3_ACCESS_KEY_ID` | `` | S3/R2 access key (only when `STORAGE_BACKEND=s3`) |
| `S3_SECRET_ACCESS_KEY` | `` | S3/R2 secret key (only when `STORAGE_BACKEND=s3`) |
| `SIGNED_URL_EXPIRATION_SECONDS` | `3600` | Expiry for pre-signed upload URLs (only when `STORAGE_BACKEND=s3`) |

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
  "version": 7,
  "segments": [
    {
      "start_ms": 0,
      "end_ms": 2500,
      "label": "low_motion",
      "confidence": 0.88,
      "metadata": { "feature_count": 12 }
    }
  ],
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
  "segmentation_summary": {
    "segment_count": 4,
    "segment_labels": [
      "low_motion",
      "active_motion",
      "transition_window"
    ],
    "total_segment_duration_ms": 373850
  },
  "clip_summary": {
    "clip_count": 4,
    "total_clip_duration_ms": 373850
  },
  "manifest_path": "data/artifacts/123/timeline_manifest.json",
  "notes": "first real CV stages: frame extraction, person detection, tracking, pose estimation, feature derivation, temporal segmentation, clip generation"
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

### `segments.json`

```json
{
  "video_id": "123",
  "version": 1,
  "segment_count": 4,
  "segments": [
    {
      "start_ms": 0,
      "end_ms": 2500,
      "label": "low_motion",
      "confidence": 0.88,
      "metadata": { "feature_count": 12 }
    },
    {
      "start_ms": 2500,
      "end_ms": 5200,
      "label": "active_motion",
      "confidence": 0.91,
      "metadata": { "feature_count": 31 }
    }
  ]
}
```

### `timeline_manifest.json`

```json
{
  "video_id": "123",
  "version": 1,
  "duration_seconds": 373.85,
  "artifacts": {
    "state": "data/artifacts/123/state.json",
    "detections": "data/artifacts/123/detections.json",
    "tracks": "data/artifacts/123/tracks.json",
    "poses": "data/artifacts/123/poses.json",
    "features": "data/artifacts/123/features.json",
    "segments": "data/artifacts/123/segments.json"
  },
  "timeline": [
    {
      "segment_index": 0,
      "start_ms": 0,
      "end_ms": 2500,
      "label": "low_motion",
      "confidence": 0.88,
      "clip_path": "data/artifacts/123/clips/segment_000_low_motion.mp4",
      "related_artifacts": {
        "segments": "data/artifacts/123/segments.json",
        "features": "data/artifacts/123/features.json"
      }
    }
  ]
}
```

## Consumption API

MotionState exposes read endpoints so downstream systems can query the latest artifact for any processing stage without having to locate files manually.

### Artifact retrieval endpoints

| Endpoint | Returns |
|----------|---------|
| `GET /videos/{video_id}/state` | Latest `state.json` — pipeline summary with detection, tracking, pose, feature, segmentation, and clip summaries |
| `GET /videos/{video_id}/detections` | Latest `detections.json` — per-frame bounding boxes |
| `GET /videos/{video_id}/tracks` | Latest `tracks.json` — persistent multi-frame track histories |
| `GET /videos/{video_id}/poses` | Latest `poses.json` — per-frame body keypoints |
| `GET /videos/{video_id}/features` | Latest `features.json` — derived scalar motion features |
| `GET /videos/{video_id}/segments` | Latest `segments.json` — temporal segments with labels and confidence scores |
| `GET /videos/{video_id}/timeline` | Latest `timeline_manifest.json` — full manifest tying all artifacts together |
| `GET /videos/{video_id}/artifacts` | All artifact metadata rows for the video |

All single-artifact endpoints:
- Return the **most recently created** artifact of that type (deterministic: ordered by `Artifact.id DESC`).
- Validate that the stored file path is inside `ARTIFACTS_DIR` before reading.
- Return `404` if the video is missing, if no artifact row exists, or if the file on disk cannot be found.
- Return `404` if the stored path escapes the configured `ARTIFACTS_DIR`.

### Schema stability

Artifact schemas are versioned. Each artifact JSON file carries a `version` integer at the top level. Version constants are defined in `libs/schemas.py` and are the canonical reference for downstream consumers:

| Artifact | Current version |
|----------|----------------|
| `state.json` | 7 |
| `detections.json` | 1 |
| `tracks.json` | 1 |
| `poses.json` | 1 |
| `features.json` | 1 |
| `segments.json` | 1 |
| `timeline_manifest.json` | 1 |

Downstream consumers should rely on the versioned schemas. When a schema changes incompatibly, the version number will be incremented.


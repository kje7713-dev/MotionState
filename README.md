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
- No OAuth or full user account system
- No billing

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

## Python SDK

The `sdk/python/motionstate_client` package is a small official SDK that wraps
the REST API so you don't have to stitch raw HTTP calls together manually.

### Installation

```bash
pip install httpx            # only external dependency
# then add sdk/python to your PYTHONPATH, or pip install -e sdk/python
```

### SDK quickstart

```python
from motionstate_client import MotionStateClient

client = MotionStateClient(
    base_url="http://localhost:8000",
    api_key="ms_your_key_here",
)

# Upload a video and wait for processing to complete.
upload = client.submit_video("my_video.mp4")
run = client.wait_for_run_completion(upload.video_id, upload.processing_run_id)

# Fetch all pipeline outputs in one call.
outputs = client.fetch_latest_outputs(upload.video_id)
print(outputs["state"]["tracking_summary"])
```

### Typical workflow

1. **Create a project and API key** (once, via the HTTP API or `curl`):

   ```bash
   # Create project
   curl -X POST "http://localhost:8000/projects?name=my-project"
   # {"id": 1, "name": "my-project", …}

   # Generate API key
   curl -X POST "http://localhost:8000/projects/1/api-keys?name=dev"
   # {"key": "ms_…", …}   ← save this, it is shown only once
   ```

2. **Upload a video** (choose one path):

   ```python
   # Simple upload (local dev / small files)
   upload = client.upload_video("clip.mp4")

   # Signed upload (production / large files → direct to S3/R2)
   init = client.upload_init("clip.mp4")
   if init.upload_url:
       import httpx
       httpx.put(init.upload_url, content=open("clip.mp4", "rb").read())
       reprocess = client.reprocess_video(init.video_id)
       run_id = reprocess.processing_run_id
   ```

3. **Wait for run completion** (polling) or receive a **webhook**:

   ```python
   # Polling
   run = client.wait_for_run_completion(video_id, run_id, timeout=300)

   # Webhook (register once)
   # curl -X POST "http://localhost:8000/projects/1/webhooks" \
   #      -H "Content-Type: application/json" \
   #      -d '{"url": "https://your-server.example/hook"}'
   # The signed payload is delivered when processing_run.completed fires.
   ```

4. **Fetch outputs**:

   ```python
   # All available artifacts in one call
   outputs = client.fetch_latest_outputs(video_id)

   # Or individually
   state    = client.get_state(video_id)
   timeline = client.get_timeline(video_id)
   tracks   = client.get_tracks(video_id)
   ```

### SDK methods

| Method | Description |
|--------|-------------|
| `upload_video(path)` | Multipart upload + enqueue |
| `submit_video(path)` | Alias for `upload_video` |
| `upload_init(filename)` | Init signed upload (S3/R2) |
| `get_video(video_id)` | Video metadata |
| `list_artifacts(video_id)` | All artifact records |
| `list_runs(video_id)` | All processing runs |
| `reprocess_video(video_id)` | Enqueue a new run |
| `get_state(video_id)` | `state.json` artifact |
| `get_detections(video_id)` | `detections.json` artifact |
| `get_tracks(video_id)` | `tracks.json` artifact |
| `get_poses(video_id)` | `poses.json` artifact |
| `get_features(video_id)` | `features.json` artifact |
| `get_segments(video_id)` | `segments.json` artifact |
| `get_timeline(video_id)` | `timeline_manifest.json` |
| `get_project_usage(project_id)` | Usage summary |
| `wait_for_run_completion(…)` | Poll until terminal status |
| `fetch_latest_outputs(video_id)` | All artifacts in one call |

All artifact read methods accept an optional `run_id` keyword argument to pin
to a specific processing run.

### SDK exceptions

| Exception | HTTP status |
|-----------|-------------|
| `AuthError` | 401, 403 |
| `NotFoundError` | 404 |
| `QuotaError` | 429 |
| `ServerError` | 5xx |
| `PollingTimeout` | — (raised after timeout expires) |

All exceptions inherit from `MotionStateError`.

### Example scripts

See `examples/python/` for runnable scripts:

| Script | Description |
|--------|-------------|
| `upload_and_poll.py` | Upload a file, wait for completion, print summary |
| `signed_upload_then_poll.py` | Signed S3 upload path, then poll |
| `fetch_latest_timeline.py` | Print the timeline manifest for a video |
| `list_runs_and_usage.py` | Show all runs and monthly usage |

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
| POST | `/projects` | Create a new project (tenancy boundary) |
| GET | `/projects/{project_id}` | Get project metadata |
| POST | `/projects/{project_id}/api-keys` | Generate a new API key for a project (raw key returned once) |
| GET | `/projects/{project_id}/api-keys` | List API keys for a project (without raw key) |
| POST | `/projects/{project_id}/webhooks` | Register a webhook endpoint for a project |
| GET | `/projects/{project_id}/webhooks` | List webhook endpoints for a project |
| PATCH | `/projects/{project_id}/webhooks/{webhook_id}` | Update a webhook endpoint (url, is_active, event_types) |
| DELETE | `/projects/{project_id}/webhooks/{webhook_id}` | Delete a webhook endpoint |

> **All video and job routes require authentication.** See [Authentication](#authentication) below.

## Authentication

MotionState uses API key authentication scoped to **projects**.  A project is
the ownership boundary: every video, processing run, and artifact belongs to
exactly one project and is only accessible to API keys from that project.

### Concepts

| Concept | Description |
|---------|-------------|
| **Project** | Tenancy boundary. Create one per application or integration. |
| **API Key** | Opaque secret tied to a project. The raw key is shown **once** at creation and never stored — only its SHA-256 hash is persisted. |

### Setup

**1. Create a project**

```bash
curl -X POST "http://localhost:8000/projects?name=MyApp"
# → {"id": 1, "name": "MyApp", "created_at": "..."}
```

**2. Generate an API key**

```bash
curl -X POST "http://localhost:8000/projects/1/api-keys?name=production"
# → {"id": 1, "name": "production", "key": "ms_live_...", "key_prefix": "ms_live_xxxx", "created_at": "..."}
```

> **Save the `key` value now.** It is returned only once and cannot be retrieved later.

### Using the API key

Pass the key in the `X-API-Key` header:

```bash
# Upload a video
curl -X POST "http://localhost:8000/videos" \
  -H "X-API-Key: ms_live_your_secret_key" \
  -F "file=@myvideo.mp4"

# Get video status
curl "http://localhost:8000/videos/1" \
  -H "X-API-Key: ms_live_your_secret_key"

# Get processed state artifact
curl "http://localhost:8000/videos/1/state" \
  -H "X-API-Key: ms_live_your_secret_key"
```

### Error responses

| Status | Cause |
|--------|-------|
| `401 Unauthorized` | Missing, invalid, or inactive API key |
| `404 Not Found` | Resource does not exist **or belongs to a different project** (cross-project existence is not revealed) |

### Key management

```bash
# List keys for a project (no raw secrets included)
curl "http://localhost:8000/projects/1/api-keys" \
  -H "X-API-Key: ms_live_your_secret_key"
```

## Webhooks

MotionState pushes signed HTTP POST notifications to registered endpoints
whenever a processing run changes state.  This eliminates polling and makes
integration straightforward.

### Setup

**1. Register a webhook endpoint**

```bash
curl -X POST "http://localhost:8000/projects/1/webhooks" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://your-service.example.com/webhooks/motionstate"}'
# → {"id": 1, "secret": "<64-char hex>", "is_active": true, "event_types": null, ...}
```

> **Save the `secret` value now.** It is generated once and never returned again.
> Use it to verify incoming request signatures.

**2. Optionally filter by event type**

```bash
curl -X POST "http://localhost:8000/projects/1/webhooks" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://your-service.example.com/webhooks",
    "event_types": ["processing_run.completed", "processing_run.failed"]
  }'
```

When `event_types` is `null` the endpoint receives **all** event types.

### Event types

| Event type | When emitted |
|------------|-------------|
| `processing_run.created` | A new processing run row is created (video upload or reprocess request) |
| `processing_run.running` | Worker picks up the job and starts processing |
| `processing_run.completed` | Pipeline finished successfully; artifacts are written |
| `processing_run.failed` | Pipeline encountered an unrecoverable error |

### Payload format

```json
{
  "event_id": "b7e2a1c0-4f3d-4e8b-91a2-123456789abc",
  "event_type": "processing_run.completed",
  "occurred_at": "2024-06-01T12:34:56.789012+00:00",
  "project_id": 1,
  "video_id": 42,
  "processing_run_id": 7,
  "status": "completed",
  "artifact_types": ["state", "detections", "tracks", "poses", "features", "segments", "timeline_manifest"]
}
```

For `processing_run.failed` events an `"error"` field is included:

```json
{
  "event_type": "processing_run.failed",
  "status": "error",
  "error": "Source video not found: ..."
}
```

For `processing_run.completed` events an `"artifact_types"` list is included.

### Verifying signatures

Every delivery includes an `X-MotionState-Signature` header containing the
HMAC-SHA256 hex digest of the **raw request body** keyed with the endpoint's
secret.  Keys are JSON-serialised with `sort_keys=True` for determinism.

```python
import hashlib, hmac

def verify_signature(raw_body: bytes, secret: str, received_sig: str) -> bool:
    expected = hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, received_sig)

# In your webhook handler:
body = request.get_data()  # raw bytes before JSON parse
sig  = request.headers.get("X-MotionState-Signature", "")
if not verify_signature(body, WEBHOOK_SECRET, sig):
    abort(401)
```

### Delivery behaviour

- Webhooks are delivered **asynchronously** by the background worker.  The API
  never blocks on webhook delivery.
- Up to **4 attempts** total (1 initial + 3 retries) are made before giving up.
- `last_success_at` and `last_failure_at` timestamps are updated on the
  endpoint row after each attempt.
- Inactive endpoints (`is_active: false`) are silently skipped.

### Managing endpoints

```bash
# List endpoints (secret not included)
curl "http://localhost:8000/projects/1/webhooks"

# Disable an endpoint
curl -X PATCH "http://localhost:8000/projects/1/webhooks/1" \
  -H "Content-Type: application/json" \
  -d '{"is_active": false}'

# Delete an endpoint
curl -X DELETE "http://localhost:8000/projects/1/webhooks/1"
```

## Usage metering and project quotas

MotionState records append-only **usage events** for each project so that
operators can see what the service is actually doing and enforce resource
limits before they become a problem.

> **This is internal accounting and quota control, not billing.**
> There is no invoice generation, no pricing logic, and no payment provider
> integration in this repo.

### Metered dimensions

The following dimensions are tracked per project:

| `event_type` | unit | when emitted |
|---|---|---|
| `videos_uploaded` | count | on `POST /videos` and `POST /videos/upload-init` |
| `video_seconds_processed` | seconds | on processing-run completion (worker) |
| `frames_extracted` | count | on processing-run completion (worker) |
| `clips_generated` | count | on processing-run completion (worker) |
| `storage_bytes_written` | bytes | on processing-run completion (all JSON artifacts + clips) |
| `webhook_deliveries` | count | on successful webhook delivery (worker) |
| `api_reads` | count | (reserved; not yet emitted on all read endpoints) |

All events are stored in the `usage_events` table and are never modified or
deleted — aggregations are computed at query time.

### Project quota fields

Each `Project` row may have the following limit fields (all nullable;
`null` means unlimited):

| Field | Default | Description |
|---|---|---|
| `max_videos_per_month` | `null` | Max videos that can be uploaded per calendar month |
| `max_video_seconds_per_month` | `null` | Max video-seconds processed per calendar month |
| `max_storage_bytes` | `null` | Cumulative storage byte ceiling |
| `max_api_reads_per_month` | `null` | Max API read calls per calendar month (reserved) |
| `is_suspended` | `false` | If `true`, all new uploads and reprocessing are rejected |

Quota fields are set directly on the `Project` row (e.g. via a database
admin tool or a future management API).

### Quota enforcement

Quota violations are checked at the start of:

- `POST /videos` (multipart upload)
- `POST /videos/upload-init` (direct-to-storage upload init)
- `POST /videos/{id}/reprocess`

Behavior on violation:

- **Suspended project** → `403 Forbidden` with `"reason": "project_suspended"`
- **Monthly limit reached** → `429 Too Many Requests` with the quota dimension and current value
- **Storage ceiling reached** → `429 Too Many Requests`

Example error response:

```json
{
  "detail": {
    "error": "quota_exceeded",
    "reason": "max_videos_per_month",
    "limit": 50,
    "current": 50,
    "message": "Project has reached its monthly video upload limit (50)."
  }
}
```

### Usage API endpoints

```bash
# Full usage summary (current month + all-time + storage)
GET /projects/{project_id}/usage

# Current calendar-month totals only
GET /projects/{project_id}/usage/current-month

# View quota configuration for a project
GET /projects/{project_id}/quotas
```

Example response for `GET /projects/1/usage`:

```json
{
  "project_id": 1,
  "current_month": {
    "year": 2025,
    "month": 4,
    "totals": {
      "videos_uploaded": 12,
      "frames_extracted": 2400,
      "clips_generated": 48,
      "storage_bytes_written": 524288000,
      "video_seconds_processed": 360,
      "webhook_deliveries": 24
    }
  },
  "alltime": {
    "videos_uploaded": 84,
    "storage_bytes_written": 3670016000
  },
  "storage_bytes_total": 3670016000
}
```

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


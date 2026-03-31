# MotionState — Video Motion Pipeline (MVP Scaffold)

> Turn video into structured human motion data.

## What this repo does

1. **Video ingest** – accepts uploaded video files via a REST API, stores them locally (S3/R2-ready interface).
2. **Normalization** – re-encodes uploads to a consistent format/resolution using FFmpeg.
3. **Queued processing** – enqueues a `process_video` job in Redis; a background worker picks it up.
4. **Structured motion/state output** – the worker extracts metadata, runs (stubbed) CV pipeline stages, and writes a time-indexed JSON artifact per video.

## What is NOT in scope (yet)

- No sport-specific ontology (no BJJ, no tennis, …)
- No coaching logic or scoring engine
- No real-time guarantees
- No frontend / product UI
- No real CV implementations (detector, tracker, pose estimator are abstract stubs)

## Architecture

```
┌──────────┐   POST /videos   ┌───────────┐   Redis queue   ┌────────────┐
│  Client  │ ──────────────►  │  FastAPI  │ ──────────────► │   Worker   │
└──────────┘                  │   (API)   │                  │  (Python)  │
                              └─────┬─────┘                  └─────┬──────┘
                                    │                               │
                              Postgres (metadata)           FFmpeg + CV stubs
                                    │                               │
                              ┌─────▼─────────────────────────────▼──────┐
                              │            Local filesystem               │
                              │  data/uploads/  data/normalized/          │
                              │  data/artifacts/{video_id}/state.json     │
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
| GET | `/jobs/{job_id}` | Get job status |

## Development

```bash
pip install -e ".[dev]"
make test      # run pytest
make lint      # ruff check
make format    # ruff format
```

## State artifact schema

Every processed video produces `data/artifacts/{video_id}/state.json`:

```json
{
  "video_id": "...",
  "version": 1,
  "segments": [],
  "tracks": [],
  "features": [],
  "notes": "placeholder artifact; CV pipeline not yet implemented"
}
```

## Next steps

1. Replace CV stubs with real detector/tracker/pose modules
2. Derive motion/state features from tracks + landmarks
3. Temporal segmentation
4. Schema hardening for queryable state output
5. Optional ontology layers on top

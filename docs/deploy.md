# Deploying MotionState

This guide covers everything needed to stand up a working MotionState
instance: minimum services, recommended startup order, storage backend
choices, auth bootstrap, and post-deploy smoke validation.

---

## Minimum services

| Service | Purpose | Required |
|---------|---------|---------|
| **PostgreSQL 14+** | Metadata (projects, videos, runs, artifacts, usage) | Yes |
| **Redis 7+** | Job queue; ephemeral, no persistence needed | Yes |
| **API process** (`apps/api`) | REST API, DB migrations on startup | Yes |
| **Worker process** (`apps/worker`) | Background job consumer | Yes |
| **Filesystem / S3-compatible bucket** | Artifact storage | Yes |

---

## Deployment order

Always start dependencies before the application processes:

```
1. postgres (wait for pg_isready)
2. redis    (wait for PONG)
3. api      (runs create_all on startup; wait for /health → {"status":"ok"})
4. worker   (consumes Redis queue; depends on postgres + redis but not api)
```

`docker compose up --build` handles this automatically via `depends_on` +
healthchecks.  For bare-metal or PaaS deployments, follow the order above.

---

## Quick start (local)

```bash
# 1. Clone and enter the repo
git clone https://github.com/kje7713-dev/MotionState.git
cd MotionState

# 2. Configure environment
cp .env.example .env
# Edit .env — set API_KEY_HMAC_SECRET and optionally ADMIN_TOKEN

# 3. Bootstrap (creates dirs, runs DB setup)
make bootstrap

# OR: bootstrap and seed a default project + API key in one step
SEED=1 make bootstrap

# 4. Start services
make dev          # docker compose up --build

# 5. Verify health
make verify-deploy
```

---

## Environment configuration

Three example files cover the common deployment modes:

| File | Use when |
|------|----------|
| `.env.local.example` | Local dev, local filesystem storage |
| `.env.s3.example` | Production / staging with S3 or Cloudflare R2 |
| `.env.webhook-dev.example` | Developing webhook integrations |

Copy the relevant file to `.env` and fill in the required values.

### Key variables

| Variable | Default | Notes |
|----------|---------|-------|
| `DATABASE_URL` | `postgresql+asyncpg://…` | Full asyncpg connection string |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |
| `API_KEY_HMAC_SECRET` | `change-me-in-production` | **Change in production** — `openssl rand -hex 32` |
| `ADMIN_TOKEN` | *(empty)* | Set to enable `/admin/*` endpoints — `openssl rand -hex 32` |
| `STORAGE_BACKEND` | `local` | `local` or `s3` |
| `S3_BUCKET` | *(empty)* | Required when `STORAGE_BACKEND=s3` |
| `S3_ENDPOINT_URL` | *(empty)* | Set for Cloudflare R2 or other S3-compatible stores |

---

## Storage backend choices

### Local filesystem (default)

Artifacts are written under `./data/` on the host.  Mount this directory
as a persistent volume when running in a container.

```env
STORAGE_BACKEND=local
ARTIFACTS_DIR=./data/artifacts
```

```yaml
# docker-compose snippet
volumes:
  - ./data:/app/data
```

### S3 / Cloudflare R2

```env
STORAGE_BACKEND=s3
S3_BUCKET=my-motionstate-bucket
S3_REGION=auto
S3_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com   # R2 only
S3_ACCESS_KEY_ID=<key>
S3_SECRET_ACCESS_KEY=<secret>
```

For AWS S3 leave `S3_ENDPOINT_URL` empty; set `S3_REGION` to the bucket
region (e.g. `us-east-1`).

---

## Auth and admin bootstrap

### Create a project and API key

The API itself has no super-admin UI.  Use `curl` or the bootstrap script:

```bash
# Option A: bootstrap script (seeds project + prints key)
SEED=1 make bootstrap

# Option B: manual via curl (API must be running)
curl -X POST "http://localhost:8000/projects?name=my-app"
# → {"id":1, "name":"my-app", …}

curl -X POST "http://localhost:8000/projects/1/api-keys?name=production"
# → {"id":1, "key":"ms_live_…", …}   ← save this, shown only once
```

### Enable the admin API

Set `ADMIN_TOKEN` to a random secret:

```bash
export ADMIN_TOKEN=$(openssl rand -hex 32)
```

Then access admin endpoints with the `X-Admin-Token` header:

```bash
curl "http://localhost:8000/admin/health/summary" \
     -H "X-Admin-Token: $ADMIN_TOKEN"
```

---

## Database migrations

The API creates all tables on startup via SQLAlchemy `create_all`.  This is
safe for fresh deployments and for adding new columns with `server_default`.
`create_all` is idempotent — it creates missing tables but does **not** modify
existing table structures.  Schema changes (e.g. adding columns to existing
tables) require an explicit Alembic migration.

For schema changes in production, use Alembic (already configured):

```bash
# Apply pending migrations
alembic upgrade head

# Generate a new migration after model changes
alembic revision --autogenerate -m "describe the change"
```

### Bootstrap script vs Alembic

| Scenario | Recommended approach |
|----------|---------------------|
| Fresh local dev | `make bootstrap` (runs `create_all`) |
| CI / test environment | `make bootstrap` |
| Production (first deploy) | `alembic upgrade head` |
| Production (schema update) | `alembic upgrade head` |

---

## Bootstrap script reference

```bash
# Local mode — relaxed checks
make bootstrap

# Local mode with project + API key seed
SEED=1 make bootstrap

# Staging mode — fails if insecure defaults are detected
make bootstrap-staging

# Direct invocation
python scripts/bootstrap.py --mode local --seed
python scripts/bootstrap.py --mode staging
```

The script:
1. Loads `.env` if present.
2. Checks required env vars (`DATABASE_URL`, `REDIS_URL`).
3. Warns about insecure defaults (`API_KEY_HMAC_SECRET`).
4. Creates local data directories.
5. Runs SQLAlchemy `create_all` to set up database tables.
6. Optionally seeds a `default` project and prints its API key.

---

## Post-deploy verification

```bash
# Basic health check
make verify-deploy

# With admin health check
ADMIN_TOKEN=<token> make verify-deploy

# With authenticated smoke check
ADMIN_TOKEN=<token> MOTIONSTATE_API_KEY=<key> SMOKE=1 make verify-deploy

# Against a remote host
MOTIONSTATE_BASE_URL=https://api.example.com ADMIN_TOKEN=<token> make verify-deploy
```

The script checks:
- `GET /health` — API is running.
- `GET /admin/health/summary` — DB, Redis, and storage are reachable (if `ADMIN_TOKEN` is set).
- Authenticated `/health` request — API key auth is working (if `--smoke` and key are provided).

---

## Generic container deployment notes

MotionState is a standard Python application.  It runs on any container
host that supports:
- A PostgreSQL 14+ database (managed or self-hosted)
- A Redis 7+ instance (managed or self-hosted)
- Persistent storage for `./data` (or an S3-compatible bucket)
- Two long-running processes: `api` and `worker`

### Environment variables

Pass all variables from `.env.example` as environment variables on your
container host.  Do **not** commit secrets to source control.

### Recommended startup sequence

1. Provision PostgreSQL and Redis first.
2. Set `DATABASE_URL` and `REDIS_URL`.
3. Deploy the `api` container; it will create DB tables on startup.
4. Deploy the `worker` container after the `api` health check passes.

### Health check endpoint

```
GET /health
→ {"status":"ok"}
```

Use this as the readiness/liveness probe for your container platform.

---

## Smoke test after deploy

```bash
# 1. Verify health
curl https://your-api.example.com/health

# 2. Create a project
curl -X POST "https://your-api.example.com/projects?name=smoke-test"

# 3. Generate API key
curl -X POST "https://your-api.example.com/projects/1/api-keys?name=smoke"

# 4. Upload a tiny test video (requires ffmpeg locally)
ffmpeg -f lavfi -i testsrc=duration=1:size=64x64:rate=1 /tmp/smoke.mp4
curl -X POST "https://your-api.example.com/videos" \
     -H "X-API-Key: ms_live_..." \
     -F "file=@/tmp/smoke.mp4"

# 5. Check admin health
curl "https://your-api.example.com/admin/health/summary" \
     -H "X-Admin-Token: $ADMIN_TOKEN"
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `CONNECTION_REFUSED` on `/health` | API container not running | Check container logs |
| `503` on `/health` | DB or Redis not reachable | Check `DATABASE_URL`/`REDIS_URL` and that services are up |
| `403` on `/admin/*` | `ADMIN_TOKEN` not set or wrong | Set/check `ADMIN_TOKEN` env var |
| Worker not processing jobs | Worker container not running | Start the worker process |
| Artifacts missing after processing | Storage misconfiguration | Check `STORAGE_BACKEND` and S3 credentials |
| `401 Unauthorized` on video endpoints | Missing/wrong API key | Pass `X-API-Key` header with a valid key |

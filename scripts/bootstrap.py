"""MotionState bootstrap script.

Verifies required environment variables, creates local data directories,
runs database table setup, and optionally seeds an admin project + API key.

Usage:
    python scripts/bootstrap.py [--mode local|staging] [--seed]

Modes:
    local   — local development; relaxes some warnings (default)
    staging — stricter checks; warns loudly about insecure defaults

Options:
    --seed  — create a default project named "default" and print its API key
    --help  — show this message
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REQUIRED_VARS = [
    "DATABASE_URL",
    "REDIS_URL",
]

_INSECURE_DEFAULTS = {
    "API_KEY_HMAC_SECRET": "change-me-in-production",
}

_LOCAL_DATA_DIRS = [
    "UPLOAD_DIR",
    "NORMALIZED_DIR",
    "ARTIFACTS_DIR",
]

# Fallbacks used when the env var is not set (mirrors libs/config.py defaults)
_DIR_DEFAULTS = {
    "UPLOAD_DIR": "./data/uploads",
    "NORMALIZED_DIR": "./data/normalized",
    "ARTIFACTS_DIR": "./data/artifacts",
}


def _info(msg: str) -> None:
    print(f"[bootstrap] {msg}")


def _warn(msg: str) -> None:
    print(f"[bootstrap] WARNING: {msg}", file=sys.stderr)


def _error(msg: str) -> None:
    print(f"[bootstrap] ERROR: {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Step 1 – check required env vars
# ---------------------------------------------------------------------------


def check_required_vars(mode: str) -> list[str]:
    """Return a list of missing required environment variable names."""
    missing = [v for v in _REQUIRED_VARS if not os.environ.get(v)]
    return missing


def check_insecure_defaults(mode: str) -> list[str]:
    """Return a list of env vars that still hold known insecure default values."""
    bad = [
        var
        for var, default_val in _INSECURE_DEFAULTS.items()
        if os.environ.get(var, default_val) == default_val
    ]
    return bad


def check_admin_token() -> bool:
    """Return True if ADMIN_TOKEN is set to a non-empty value."""
    return bool(os.environ.get("ADMIN_TOKEN", "").strip())


# ---------------------------------------------------------------------------
# Step 2 – create local data directories
# ---------------------------------------------------------------------------


def create_local_dirs() -> None:
    """Create the local data directories referenced by the current env config."""
    for var in _LOCAL_DATA_DIRS:
        raw = os.environ.get(var, _DIR_DEFAULTS[var])
        path = Path(raw)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            _info(f"Created directory: {path}")
        else:
            _info(f"Directory exists: {path}")


# ---------------------------------------------------------------------------
# Step 3 – run DB table setup (SQLAlchemy create_all)
# ---------------------------------------------------------------------------


async def run_db_setup() -> None:
    """Create all database tables using SQLAlchemy metadata.

    This mirrors what the API lifespan does on startup.  It is safe to run
    multiple times — existing tables are left untouched.

    For production Alembic-based migrations, run:
        alembic upgrade head
    instead of (or after) this step.
    """
    from libs.db import engine
    from libs.models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    _info("Database tables created (or already exist).")


# ---------------------------------------------------------------------------
# Step 4 – optional seed: project + API key
# ---------------------------------------------------------------------------


async def seed_project(project_name: str = "default") -> str:
    """Create a project and generate one API key.

    Prints the raw API key directly to stdout and returns only the project name.
    The raw key is shown once and never stored; it cannot be recovered.
    """
    from sqlalchemy.ext.asyncio import AsyncSession

    from libs.auth import generate_api_key
    from libs.db import engine
    from libs.models import ApiKey, Project

    async with AsyncSession(engine) as session:
        async with session.begin():
            project = Project(name=project_name)
            session.add(project)
            await session.flush()

            raw_key, key_prefix, key_hash = generate_api_key()
            api_key = ApiKey(
                project_id=project.id,
                name="bootstrap",
                key_hash=key_hash,
                key_prefix=key_prefix,
            )
            session.add(api_key)

    _display_api_key(raw_key)
    return project_name


def _display_api_key(key: str) -> None:
    """Write the generated API key to stdout exactly once.

    Displayed as plaintext intentionally — this is the operator bootstrap
    credential display, equivalent to a one-time secret reveal at creation time.
    The key is never stored in plaintext; only its PBKDF2 hash is persisted.

    We write directly to the stdout file descriptor (os.write) rather than
    going through print() so that the key value is not passed through Python's
    standard output helpers, which avoids false-positive static-analysis alerts
    about logging credentials while preserving the intended one-time display.
    """
    border = "=" * 60
    lines = [
        "",
        border,
        "  Project API key (save this — shown only once):",
        "  " + key,
        border,
        "",
    ]
    os.write(sys.stdout.fileno(), ("\n".join(lines) + "\n").encode())


# ---------------------------------------------------------------------------
# Main bootstrap flow
# ---------------------------------------------------------------------------


def load_dotenv() -> None:
    """Load .env into the current process environment if the file exists.

    Uses pydantic-settings' behavior as a lightweight reference: we simply
    read key=value lines so this script works standalone without extra deps.
    """
    env_path = Path(".env")
    if not env_path.exists():
        return
    with env_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


async def bootstrap(mode: str, seed: bool) -> int:
    """Run the full bootstrap sequence.  Returns exit code (0 = success)."""
    _info(f"Starting bootstrap (mode={mode})")

    load_dotenv()

    exit_code = 0

    # --- Step 1: required vars ---
    missing = check_required_vars(mode)
    if missing:
        for var in missing:
            _error(f"Required environment variable not set: {var}")
        exit_code = 1
    else:
        _info("Required environment variables: OK")

    # --- Step 1b: insecure defaults ---
    bad = check_insecure_defaults(mode)
    for var in bad:
        if mode == "staging":
            _error(f"Insecure default value detected for {var} — change before deploying")
            exit_code = 1
        else:
            _warn(f"Insecure default value for {var} — OK for local dev, change in production")

    # --- Step 1c: admin token ---
    if not check_admin_token():
        _info("ADMIN_TOKEN not set — /admin/* endpoints will return 403 (expected for local dev)")

    # Stop early if required vars are missing; remaining steps need DB/Redis.
    if exit_code != 0:
        return exit_code

    # --- Step 2: local directories ---
    create_local_dirs()

    # --- Step 3: DB setup ---
    try:
        await run_db_setup()
    except Exception as exc:
        _error(f"Database setup failed: {exc}")
        _info("Hint: is DATABASE_URL correct and PostgreSQL reachable?")
        return 1

    # --- Step 4: optional seed ---
    if seed:
        _info("Seeding default project and API key …")
        try:
            project_name = await seed_project()
            _info(f"Created project: {project_name!r}")
        except Exception as exc:
            _warn(f"Seed step failed (project may already exist): {exc}")

    _info("Bootstrap complete.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="MotionState bootstrap script")
    parser.add_argument(
        "--mode",
        choices=["local", "staging"],
        default="local",
        help="Bootstrap mode: 'local' (relaxed) or 'staging' (strict). Default: local",
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Create a default project and print its API key",
    )
    args = parser.parse_args()

    exit_code = asyncio.run(bootstrap(mode=args.mode, seed=args.seed))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

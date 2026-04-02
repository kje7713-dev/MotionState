"""MotionState post-deploy verification script.

Verifies that a running MotionState instance is healthy by:
1. Hitting the public /health endpoint.
2. Hitting /admin/health/summary (requires ADMIN_TOKEN to be set).
3. Optionally running a minimal authenticated API smoke check.

Usage:
    python scripts/verify_deploy.py [--base-url URL] [--admin-token TOKEN]
                                    [--api-key KEY] [--smoke]

Environment variables (used when CLI flags are not set):
    MOTIONSTATE_BASE_URL  — defaults to http://localhost:8000
    ADMIN_TOKEN           — used for /admin/health/summary
    MOTIONSTATE_API_KEY   — used for the optional smoke check

Exit code:
    0  — all requested checks passed
    1  — one or more checks failed
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _ok(msg: str) -> None:
    print(f"[verify] ✓ {msg}")


def _fail(msg: str) -> None:
    print(f"[verify] ✗ {msg}", file=sys.stderr)


def _info(msg: str) -> None:
    print(f"[verify] … {msg}")


# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------


def check_public_health(client: httpx.Client) -> bool:
    """GET /health — basic liveness check."""
    _info("Checking /health …")
    try:
        resp = client.get("/health", timeout=10)
        resp.raise_for_status()
        data: Any = resp.json()
        status = data.get("status", "")
        if status == "ok":
            _ok(f"/health → {data}")
            return True
        _fail(f"/health returned unexpected status: {data!r}")
        return False
    except Exception as exc:
        _fail(f"/health check failed: {exc}")
        return False


def check_admin_health(client: httpx.Client, admin_token: str) -> bool:
    """GET /admin/health/summary — deep dependency health check."""
    _info("Checking /admin/health/summary …")
    try:
        resp = client.get(
            "/admin/health/summary",
            headers={"X-Admin-Token": admin_token},
            timeout=15,
        )
        if resp.status_code == 403:
            _fail("/admin/health/summary returned 403 — is ADMIN_TOKEN set correctly?")
            return False
        resp.raise_for_status()
        data: Any = resp.json()
        all_ok = True
        for component, result in data.items():
            if result == "ok":
                _ok(f"  {component}: {result}")
            else:
                _fail(f"  {component}: {result}")
                all_ok = False
        if all_ok:
            _ok("/admin/health/summary — all components healthy")
        return all_ok
    except Exception as exc:
        _fail(f"/admin/health/summary check failed: {exc}")
        return False


def check_api_smoke(client: httpx.Client, api_key: str) -> bool:
    """Minimal authenticated API smoke check.

    Sends an authenticated GET /health request to confirm that API key
    authentication and end-to-end request handling are working correctly.
    """
    _info("Running authenticated API smoke check …")
    try:
        resp = client.get(
            "/health",
            headers={"X-API-Key": api_key},
            timeout=10,
        )
        resp.raise_for_status()
        _ok("Authenticated /health request succeeded")
        return True
    except Exception as exc:
        _fail(f"Authenticated smoke check failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_summary(results: dict[str, bool]) -> None:
    print()
    print("=" * 50)
    print("Verification summary:")
    for check, passed in results.items():
        mark = "✓" if passed else "✗"
        print(f"  {mark} {check}")
    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(description="MotionState deploy verification")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("MOTIONSTATE_BASE_URL", "http://localhost:8000"),
        help="Base URL of the running MotionState API (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--admin-token",
        default=os.environ.get("ADMIN_TOKEN", ""),
        help="Admin token for /admin/* endpoints (env: ADMIN_TOKEN)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("MOTIONSTATE_API_KEY", ""),
        help="API key for authenticated smoke check (env: MOTIONSTATE_API_KEY)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run authenticated API smoke check (requires --api-key or MOTIONSTATE_API_KEY)",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    results: dict[str, bool] = {}

    with httpx.Client(base_url=base_url) as client:
        results["public /health"] = check_public_health(client)

        if args.admin_token:
            results["admin /health/summary"] = check_admin_health(client, args.admin_token)
        else:
            _info("ADMIN_TOKEN not set — skipping admin health check")

        if args.smoke:
            if not args.api_key:
                _fail("--smoke requires --api-key or MOTIONSTATE_API_KEY to be set")
                results["authenticated smoke"] = False
            else:
                results["authenticated smoke"] = check_api_smoke(client, args.api_key)

    build_summary(results)

    if all(results.values()):
        _ok("All checks passed.")
        sys.exit(0)
    else:
        _fail("One or more checks failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()

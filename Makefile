.PHONY: dev test lint format down smoke \
        bootstrap bootstrap-local bootstrap-staging \
        verify-deploy

dev:
	docker compose up --build

down:
	docker compose down -v

test:
	pip install -e ".[dev]" -q
	pytest tests/ -v

smoke:
	pip install -e ".[dev]" -q
	pytest tests/ -v -m smoke

lint:
	ruff check .

format:
	ruff format .

# ---------------------------------------------------------------------------
# Bootstrap targets
# ---------------------------------------------------------------------------

# Bootstrap for local development (relaxed checks, optional seed).
# Pass SEED=1 to create a default project and print its API key.
bootstrap-local:
	@if [ "$(SEED)" = "1" ]; then \
		python scripts/bootstrap.py --mode local --seed; \
	else \
		python scripts/bootstrap.py --mode local; \
	fi

# Bootstrap for a staging-style environment (stricter checks).
# Fails if insecure defaults are detected.
bootstrap-staging:
	python scripts/bootstrap.py --mode staging

# Default bootstrap target — runs local mode.
bootstrap: bootstrap-local

# ---------------------------------------------------------------------------
# Post-deploy verification
# ---------------------------------------------------------------------------

# Verify a running MotionState instance is healthy.
# Reads MOTIONSTATE_BASE_URL and ADMIN_TOKEN from the environment.
# Pass SMOKE=1 to also run an authenticated API check (requires MOTIONSTATE_API_KEY).
verify-deploy:
	@if [ "$(SMOKE)" = "1" ]; then \
		python scripts/verify_deploy.py --smoke; \
	else \
		python scripts/verify_deploy.py; \
	fi

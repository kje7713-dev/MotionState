.PHONY: dev test lint format down

dev:
	docker compose up --build

down:
	docker compose down -v

test:
	pip install -e ".[dev]" -q
	pytest tests/ -v

lint:
	ruff check .

format:
	ruff format .

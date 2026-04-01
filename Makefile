.PHONY: dev test lint format down smoke

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

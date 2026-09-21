.PHONY: sync test lint fmt data

sync:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff format .

data:
	uv run python scripts/fetch_visdrone.py --split val

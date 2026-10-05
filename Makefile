.PHONY: sync sync-model test lint fmt data

sync:
	uv sync

sync-model:
	uv sync --extra model

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff format .

data:
	uv run python scripts/fetch_data.py visdrone
	uv run python scripts/fetch_data.py seadronessee

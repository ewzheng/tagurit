.PHONY: sync sync-model sync-jetson test lint fmt data

sync:
	uv sync

sync-model:
	uv sync --extra model

# torch only searches its own folder for cuDSS, so link the wheel's copy in.
# Rerun after anything reinstalls torch.
sync-jetson:
	uv sync --python 3.10 --extra model
	ln -sf ../../nvidia/cu12/lib/libcudss.so.0 .venv/lib/python3.10/site-packages/torch/lib/libcudss.so.0

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

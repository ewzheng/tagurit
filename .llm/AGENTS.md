# tagurit

15-821 Mobile Computing project. A robot or drone streams images to a cloudlet.
When the network drops, an on-device pipeline tags what matters and caches it.
When the link returns, the backlog drains alongside the live stream.

`architecture.md` in this directory has the module map, dataflow, and state machine.
Read it before touching `src/`.

## Layout

- `src/tagurit/` — Python package, one module per pipeline stage.
- `tests/` — pytest, one file per module (`tests/test_cache.py`, ...).
- `docs/specs/` — design docs, named `YYYY-MM-DD-<topic>-design.md`.
- `docs/plans/` — implementation plans.
- `scripts/` — dev helpers: trace tools, network emulation.
- `data/` — gitignored. Traces and sample images live here locally.
- `native/` — planned. C for the systems-level pieces, with its own Makefile.

## Commands

- `make sync` — install deps into `.venv` (`uv sync`)
- `make test` — `uv run pytest`
- `make lint` — `uv run ruff check .` and `ruff format --check .`
- `make fmt` — `uv run ruff format .`

## Conventions

- Python 3.12+, managed by `uv`. Add deps with `uv add`, not `pip install`.
- One module, one responsibility. Each package's `__init__.py` docstring states it.
- Data crossing module boundaries goes through `protocol.py`. Modules do not
  import each other's internals.
- Keep heavy imports (torch, ultralytics) out of package `__init__.py` files so
  `import tagurit` stays cheap and tests stay fast.
- Unit tests run offline: no network, no model weight downloads.
- `.llm/` is the single source of agent context. Root `AGENTS.md` is a symlink
  into it and root `CLAUDE.md` imports it. Edit the files here, not the root ones.

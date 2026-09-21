# tagurit

15-821 Mobile Computing project: data foraging in disconnected environments.
A robot or drone streams images to a cloudlet. When the network drops, an
on-device pipeline tags what matters and caches it. When the link returns, the
backlog drains alongside the live stream.

`architecture.md` in this directory has the module map, dataflow, state machine,
and stack. Read it before touching `src/`. The checkpoint 1 deck in
`docs/materials/` is the source for both.

## Goals

- A tagging and caching system light enough to run on the edge device (NVIDIA
  Jetson Orin NX 8GB), demoed end to end against a cloudlet.
- Every frame gets a priority score on the device. That score is what the cache
  and the transmission scheduler act on.
- Then experiment: caching and transmission policies, and caching embeddings
  instead of raw images.

## Milestones

| Milestone | Deliverable |
|---|---|
| Checkpoint 1 (done) | Goals and architecture |
| Checkpoint 2 | Working prototype with prioritized caching |
| Final | Experiments: embedding transmission, live learning on the cloudlet, transmission and compression |

Fallback tiers, in order. Finish one before starting the next.

1. Stream everything while connected, cache everything while disconnected,
   interleave cached and live while reintegrating.
2. A tagging heuristic and a cache inclusion policy.
3. The experiments above.

## Layout

- `src/tagurit/` — Python package, one module per pipeline stage.
- `tests/` — pytest, one file per module (`tests/test_cache.py`, ...).
- `docs/specs/` — design docs, named `YYYY-MM-DD-<topic>-design.md`.
- `docs/plans/` — implementation plans.
- `docs/materials/` — course deliverables: checkpoint decks and the like.
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
- Code and docstring style is in `style.md` in this directory. Read it before writing code.
- `.llm/` is the single source of agent context. Root `AGENTS.md` is a symlink
  into it and root `CLAUDE.md` imports it. Edit the files here, not the root ones.

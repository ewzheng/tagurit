# tagurit

15-821 Mobile Computing project: data foraging in disconnected environments.
A robot or drone streams images to a cloudlet. When the network drops, an
on-device pipeline tags what matters and caches it. When the link returns, the
backlog drains alongside the live stream.

`architecture.md` in this directory has the current module map, dataflow,
state machine and stack. Read it before touching `src/`.
The checkpoint 1 deck in `docs/materials/` provides the original project design.

## Goals

- A tagging and caching system light enough to run on the edge device (NVIDIA
  Jetson Orin NX 8GB), demoed end to end against a cloudlet.
- Score frames on the device and use those scores to guide prioritized retention
  and transmission. The current frame type also permits unscored live frames;
  frames entering the priority bank must have a score.
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

- `src/tagurit/` — Python package
  - `protocol.py` — shared image handoff datatype
  - `model/` and `tagging/` — perception and priority scoring
  - `client/` — queues, scheduling, operating states and Gabriel transmission
  - `cloudlet/` — image receiver, duplicate tracking and receipts
  - `shared/` — wire message types, encoding, decoding and hashes
  - `sim/` — sample image input, scheduled connectivity and demo entry point
  - `orchestrator.py` — reserved for future top-level integration
- `tests/` — pytest checks for package imports, client, cloudlet and simulation
- `docs/specs/` — design docs, named `YYYY-MM-DD-<topic>-design.md`
- `docs/plans/` — implementation plans
- `docs/materials/` — course deliverables, including checkpoint decks
- `scripts/` — dev helpers, trace tools and network emulation
- `data/` — gitignored local traces, sample images and manifests
- `native/` — planned C implementations for selected systems components

## Commands

- `make sync` — install dependencies into `.venv` using `uv sync`
- `make test` — `uv run pytest`
- `make lint` — check lint rules and formatting
- `make fmt` — `uv run ruff format .`
- `uv run python3 -m tagurit.cloudlet.image_receiver` — start the receiver
- `uv run python3 -m tagurit.sim.run_client` — start the sample-image client demo

The demo requires `data/demo/manifest.csv` and its referenced images.
These files are local and are not included in a fresh clone.

## Conventions

- Python 3.12+, managed by `uv`. Add dependencies with `uv add`.
- One module, one responsibility. Each package's `__init__.py` docstring states it.
- `protocol.py` defines the perception-to-client handoff. Keep Gabriel messages,
  receipts and wire encoding out of it.
- `shared/image_protocol.py` defines the wire contract used by both client
  and cloudlet.
- Keep scheduler states and internal priority bank entries in
  `client/scheduler_datatypes.py`.
- The scheduler owns queues, arbitration, state transitions and the unresolved
  frame. The transport reports communication availability and validated receipts.
- Retain unresolved images until acknowledged. Retries preserve their identity
  and contents; the receiver suppresses duplicate acceptance.
- `client/` must not import `sim/`. The demo supplies a frame source to the client;
  the future orchestrator will supply real input through the same interface.
- Use public interfaces instead of importing another component's private internals.
- Keep heavy imports such as torch and ultralytics out of package `__init__.py`
  files so `import tagurit` stays cheap and tests stay fast.
- Unit tests run offline with temporary or generated inputs: no network,
  model weight downloads or dependency on local demo files.
- Code and docstring style is in `style.md` in this directory.
  Read it before writing code.
- Known limitations are documented in `architecture.md`.
- `.llm/` is the single source of agent context. Root `AGENTS.md` is a symlink
  into it and root `CLAUDE.md` imports it. Edit the files here, not the root ones.
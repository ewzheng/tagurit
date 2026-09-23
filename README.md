# tagurit

15-821 Mobile Computing project.

A robot or drone streams images to a cloudlet. When the network drops, an
on-device pipeline tags what matters and caches it. When the link returns, the
backlog drains alongside the live stream.

## Setup

```
uv sync
make test
```
## Layout

- `src/tagurit/`
  - `protocol.py` — shared image handoff datatype
  - `client/` — queues, scheduling and Gabriel transmission
  - `cloudlet/` — image receiver and duplicate tracking
  - `shared/` — wire messages, encoding and receipts
  - `model/` and `tagging/` — perception and priority scoring
  - `sim/` — sample image input and simulation helpers
  - `orchestrator.py` — reserved for top-level integration
- `data/` — local demo images and manifest, gitignored
- `tests/` — pytest
- `docs/` — design specs and implementation plans
- `scripts/` — dev helpers
- `.llm/` — shared context for coding agents

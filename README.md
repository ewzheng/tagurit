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

- `src/tagurit/` — Python package, one module per pipeline stage. See `.llm/architecture.md`.
- `tests/` — pytest
- `docs/` — design specs and implementation plans
- `scripts/` — dev helpers
- `.llm/` — shared context for coding agents (Codex, Claude Code)

# Architecture

## Pipeline

```
images ──> model ──> tagging ──> cache ──> transfer ──> cloudlet
                                             ▲
                                            link
              sim wraps everything with traces and metrics
```

Under full connectivity, images skip `model` and `tagging` and go straight to
`transfer`. The ML stages only run when the link is down.

## Modules (`src/tagurit/`)

| Module | Responsibility |
|---|---|
| `protocol.py` | Data contract: image + tags + priority in memory, and its wire encoding to the cloudlet |
| `orchestrator.py` | The three-state machine and the routing each state implies |
| `model/` | Wraps the perception model (YOLO now, a trained model later); consumes images, emits detections |
| `tagging/` | Heuristics and ML over detections; decides what to tag and how urgent it is |
| `cache/` | The image bank: persistence, priority ordering, eviction |
| `link/` | Connectivity monitoring: link up/down events, bandwidth estimate |
| `transfer/` | Sends to the cloudlet: the live stream plus the backlog drain, and how they share the link |
| `cloudlet/` | Receiver: accept, unpack, later decompress |
| `sim/` | Trace replay, disconnection scenarios, metrics |

## States

| State | New images | Cache |
|---|---|---|
| Connected | straight to `transfer`, bypassing `model` and `tagging` | idle |
| Disconnected | `model` → `tagging` → `cache` | filling |
| Reintegrating | straight to `transfer` | draining alongside the live stream |

Transitions are driven by `link` events:

- Connected → Disconnected on link down
- Disconnected → Reintegrating on link up
- Reintegrating → Connected once the cache is drained
- Reintegrating → Disconnected on link down

## Deferred

- `capture/`: until real hardware exists, every input is a trace replayed by `sim/`.
- Compression: a hook in `transfer/` with its inverse in `cloudlet/`, agreed through `protocol.py`.
- Multimodal input: images only for now.
- `native/`: C for the systems-level pieces.

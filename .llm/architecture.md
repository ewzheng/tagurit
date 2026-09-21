# Architecture

Source: the checkpoint 1 deck, `docs/materials/DataForaging_Checkpoint1Presentation.pdf`.
Vocabulary follows it. *Data foraging* is the whole problem, the cache is the
*hoard*, and draining it after a link returns is *reintegration*.

## Pipeline

```
images ──> model ──> tagging ──> [route by state] ──> transfer ──> cloudlet
                                        │                 ▲
                                        └────> cache ─────┘
                                                 (drains while reintegrating)

link drives the state. sim wraps everything with traces and metrics.
```

`model` and `tagging` run in every state. Every frame leaves `tagging` as a
**scored frame**: image + priority + metadata, defined in `protocol.py`. That
scored frame is the boundary between perception and everything downstream.
Downstream decides whether to keep it, when to send it, and how.

## Modules (`src/tagurit/`)

| Module | Responsibility |
|---|---|
| `protocol.py` | The scored frame in memory (image + tags + priority + metadata) and its wire encoding to the cloudlet |
| `orchestrator.py` | The three-state machine and the routing each state implies |
| `model/` | Wraps the perception model: YOLO pretrained on COCO now, a trained model later. Consumes images, emits detections. Must stay light enough for the edge device |
| `tagging/` | Decides what counts as "interesting" and emits a per-frame priority score. Heuristics first, ML later. Evaluated on tagging accuracy and recall |
| `cache/` | The hoard: admission policy (the deck's "Policy" box), persistence, priority ordering, eviction |
| `link/` | Connectivity monitoring: link up/down events, bandwidth estimate |
| `transfer/` | The deck's "Arbitration / Transmission Policy": the live stream plus the backlog drain, and how they share the link. Gabriel client side |
| `cloudlet/` | Receiver, Gabriel server side: accept, unpack, later decompress |
| `sim/` | Trace replay, disconnection scenarios, metrics |

## States

| State | Deck name | New images | Cache |
|---|---|---|---|
| Connected | Full Connection | `model` → `tagging` → `transfer` | idle |
| Disconnected | Disconnected | `model` → `tagging` → `cache` | filling |
| Reintegrating | Reintegration | `model` → `tagging` → `transfer` | draining alongside the live stream |

Transitions are driven by `link` events:

- Connected → Disconnected on link down
- Disconnected → Reintegrating on link up
- Reintegrating → Connected once the cache is drained
- Reintegrating → Disconnected on link down

## Stack

- Edge: NVIDIA Jetson Orin NX 8GB, standing in for the drone or robot.
- Cloudlet: a remote compute node for heavier processing. Access is arranged.
- Model: YOLO pretrained on COCO.
- Middleware: Gabriel, the CMU SatyaLab edge-cloud framework. Client on the
  edge, server on the cloudlet. Lives in `transfer/` and `cloudlet/`.

## Deferred

- `capture/`: the edge target is known, but until the camera path exists every
  input is a trace replayed by `sim/`.
- Compression: a hook in `transfer/` with its inverse in `cloudlet/`, agreed
  through `protocol.py`. Later adaptive to available bandwidth.
- Embeddings: cache and transmit embeddings instead of raw images, with lookup
  on the cloudlet. A final-milestone experiment.
- Live learning: the cloudlet retrains the model on a new class while running.
- Bandwidth-aware transmission: detect weak links and adapt rate and compression.
- A custom transport replacing Gabriel.
- Multimodal input: images only for now.
- `native/`: C for the systems-level pieces.

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
| `protocol.py` | Shared image handoff datatype and priority validation |
| `orchestrator.py` | Reserved for future top-level camera, model and client integration |
| `model/` | Tile-scoring models behind `scorer.TileScorer`: anomalib PatchCore, zero-shot CLIP through `transformers`, shared tile preprocessing, tiling, device choice; owned by Elias |
| `tagging/` | Frame priority from the most unusual tile, a non-saturating score mapping, and saved bundles; owned by Elias |
| `client/frame_scheduler.py` | State transitions, live FIFO queue, priority bank, arbitration and in-flight frame ownership |
| `client/scheduler_datatypes.py` | Operating states and internal priority bank entries |
| `client/gabriel_transport.py` | One Gabriel image producer, receipt matching, timeouts, retries and communication-health reporting |
| `client/main.py` | Runs the scheduler and transport with a caller-supplied asynchronous frame source |
| `cloudlet/` | Gabriel receiver, image validation, duplicate tracking and receipts |
| `shared/image_protocol.py` | Wire message and receipt types, encoding, decoding and content hashes |
| `shared/image_codec.py` | Software H.264/H.265 encoding and decoding of independent frames |
| `sim/` | Manifest image feeder, scheduled connectivity, console demo, dataset trace loaders (box-annotated and frame-labelled), synthetic sparse crops, normal-tile sampling and ranking metrics |

`sim/run_client.py` starts the demo by supplying sample frames to
`client/main.py`. The future orchestrator will supply real frames through
the same interface without depending on `sim/`.

## Tagging

A frame is decoded and cut into a grid of square tiles (512 px by default),
because aerial targets are a few dozen pixels across and would vanish if the
whole frame were resized to the model's input. A `TileScorer` gives each tile
a raw score. The frame's raw score is its highest tile score, and
`s / (s + reference)` maps it into [0, 1). The mapping never saturates, so
the bank's highest-score-first order is preserved instead of collapsing into
ties at 1.0. `reference` is a high percentile of scores on target-free tiles,
so 0.5 means "as unusual as the most unusual normal tiles".

The current scorer is PatchCore through anomalib: a frozen WideResNet-50
embeds patches, and a tile's score is its largest distance to a memory bank
of patches from normal tiles. `scripts/fit_patchcore.py` builds the bank
from tiles that no ground-truth box comes near and saves a bundle directory;
`scripts/score_trace.py` scores held-out frames, reports frame-level and
tile-level ranking metrics, and can write a manifest the client demo replays. Fitting holds
every embedding in memory, so it runs on a laptop or the cloudlet; the edge
device only loads a bundle and scores. Fitting always runs in float32; a
bundle can score in float16, which matters on tensor-core GPUs such as the
Jetson's and changes scores by well under one percent.

Evaluation needs only frame-level labels: does a frame hold something worth
sending. `TraceFrame.target` carries such a label, and `TraceFrame.has_target`
falls back to boxes for box-annotated datasets. `sim/framelist.py` reads
frame-labelled sequences from folders with a `frames.csv`. Public aerial
datasets put a target in almost every frame, so `scripts/make_sparse.py`
builds sparse streams from them by cutting fixed-size crops of real frames:
mostly windows no box comes near, and a set share that hold one whole target.

## States

| State | Incoming frames | Transmission |
|---|---|---|
| Connected | Enter the live FIFO queue | Send live frames in arrival order |
| Disconnected | Enter the priority bank | Select no new frames for transmission |
| Reintegrating | Enter the live FIFO queue | Share turns between live traffic and the priority bank |

The priority bank selects the highest score first, with arrival order
breaking ties. The default reintegration ratio is two live turns per
stored turn. Stored traffic can use a live turn when the live queue is empty.

The transport reports communication availability to the scheduler:

- Connection errors, closed connections or receipt timeouts trigger Disconnected
- Successful Gabriel registration restores availability
- Restored availability triggers Reintegrating if stored work remains,
  otherwise Connected
- Reintegrating returns to Connected after the final stored frame is acknowledged
- Another communication failure returns Reintegrating to Disconnected

A stored frame awaiting acknowledgment still counts as stored work.

The current demo uses actual transport health. The schedule in
`sim/connectivity.py` is available for separate simulation tests.

## Delivery and recovery

Only one image is unresolved at a time. Selecting or sending a frame does
not release it from the scheduler.

Each transmission includes a session UUID, frame ID and image hash.
The client releases a frame only after receiving a matching receipt.
The current default transcodes each JPEG to one independent H.264 frame at
CRF 32. The wire header identifies its codec and original dimensions. The
encoded bytes are retained unchanged across retries, and the cloudlet decodes
them before returning a receipt.

Connection failures and receipt timeouts cause the client to clean up the
old connection, wait, and reconnect. The unresolved image is retried before
new work, using the same identity and contents.

The receiver validates and decodes a new image, then retains it and its
receipt before returning the acknowledgment. Matching retries return the
original receipt without accepting another copy. Reusing an identity with
different contents is rejected.

Queues and accepted-image records are memory-only. Delivery recovery assumes
both applications remain running, sufficient memory remains available, and
communication eventually allows the retained work to be delivered.

## Stack

- Edge target: NVIDIA Jetson Orin NX 8GB; currently a laptop
- Cloudlet target: remote compute node; currently tested locally
- Models: PatchCore anomaly scoring through anomalib, and zero-shot CLIP through
  Hugging Face `transformers` (optional `model` extra). New models load through
  `transformers` rather than model-specific packages
- Middleware: Gabriel client and server using WebSockets over TCP
- Python: 3.12+, with dependencies managed by uv

## Known limitations

- Frames arriving before an outage is detected enter the live queue and remain
  there after disconnection is detected; moving or scoring this backlog is unresolved
- A receipt timeout cannot distinguish network failure from slow transmission
  or slow server processing
- Connection quality and available bandwidth are not yet measured
- Stopping either application loses its in-memory records
- The repeating demo stops immediately on Ctrl+C and may leave queued frames
- The receiver accepts images but does not yet run a heavy model
- Tagging is blocking compute; the orchestrator must call it off the asyncio
  loop, for example with `asyncio.to_thread`, or it stalls transport
- The priority reference is calibrated on tiles, while frames take the
  maximum over many tiles, so target-free frames land somewhat above 0.5
- PatchCore treats anything absent from its bank as unusual: new terrain,
  lighting or camera altitude raises every score until the bank is refitted
- The sparse streams are synthetic: crops are smaller than real frames, empty
  crops may hold unannotated targets, and they share scenes with the fit data
- On the Jetson's JetPack 6 the only Python 3.12 CUDA PyTorch build is torch
  2.8 from the Jetson AI Lab `jp6/cu129` index; it is untested on the device

## Deferred

- Real camera input, and wiring `tagging.Tagger` into the frame source
- Top-level integration through `orchestrator.py`
- Jetson hardware encoding and adaptive codec or quality selection
- Embedding transmission
- Live learning on the cloudlet
- Bandwidth-aware scheduling and transmission
- A custom transport replacing Gabriel
- Multimodal input beyond images
- Native implementations for selected systems components

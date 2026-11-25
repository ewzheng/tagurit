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
| `protocol.py` | Shared image handoff datatype, priority validation, and the opaque state a tagger keeps with a frame for rescoring |
| `orchestrator.py` | Edge pipeline: stamps captured image bytes, tags the newest one off the event loop into an `ImageFrame` whenever the tagger is free, dropping images that go stale while it is busy, and runs the client on the tagged frames |
| `model/` | Tile-scoring models behind `scorer.TileScorer`, and `scorer.WindowedTileScorer` for scorers that remember recent frames: anomalib PatchCore, query-only CLIP with window novelty through `transformers`, the novelty window and its scale, shared tile preprocessing, tiling, device choice; owned by Elias |
| `tagging/` | Frame priority from the most unusual tile, capture times passed to windowed scorers, a non-saturating score mapping, saved bundles, and the time-aware send order for banked frames (`order`); owned by Elias |
| `client/frame_scheduler.py` | State transitions, live FIFO queue, priority bank with an optional caller-supplied send order, arbitration and in-flight frame ownership |
| `client/scheduler_datatypes.py` | Operating states, internal priority bank entries and the bank order's signature |
| `client/gabriel_transport.py` | One Gabriel image producer, receipt matching, timeouts, retries and communication-health reporting |
| `client/main.py` | Runs the scheduler and transport with a caller-supplied asynchronous frame source |
| `cloudlet/` | Gabriel receiver, image validation, duplicate tracking and receipts |
| `shared/image_protocol.py` | Wire message and receipt types, encoding, decoding and content hashes |
| `shared/image_codec.py` | Software H.264/H.265 encoding and decoding of independent frames |
| `sim/` | Manifest image feeder, scheduled connectivity, console demo, dataset trace loaders (box-annotated and frame-labelled), synthetic sparse crops, normal-tile sampling and ranking metrics |

`sim/run_client.py` starts the queueing demo by supplying pre-scored sample
frames to `client/main.py`. `sim/run_pipeline.py` is the end-to-end demo: it
replays a dataset as captured image bytes into `orchestrator.run`, which tags
each one with a bundle and feeds the same client, while `sim/pipeline_view.py`
draws a live dashboard of the latest frame and the bank and live queue in
send order, the latest frame sent marked, optionally recorded to MP4. The orchestrator takes
images from any source and never imports `sim/`, so a camera source plugs in
the same way.

## Tagging

A frame is decoded and cut into a grid of square tiles (512 px by default
for PatchCore, 224 px for CLIP), because aerial targets are a few dozen
pixels across and would vanish if the whole frame were resized to the
model's input. A `TileScorer` gives each tile
a raw score. The frame's raw score is its highest tile score, and
`s / (s + reference)` maps it into [0, 1). The mapping never saturates, so
the bank's highest-score-first order is preserved instead of collapsing into
ties at 1.0. `reference` is a high percentile of scores on target-free tiles,
so 0.5 means "as unusual as the most unusual normal tiles".

The CLIP scorer (`model/clip.py`) embeds each tile once and scores it on two
terms. The query term is the tile's best dot product with "an aerial photo
of {phrase}." minus "an aerial photo.", with no reject phrases. The novelty
term is its distance to the nearest tile of an earlier frame in the last 10
s (`model/novelty.py`), so a target that stays in view fades. On arrival
each term becomes a percentile: the query against target-free calibration
tiles saved in the bundle, novelty against a sample of the flight's earlier
novelty. A tile scores the mean of the two percentiles, as a non-saturating
log-odds. `Tagger.tag` passes each frame's capture time to such a windowed
scorer, so frames must be tagged in capture order, and `Tagger.reset`
starts a new flight. The scorer keeps the text encoder, so `set_targets`
switches the query on board, and `retarget` rescores stored tile
embeddings and novelty percentiles against a new query with one matrix
product. `scripts/calibrate_clip.py` builds a bundle.

A mission change is a new set of query phrases. `orchestrator.run` takes
them as `missions`: for each one it switches the tagger's query
(`Tagger.set_targets`), so frames tagged from then on use it, and hands the
client a rescorer. The client gives the rescorer every frame still waiting
in the live queue or the bank and replaces their priorities
(`FrameScheduler.reprioritize`), then again for any frame that arrived
meanwhile. Rescoring reads no image: a CLIP-tagged `ImageFrame` carries
`kept`, the frame's tile embeddings in half precision and the novelty its
tiles had on arrival, about 25 KB for 24 tiles, and `Tagger.rescore` turns
those into new priorities with one matrix product (`clip.retarget`). The
client holds `kept` as long as it holds the frame and never looks inside
or transmits it. Encounters regroup on their own, since the bank order
derives them from priorities on every turn.

`tagging/order.py` says which banked frame to send next, from capture
times and priorities alone. A frame at or above the bundle's detection
threshold (`Tagger.detection`) is a detection, and detections at most 10 s
apart form an encounter. Detections go first, in passes: each encounter
gets one frame, its best, before any gets a second, and the newest
encounter goes first within a pass. A recent sighting is sent before an
old one, and an old sighting seen once still goes before any repeat and
before every frame below the threshold; those follow by priority.
Encounters are derived from the frames on every call, so rescoring the
bank for a new query regroups them. `calibrate_clip.py` sets the threshold
at the 99.5th percentile of the priorities of target-free frames, scored
in capture order as the scout would.

PatchCore runs through anomalib: a frozen WideResNet-50
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
mostly windows no target box comes near, and a set share that hold one whole
target. Targets are every labelled object by default, or only the classes
given with `--classes`, so a stream for one query keeps other objects in its
empty crops.

## States

| State | Incoming frames | Transmission |
|---|---|---|
| Connected | Enter the live FIFO queue | Send live frames in arrival order |
| Disconnected | Enter the priority bank | Select no new frames for transmission |
| Reintegrating | Enter the live FIFO queue | Share turns between live traffic and the priority bank |

The priority bank selects the highest score first, with arrival order
breaking ties. When the orchestrator's caller supplies a bank order, the
bank asks it instead on every stored turn, passing the capture time and
priority of each waiting frame and of every frame already acknowledged
from either queue. The demo supplies `tagging.order.next_frame` whenever
the bundle has a detection threshold, so the bank sends one frame per
encounter before any repeat, newest encounter first, and a detection that
already went out live counts towards its encounter. The scheduler never
imports `tagging/`; the order is passed in. The default reintegration ratio is two live turns per
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
- Models: PatchCore anomaly scoring through anomalib, and query-only CLIP with
  window novelty through Hugging Face `transformers` (optional `model` extra).
  New models load through `transformers` rather than model-specific packages
- Middleware: Gabriel client and server using WebSockets over TCP
- Python: 3.10+, with dependencies managed by uv. The Jetson runs 3.10,
  the only Python with CUDA PyTorch on JetPack 6

## Known limitations

- Frames arriving before an outage is detected enter the live queue and remain
  there after disconnection is detected; moving or scoring this backlog is unresolved
- A receipt timeout cannot distinguish network failure from slow transmission
  or slow server processing
- Connection quality and available bandwidth are not yet measured
- Stopping either application loses its in-memory records
- The repeating demo stops immediately on Ctrl+C and may leave queued frames
- The receiver accepts images but does not yet run a heavy model
- The orchestrator drops an image only for being stale or, with
  `min_interval`, too soon after the last one tagged; it does not test
  whether the picture changed. Consecutive drone frames always differ
  somewhere by more than a small target would (measured on VisDrone and
  SeaDronesSee), so a pixel-change test that is safe for small targets
  would drop nothing in flight
- The scheduler keeps the capture time and priority of every acknowledged
  frame for the bank order and reads the whole bank on each stored turn,
  which is fine for thousands of frames and unbounded beyond that
- The priority reference is calibrated on tiles, while frames take the
  maximum over many tiles, so target-free frames land somewhat above 0.5
- PatchCore treats anything absent from its bank as unusual: new terrain,
  lighting or camera altitude raises every score until the bank is refitted
- CLIP's novelty is neutral for a flight's first 200 tiles, until its scale
  has a sample, so those frames are ranked by the query alone
- A mission change reaches the scout only through the demo's `--retarget`
  flag; the cloudlet cannot send one yet. The priority scale and detection
  threshold are kept across a query change, which relies on the query term
  being rescaled against the bundle's calibration tiles and is untested
- Acknowledged frames give up their kept embeddings, so after a query
  change the bank order forgets what was already delivered and every
  encounter is owed a first frame again
- PatchCore bundles, and CLIP bundles built before the threshold was
  added, carry no detection threshold; with those the bank sends by
  priority alone
- The demo stamps frames with replay times, not the dataset's, so a replay faster than one
  frame every few seconds puts all its detections in one encounter
- An empty frame above the detection threshold within 10 s of two
  encounters joins them into one, so a low threshold merges encounters; the
  sparse streams, with several crops per moment, do this more than a real
  1 fps stream would
- The send order was tested only on SeaDronesSee, whose targets are counted
  by the same 10 s rule that defines an encounter, and never on a mission
  long enough for a target to be seen again hours later
- The sparse streams are synthetic: crops are smaller than real frames, empty
  crops may hold unannotated targets, and they share scenes with the fit data
- Empty crops that clear every object sit away from where objects gather, off
  the streets in a city, so scenery alone partly separates them from target
  crops; per-class streams (`make_sparse.py --classes`) let empty crops keep
  other objects
- The Jetson's JetPack 6 has CUDA PyTorch only for Python 3.10, so the
  Jetson and the laptop run different torch releases. The Jetson AI Lab's
  Python 3.12 build needs glibc 2.38 (Ubuntu 22.04 has 2.35), and PyPI's
  aarch64 wheels need a newer CUDA driver and fall back to the CPU

## Deferred

- Real camera input as an orchestrator image source
- Jetson hardware encoding and adaptive codec or quality selection
- Embedding transmission
- Live learning on the cloudlet
- Bandwidth-aware scheduling and transmission
- A custom transport replacing Gabriel
- Multimodal input beyond images
- Native implementations for selected systems components

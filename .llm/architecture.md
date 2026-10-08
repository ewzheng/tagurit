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
| `model/` | Perception model integration, owned by Elias |
| `tagging/` | Converts detections into frame priority scores, owned by Elias |
| `client/frame_scheduler.py` | State transitions, live FIFO queue, priority bank, arbitration and in-flight frame ownership |
| `client/scheduler_datatypes.py` | Operating states and internal priority bank entries |
| `client/gabriel_transport.py` | One Gabriel image producer, receipt matching, timeouts, retries and communication-health reporting |
| `client/main.py` | Runs the scheduler and transport with a caller-supplied asynchronous frame source |
| `cloudlet/` | Gabriel receiver, image validation, duplicate tracking and receipts |
| `shared/image_protocol.py` | Wire message and receipt types, encoding, decoding and content hashes |
| `shared/telemetry.py` | Optional CSV event recording for timing experiments |
| `sim/flow_metrics.py` | Per-frame and aggregate baseline timing summaries |
| `shared/image_codec.py` | Software H.264/H.265 encoding and decoding of independent frames |
| `sim/` | Manifest image feeder, optional scheduled connectivity and console demo |

`sim/run_client.py` starts the demo by supplying sample frames to
`client/main.py`. The future orchestrator will supply real frames through
the same interface without depending on `sim/`.

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

The fixed client window is configurable and defaults to two reserved images.
It counts both encoding and awaiting-receipt work. Each frame remains in a map
owned by the scheduler until its matching receipt; acknowledgments may arrive
out of order. Gabriel grants a separate server credit budget at registration.
The effective sending limit is bounded by both budgets. A validated receipt
releases exactly one application slot; invalid receipts halt the transport.
Default artificial send pacing is zero.

Each transmission includes a session UUID, frame ID and image hash.
The client releases a frame only after receiving a matching receipt.
The current default transcodes each JPEG to one independent H.264 frame at
CRF 32. The wire header identifies its codec and original dimensions. The
encoded bytes are retained unchanged across retries, and the cloudlet decodes
them before returning a receipt.

Connection failures and receipt timeouts cause the client to clean up the
old connection, wait, and reconnect. All unresolved images are replayed in reservation order before new work, using
the same identities and encoded bytes. Each submitted frame has its own receipt
deadline. Reconnection resets connection credits and submission times, while
retaining payloads; callbacks from old connections are ignored.

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
- Model target: lightweight YOLO with COCO classes
- Middleware: Gabriel client and server using WebSockets over TCP
- Python: 3.12+, with dependencies managed by uv

## Timing experiments

`scripts/benchmark_flow_control.py` runs a finite, preloaded JPEG sample with
explicit test priorities against a separately started cloudlet. Client event
logging and the send interval are optional parameters of `run_client`; the
regular demo accepts `--window` and `--send-interval`. The baseline sets the interval to
zero and accepts `--window N` (default one for baseline compatibility). Raw events, input hashes, settings,
per-frame delays and summary statistics are saved under gitignored `data/`.
The cloudlet's optional `--events` CSV records acceptance using the same session
and frame identity without changing the image wire format.

Queue delay ends at scheduler selection; encoding is measured separately.
Send-to-receipt starts at the encoded payload handoff to Gabriel, after encoding,
and includes transport scheduling and cloudlet decoding. It is not network RTT.
Durations use client monotonic time, never subtraction of server/client clocks.
Retries preserve bytes and the original arrival event; retry-free latency
percentiles exclude them. Incomplete runs retain unfinished rows.

The cloudlet accepts `--tokens N` and `--queue-size N`. Its queue defaults to
at least N and rejects smaller explicit values. Capacity sizing assumes one
producer; additional clients share the server queue and may cause overload.
The server allowance must be at least the desired client window to exercise
that window fully. No dynamic window controller is implemented yet.

## Known limitations

- Frames arriving before an outage is detected enter the live queue and remain
  there after disconnection is detected; moving or scoring this backlog is unresolved
- A receipt timeout cannot distinguish network failure from slow transmission
  or slow server processing
- Connection quality and available bandwidth are not yet measured
- Stopping either application loses its in-memory records
- The repeating demo stops immediately on Ctrl+C and may leave queued frames
- The receiver accepts images but does not yet run a heavy model

## Deferred

- Real camera input and integration with Elias's model and tagging
- Top-level integration through `orchestrator.py`
- Jetson hardware encoding and adaptive codec or quality selection
- Embedding transmission
- Live learning on the cloudlet
- Bandwidth-aware scheduling and transmission
- A custom transport replacing Gabriel
- Multimodal input beyond images
- Native implementations for selected systems components

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

Only one image is unresolved at a time. Selecting or sending a frame does
not release it from the scheduler.

Each transmission includes a session UUID, frame ID and image hash.
The client releases a frame only after receiving a matching receipt.

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
- Model target: lightweight YOLO with COCO classes
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

## Deferred

- Real camera input and integration with Elias's model and tagging
- Top-level integration through `orchestrator.py`
- Adaptive compression in the client and corresponding cloudlet decoding,
  with wire-format changes in `shared/image_protocol.py`
- Embedding transmission
- Live learning on the cloudlet
- Bandwidth-aware scheduling and transmission
- A custom transport replacing Gabriel
- Multimodal input beyond images
- Native implementations for selected systems components

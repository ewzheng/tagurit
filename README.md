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

The client currently sends one independently decodable H.264 frame per image
using software encoding at CRF 32. It retains the original JPEG in the
scheduler and the encoded wire payload until a matching receipt arrives.
`src/tagurit/client/config.py` holds the codec and CRF settings. The cloudlet
decodes H.264 before acknowledging it. Jetson hardware encoding is still a
separate backend to implement and measure on the device.

## Lossless H.264 size experiment

Fetch the VisDrone validation images with
`uv run python scripts/fetch_data.py visdrone --split val`, then run
`uv run python scripts/benchmark_h264.py`.
The experiment decodes each JPEG, encodes its pixels as one independent
lossless H.264 frame with `libx264rgb`, decodes that frame in isolation, and
compares every reconstructed pixel value. Per-image byte counts and timings go
to `data/compression/visdrone-val.csv`. The run resumes from existing rows.
Use `--limit 100` for a quick sample or `--workers 1` to reduce CPU and memory
usage. Encoded frame bytes are discarded after measurement.

To compare lossy, independently decodable frames, run
`uv run python scripts/compare_lossy_codecs.py`. It samples two frames from
each VisDrone sequence and compares H.264 and H.265 at CRF 18, 23, and 28.
The CSV in `data/compression/lossy-sample.csv` records size, PSNR, and
grayscale SSIM against the pixels decoded from the source JPEG. Side-by-side
images and full-resolution crops are saved in `data/compression/previews/`.
Use `--limit 1 --crf 23` for a quick check. CRF values are codec-specific;
compare codecs at similar measured quality, not just the same CRF number.

## Configurable flow-control window

Run the normal demo with matching window and server credit limits:

```sh
# Terminal 1: server credits; queue defaults to at least this size
uv run python -m tagurit.cloudlet.image_receiver --tokens 2

# Terminal 2: maximum reserved images (encoding or waiting for a receipt)
uv run python -m tagurit.sim.run_client --window 2 --send-interval 0
```

Change both values to 4 to use four credits. The effective sending limit is
bounded by the client window and server credits; a server configured with one
token will still serialize submissions. `--queue-size` on the server must be
at least `--tokens`; this sizing assumes one producer.

Each frame retains its bytes until a matching receipt releases its own slot.
On reconnect all unresolved frames are replayed before new work. Receipts may
arrive out of order. Cached frames awaiting acknowledgment keep the scheduler
in reintegration mode even when the waiting bank is empty.

## One-token flow-control baseline

Timing collection is opt-in and reusable. The regular demo uses its configured
send interval; the finite benchmark defaults to **zero artificial send delay**.
The regular client and cloudlet now default to two outstanding frames with no
artificial send delay. The benchmark defaults to `--window 1` to preserve the
one-token baseline; pass `--window 2` or `--window 4` for larger fixed windows.
An adaptive controller is not implemented yet.

Start the cloudlet in terminal 1 (choose a new event filename for each run):

```sh
uv run python -m tagurit.cloudlet.image_receiver --tokens 1 --events data/flow-control/server-events.csv
```

In terminal 2, send 28 deterministically sampled VisDrone validation images at
10 arrivals/second. The source is preloaded before timing and assigns each frame
an explicit priority of 0.5; it does not run perception or priority scoring.

```sh
uv run python scripts/benchmark_flow_control.py \
  --count 28 --arrival-interval 0.1 --send-interval 0 \
  --output data/flow-control/one-token-run
```

The client stops after all frames are acknowledged, or fails after `--timeout`
seconds (default 120), preserving partial data. The server runs until Ctrl+C.
Use a fresh output directory for each run. `--images` changes the JPEG directory;
`--endpoint ws://HOST:9099` targets another machine. The server supports `--port`.
Both paths currently use software H.264 at the configured CRF (default 32).

Outputs under the gitignored `data/` directory:

- `config.json`: run settings, codec, CRF and application in-flight limit.
- `inputs.csv`: exact source paths, JPEG sizes, hashes and test priorities.
- `client-events.csv`: arrival, scheduler selection, encode start/end, submission
  and validated acknowledgment events, plus queue sizes at those events.
- `frames.csv`: one row per arrival, including unfinished frames and retry counts.
- `summary.json`: counts, delay median/p95/max, observed throughput and peak queue.
- The separate server CSV records acceptance or duplicate handling, correlated
  with client rows by session UUID and frame ID.

Metric definitions:

| Metric | Measurement |
|---|---|
| `queue_ms` | Client arrival to scheduler selection; excludes encoding |
| `encode_ms` | Completed encoding attempts, including JPEG decode and FFmpeg startup |
| `arrival_to_submit_ms` | Client arrival to first encoded-payload handoff to Gabriel |
| `send_to_receipt_ms` | Last handoff to validated receipt; includes Gabriel scheduling, network and server decoding |
| `arrival_to_receipt_ms` | Total time from client arrival until validated acknowledgment |
| `attempts` | Number of submissions, including retries |

Submission is the application handoff to Gabriel, not a packet-capture timestamp.
Send-to-receipt is not pure network RTT. All durations use the client's monotonic
clock. Server Unix timestamps provide correlation only: do not subtract clocks
across hosts without synchronization. Retry-free send-to-receipt percentiles
exclude retried frames because a response may belong to an earlier attempt.
Unfinished frames remain visible and are excluded from completed-frame delays.
CSV logging is synchronous and adds a small amount of overhead; keep it enabled
consistently across comparisons.

Throughput counts acknowledged application bytes including our image header,
excluding WebSocket/TCP/IP overhead and retransmissions. It measures this workload,
not available link capacity. A growing client queue means offered work exceeds
completion rate; it does not by itself identify a network bottleneck. The local
baseline includes software encode/decode costs and the configured arrival rate.

For the two-token benchmark, start the receiver with `--tokens 2`, and add
`--window 2` to the benchmark command with a fresh output directory.
`max_submitted_frames` in the summary reports the observed peak number of
submitted images awaiting a receipt; it can be smaller than the configured
window when encoding or input arrival is slower than server responses.

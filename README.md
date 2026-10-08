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

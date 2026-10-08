"""
Run a finite H.264 fixed-window experiment against a separately started cloudlet.

The sample is deterministic and loaded before timing. Every input has an explicit
constant test priority; this experiment does not evaluate perception or tagging.
The output directory must be new so prior measurements cannot be overwritten.
"""

import argparse
import asyncio
import csv
import hashlib
import json
import math
import time
from pathlib import Path

from tagurit.client.config import GABRIEL_ENDPOINT, IMAGE_CODEC, VIDEO_CRF
from tagurit.client.main import run_client
from tagurit.protocol import ImageFrame
from tagurit.shared.telemetry import EventLog
from tagurit.sim.flow_metrics import summarize_events


def main() -> None:
    """
    Record a finite baseline, then save per-frame and aggregate measurements.

    Return: void; incomplete runs exit with an error after saving their events
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("data/visdrone/VisDrone2019-MOT-val/sequences"),
    )
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--count", type=int, default=28)
    parser.add_argument("--arrival-interval", type=float, default=0.1)
    parser.add_argument("--send-interval", type=float, default=0)
    parser.add_argument("--endpoint", default=GABRIEL_ENDPOINT)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.window < 1:
        parser.error("window must be positive")
    if args.count < 1:
        parser.error("count must be positive")
    for name in ("arrival_interval", "send_interval", "timeout"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name == "timeout" and value == 0):
            parser.error(
                f"{name} must be finite and {'positive' if name == 'timeout' else 'nonnegative'}"
            )
    paths = sorted(args.images.rglob("*.jpg"))
    if len(paths) < args.count:
        parser.error(f"Need {args.count} JPEGs; found {len(paths)} under {args.images}")
    if args.output.exists():
        parser.error(f"Output directory already exists: {args.output}")
    indexes = [i * (len(paths) - 1) // max(1, args.count - 1) for i in range(args.count)]
    chosen = [paths[i] for i in indexes]
    images = [path.read_bytes() for path in chosen]
    args.output.mkdir(parents=True)
    with (args.output / "inputs.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("frame_id", "path", "jpeg_bytes", "sha256", "priority"))
        for frame_id, (path, payload) in enumerate(zip(chosen, images, strict=True), start=1):
            writer.writerow(
                (
                    frame_id,
                    str(path.resolve()),
                    len(payload),
                    hashlib.sha256(payload).hexdigest(),
                    0.5,
                )
            )
    config = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    config.update(
        codec=IMAGE_CODEC, crf=VIDEO_CRF, application_inflight_limit=args.window, test_priority=0.5
    )
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    log = EventLog(args.output / "client-events.csv")

    async def source():
        started = time.monotonic()
        for i, payload in enumerate(images):
            await asyncio.sleep(max(0, started + i * args.arrival_interval - time.monotonic()))
            yield ImageFrame(i + 1, time.time(), payload, 0.5)

    def report(event, detail, scheduler):
        print(
            f"{event:<8} | {detail} | waiting={scheduler.live_count() + scheduler.stored_count()}",
            flush=True,
        )

    async def run():
        async with asyncio.timeout(args.timeout):
            await run_client(
                source(),
                report,
                window_size=args.window,
                send_interval=args.send_interval,
                endpoint=args.endpoint,
                event_log=log,
            )

    try:
        asyncio.run(run())
    finally:
        log.close()
        rows, summary = summarize_events(args.output / "client-events.csv")
        if rows:
            with (args.output / "frames.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

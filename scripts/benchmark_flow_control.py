"""
Run a finite H.264 client-window experiment against a separately started cloudlet.

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
import sys
import time
from pathlib import Path

from tagurit.client.adaptive_window import (
    INITIAL_WINDOW,
    MIN_PROBE_GAIN,
    MIN_RECEIPTS,
    PROBE_COOLDOWN_SECONDS,
    TARGET_DELAY_MS,
    WINDOW_LEVELS,
)
from tagurit.client.config import GABRIEL_ENDPOINT, IMAGE_CODEC, VIDEO_CRF
from tagurit.client.main import run_client
from tagurit.protocol import ImageFrame
from tagurit.shared.telemetry import EventLog
from tagurit.sim.flow_metrics import summarize_events


def record_condition(log: EventLog, label: str, started: float) -> str:
    """
    Timestamp an operator-reported network profile in the client event log.

    The timestamp is when the operator presses Enter, so a delay between
    changing Network Link Conditioner and typing the label is possible.

    Parameters:
        - log (EventLog): Caller-owned event log
        - label (str): Human-readable active network profile
        - started (float): Monotonic start of the experiment

    Return: Console line containing the profile and elapsed time
    """
    label = label.strip()
    if not label:
        raise ValueError("Network condition label cannot be empty")
    now = time.monotonic()
    log.record("network_condition", "", at=now, network_condition=label)
    return f"CONDITION | +{now - started:6.1f}s | {label}"


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
    parser.add_argument("--window", type=int)
    parser.add_argument("--adaptive-window", action="store_true")
    parser.add_argument("--count", type=int, default=28)
    parser.add_argument("--arrival-interval", type=float, default=0.1)
    parser.add_argument("--send-interval", type=float, default=0)
    parser.add_argument("--endpoint", default=GABRIEL_ENDPOINT)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--interactive-markers", action="store_true")
    parser.add_argument("--initial-condition", default="unspecified")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.window is None:
        args.window = INITIAL_WINDOW if args.adaptive_window else 1
    if args.window < 1:
        parser.error("window must be positive")
    if args.adaptive_window and args.window not in WINDOW_LEVELS:
        parser.error("adaptive window must be one of 1, 2, 4, or 8")
    if args.count < 1:
        parser.error("count must be positive")
    if args.interactive_markers and not sys.stdin.isatty():
        parser.error("--interactive-markers requires an interactive terminal")
    if not args.initial_condition.strip():
        parser.error("initial condition cannot be empty")
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
        codec=IMAGE_CODEC,
        crf=VIDEO_CRF,
        application_inflight_limit=args.window,
        adaptive_window=args.adaptive_window,
        adaptive_policy=(
            {
                "window_levels": WINDOW_LEVELS,
                "target_delay_ms": TARGET_DELAY_MS,
                "min_receipts": MIN_RECEIPTS,
                "min_probe_gain": MIN_PROBE_GAIN,
                "probe_cooldown_seconds": PROBE_COOLDOWN_SECONDS,
                "measurement_seconds": 10.0,
            }
            if args.adaptive_window
            else None
        ),
        test_priority=0.5,
    )
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    log = EventLog(args.output / "client-events.csv")
    console = (
        (args.output / "live-console.log").open("x", encoding="utf-8")
        if args.interactive_markers
        else None
    )
    started = time.monotonic()

    def emit(line: str) -> None:
        print(line, flush=True)
        if console is not None:
            console.write(line + "\n")
            console.flush()

    async def source():
        started = time.monotonic()
        for i, payload in enumerate(images):
            await asyncio.sleep(max(0, started + i * args.arrival_interval - time.monotonic()))
            yield ImageFrame(i + 1, time.time(), payload, 0.5)

    def report(event, detail, scheduler):
        if args.interactive_markers and event not in {
            "START",
            "STATE",
            "LINK",
            "WINDOW",
            "METRICS",
            "INPUT",
            "DRAINED",
            "END",
        }:
            return
        elapsed = f"+{time.monotonic() - started:6.1f}s | " if args.interactive_markers else ""
        emit(
            f"{event:<8} | {elapsed}{detail} | "
            f"waiting={scheduler.live_count() + scheduler.stored_count()}"
        )

    async def run():
        loop = asyncio.get_running_loop()

        def read_condition() -> None:
            label = sys.stdin.readline()
            if not label:
                loop.remove_reader(sys.stdin.fileno())
            elif label.strip():
                emit(record_condition(log, label, started))

        if args.interactive_markers:
            emit(record_condition(log, args.initial_condition, started))
            emit("After each NLC switch, type its profile name and press Enter.")
            loop.add_reader(sys.stdin.fileno(), read_condition)
        try:
            await asyncio.wait_for(
                run_client(
                    source(),
                    report,
                    window_size=args.window,
                    adaptive_window=args.adaptive_window,
                    send_interval=args.send_interval,
                    endpoint=args.endpoint,
                    event_log=log,
                ),
                args.timeout,
            )
        finally:
            if args.interactive_markers:
                loop.remove_reader(sys.stdin.fileno())

    try:
        asyncio.run(run())
    finally:
        log.close()
        if console is not None:
            console.close()
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

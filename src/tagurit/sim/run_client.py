"""
Run the client with repeating sample images and console queue displays

Image contents and scores come from the local manifest
Communication uses the actual Gabriel connection
"""

import argparse
import asyncio
import math
import time
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from tagurit.client.adaptive_window import INITIAL_WINDOW, WINDOW_LEVELS
from tagurit.client.config import INFLIGHT_WINDOW, SEND_INTERVAL_SECONDS
from tagurit.client.frame_scheduler import FrameScheduler
from tagurit.client.main import run_client
from tagurit.protocol import ImageFrame
from tagurit.shared.telemetry import EventLog
from tagurit.sim.config import FRAME_INTERVAL_SECONDS, QUEUE_DISPLAY_LIMIT
from tagurit.sim.image_feeder import feed_images


# Repeat sample images with a new identity and timestamp for each arrival
async def repeat_frames() -> AsyncIterator[ImageFrame]:
    """
    Yield sample frames at the configured arrival interval

    Frame IDs keep increasing across passes through the manifest
    Waiting between arrivals yields control to the communication tasks
    An empty manifest RAISES ValueError

    Return:
        Asynchronous iterator of scored sample frames
    """
    next_frame_id = 1

    # Start another pass when every manifest image has been used
    while True:
        found_frame = False

        for frame in feed_images():
            found_frame = True

            # Give this simulated arrival its own ID and capture time
            arriving_frame = replace(frame, frame_id=next_frame_id, timestamp=time.time())
            next_frame_id += 1

            yield arriving_frame

            # Allow the client to communicate while waiting for another arrival
            await asyncio.sleep(FRAME_INTERVAL_SECONDS)

        if not found_frame:
            raise ValueError("The manifest must contain at least one image")


# Show queue sizes without printing an unlimited number of blocks
def queue_display(count: int) -> str:
    """
    Format a queue count with a capped row of blocks

    Parameters:
        - count (int): Number of waiting frames

    Return:
        Text containing the count and queue display
    """
    blocks = "■" * min(count, QUEUE_DISPLAY_LIMIT)
    overflow = "+" if count > QUEUE_DISPLAY_LIMIT else ""
    return f"{count:3d} [{blocks}{overflow}]"


# Start the sample image source and the real client
async def main() -> None:
    """
    Run the repeating image demo until cancelled

    Queue sizes and communication events are printed to the terminal
    The simulated connectivity schedule is not used

    Return:
        void
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int)
    parser.add_argument("--adaptive-window", action="store_true")
    parser.add_argument("--send-interval", type=float, default=SEND_INTERVAL_SECONDS)
    parser.add_argument("--metrics-window", type=float, default=10.0)
    parser.add_argument("--metrics-interval", type=float, default=2.0)
    parser.add_argument("--events", type=Path, help="Client per-frame CSV path")
    args = parser.parse_args()
    if args.window is None:
        args.window = INITIAL_WINDOW if args.adaptive_window else INFLIGHT_WINDOW
    if args.adaptive_window and args.window not in WINDOW_LEVELS:
        parser.error("adaptive window must be one of 1, 2, 4, or 8")
    for name in ("metrics_window", "metrics_interval"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name.replace('_', '-')} must be finite and positive")
    start_time = time.monotonic()
    event_path = args.events or Path("data/flow-control") / (
        f"client-events-{datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:8]}.csv"
    )
    event_log = EventLog(event_path)

    # Print events with elapsed time and current queue sizes
    def report(event: str, detail: str, scheduler: FrameScheduler) -> None:
        elapsed = time.monotonic() - start_time

        print(f"\n{elapsed:6.2f}s | {event:<8} | {scheduler.state.value:<13} | {detail}")
        print(
            f"         Live {queue_display(scheduler.live_count())}  "
            f"Bank {queue_display(scheduler.stored_count())}  "
            f"In flight {scheduler.inflight_count()}",
            flush=True,
        )

    print("Local image demo with real Gabriel communication", flush=True)
    print(
        f"Window mode: {'adaptive' if args.adaptive_window else 'fixed'} | initial={args.window}",
        flush=True,
    )
    print(f"Client events: {event_path}", flush=True)
    print("Press Ctrl+C to stop | Queued frames are not drained on exit", flush=True)

    try:
        await run_client(
            repeat_frames(),
            report,
            window_size=args.window,
            adaptive_window=args.adaptive_window,
            send_interval=args.send_interval,
            event_log=event_log,
            metrics_window_seconds=args.metrics_window,
            metrics_report_interval_seconds=args.metrics_interval,
        )

    finally:
        event_log.close()
        print("Queues and in-flight data are memory-only and are lost on exit", flush=True)


# Run the demo when this module is launched directly
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nClient stopped")

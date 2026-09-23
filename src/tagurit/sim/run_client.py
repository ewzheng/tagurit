"""
Run the client with repeating sample images and console queue displays

Image contents and scores come from the local manifest
Communication uses the actual Gabriel connection
"""

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import replace

from tagurit.client.frame_scheduler import FrameScheduler
from tagurit.client.main import run_client
from tagurit.protocol import ImageFrame
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
            arriving_frame = replace(
                frame,
                frame_id=next_frame_id,
                timestamp=time.time()
            )
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
    start_time = time.monotonic()

    # Print events with elapsed time and current queue sizes
    def report(event: str, detail: str, scheduler: FrameScheduler) -> None:
        elapsed = time.monotonic() - start_time

        print(
            f"\n{elapsed:6.2f}s | {event:<8} | "
            f"{scheduler.state.value:<13} | {detail}"
        )
        print(
            f"         Live {queue_display(scheduler.live_count())}  "
            f"Bank {queue_display(scheduler.stored_count())}  "
            f"In flight {scheduler.inflight_count()}",
            flush=True
        )

    print("Local image demo with real Gabriel communication", flush=True)
    print("Press Ctrl+C to stop | Queued frames are not drained on exit", flush=True)

    try:
        await run_client(repeat_frames(), report)

    finally:
        print(
            "Queues and in-flight data are memory-only and are lost on exit",
            flush=True
        )


# Run the demo when this module is launched directly
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nClient stopped")
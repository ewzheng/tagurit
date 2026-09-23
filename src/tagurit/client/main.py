"""
Run the client using a frame source supplied by the caller

Coordinate image arrivals, scheduling and Gabriel communication
This module does not depend on simulation or model code
"""

import asyncio
from collections.abc import AsyncIterable, Callable
from uuid import uuid4

from tagurit.client.config import LOOP_INTERVAL_SECONDS
from tagurit.client.frame_scheduler import FrameScheduler
from tagurit.client.gabriel_transport import GabrielTransport
from tagurit.client.scheduler_datatypes import SchedulerState
from tagurit.protocol import ImageFrame


# Run the client with a supplied frame source and event reporter
async def run_client(frame_source: AsyncIterable[ImageFrame], report: Callable[[str, str, FrameScheduler], None]) -> None:
    """
    Collect frames while Gabriel sends selected images and handles reconnection

    Frame IDs MUST be unique within this run
    Frames arriving while disconnected MUST have a priority score
    The source must yield control while waiting so communication can continue

    When the source ends, wait until all retained frames are acknowledged
    Cancellation stops the client immediately and may leave unfinished work

    Parameters:
        - frame_source (AsyncIterable[ImageFrame]): Stream of incoming frames
        - report (Callable): Callback receiving event, detail and scheduler

    Return:
        void
    """

    # Create the scheduler and counters for this client run
    scheduler = FrameScheduler()
    arrivals = 0
    session_id = str(uuid4())

    # Attach the scheduler to events reported by the transport
    def report_event(event: str, detail: str) -> None:
        report(event, detail, scheduler)

    # Use actual transport health without a simulated availability restriction
    transport = GabrielTransport(scheduler, session_id, report_event)
    transport.set_link_allowed(True)

    # Collect frames independently of connection attempts and transmission
    async def collect_frames() -> None:
        nonlocal arrivals

        # Route each supplied frame according to the current operating state
        async for frame in frame_source:
            scheduler.add_frame(frame)
            arrivals += 1

            lane = "bank" if scheduler.state == SchedulerState.DISCONNECTED else "live"
            report_event("ARRIVE", f"Frame {frame.frame_id} -> {lane}")

        # Keep transmission running after a finite source finishes
        report_event("INPUT", "Frame source ended; waiting for retained frames")

        while scheduler.pending_count():
            await asyncio.sleep(LOOP_INTERVAL_SECONDS)

        report_event("DRAINED", "All supplied frames acknowledged")

    report_event("START", f"Session {session_id} | Actual transport health controls connectivity")

    # Run collection and communication on the same event loop
    tasks = [
        asyncio.create_task(collect_frames()),
        asyncio.create_task(transport.run())
    ]

    try:

        # Finish after the source drains or surface an unexpected task failure
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        for task in done:
            task.result()

    finally:

        # Stop both tasks before reporting the remaining work
        transport.stop_accepting()

        for task in tasks:
            task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)

        report_event(
            "END",
            f"Arrived={arrivals} | "
            f"Acknowledged={transport.completed} | "
            f"Unfinished={scheduler.pending_count()}"
        )
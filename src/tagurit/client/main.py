"""
Run the client using a scored frame source supplied by the caller

Coordinate arrivals, queue maintenance and Gabriel communication
This module does not depend on simulation or model code
"""

import asyncio
from collections.abc import AsyncIterable, Callable
from uuid import uuid4

from tagurit.client.config import INFLIGHT_WINDOW, LOOP_INTERVAL_SECONDS
from tagurit.client.frame_scheduler import FrameScheduler
from tagurit.client.gabriel_transport import GabrielTransport
from tagurit.client.scheduler_datatypes import SchedulerState
from tagurit.protocol import ImageFrame
from tagurit.shared.telemetry import EventLog


# Run the client with a supplied frame source and event reporter
async def run_client(
    frame_source: AsyncIterable[ImageFrame],
    report: Callable[[str, str, FrameScheduler], None],
    *,
    window_size: int = INFLIGHT_WINDOW,
    send_interval: float | None = None,
    endpoint: str | None = None,
    event_log: EventLog | None = None,
) -> None:
    """
    Collect scored frames while Gabriel sends and reconnects as needed

    Every frame MUST have a score and an ID unique within this run
    The source MUST yield control while waiting so other tasks can continue
    When input ends the client waits for all retained frames to be acknowledged
    Cancellation stops the client and may leave unfinished work

    Parameters:
        - frame_source (AsyncIterable[ImageFrame]): Stream of scored frames
        - report (Callable): Callback receiving event, detail and scheduler
        - window_size (int): Maximum reserved frames
        - send_interval (float | None): Minimum seconds between submissions; None uses config
        - endpoint (str | None): Gabriel URL override; None uses config
        - event_log (EventLog | None): Optional caller-owned CSV recorder

    Return:
        void
    """

    # Create the scheduler and counters for this client run
    scheduler = FrameScheduler(window_size=window_size)
    arrivals = 0
    session_id = str(uuid4())

    # Attach the scheduler to events reported by the transport
    def report_event(event: str, detail: str) -> None:
        report(event, detail, scheduler)

    # Use actual transport health without a simulated availability restriction
    options = {} if endpoint is None else {"endpoint": endpoint}
    transport = GabrielTransport(
        scheduler,
        session_id,
        report_event,
        send_interval=send_interval,
        event_log=event_log,
        **options,
    )
    transport.set_link_allowed(True)

    # Collect frames independently of connection attempts and transmission
    async def collect_frames() -> None:
        nonlocal arrivals

        # Route every scored frame according to the current mode
        async for frame in frame_source:
            scheduler.add_frame(frame)
            arrivals += 1
            if event_log is not None:
                event_log.record(
                    "arrive",
                    session_id,
                    frame.frame_id,
                    live_count=scheduler.live_count(),
                    bank_count=scheduler.stored_count(),
                    inflight_count=scheduler.inflight_count(),
                    jpeg_bytes=len(frame.image_bytes),
                )
            lane = "bank" if scheduler.state == SchedulerState.DISCONNECTED else "live"
            report_event("ARRIVE", f"Frame {frame.frame_id} -> {lane}")

        # Keep transmission and queue maintenance active after input ends
        report_event("INPUT", "Frame source ended; waiting for retained frames")

        while scheduler.pending_count():
            await asyncio.sleep(LOOP_INTERVAL_SECONDS)

        report_event("DRAINED", "All supplied frames acknowledged")

    # Check the disconnected timer even while the image source is waiting
    async def maintain_queues() -> None:
        while True:
            frame = scheduler.migrate_live_frame()

            if frame is not None:
                report_event(
                    "MIGRATE",
                    f"Frame {frame.frame_id} live -> bank | priority={frame.priority:.2f}",
                )

            await asyncio.sleep(LOOP_INTERVAL_SECONDS)

    if event_log is not None:
        event_log.record("start", session_id)
    report_event("START", f"Session {session_id} | Actual transport health controls connectivity")

    # Serialize all three tasks on the same event loop
    tasks = [
        asyncio.create_task(collect_frames()),
        asyncio.create_task(transport.run()),
        asyncio.create_task(maintain_queues()),
    ]

    try:
        # Finish when the source drains or surface an unexpected task failure
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        for task in done:
            task.result()

    finally:
        # Stop all tasks before reporting remaining work
        transport.stop_accepting()

        for task in tasks:
            task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)

        if event_log is not None:
            event_log.record(
                "end",
                session_id,
                live_count=scheduler.live_count(),
                bank_count=scheduler.stored_count(),
                inflight_count=scheduler.inflight_count(),
            )
        report_event(
            "END",
            f"Arrived={arrivals} | "
            f"Acknowledged={transport.completed} | "
            f"Unfinished={scheduler.pending_count()}",
        )

"""
Run the client using a scored frame source supplied by the caller

Coordinate arrivals, queue maintenance and Gabriel communication
This module does not depend on simulation or model code
"""

import asyncio
import math
import time
from collections.abc import AsyncIterable, Callable
from uuid import uuid4

from tagurit.client.adaptive_window import (
    INITIAL_WINDOW,
    AdaptiveWindowController,
    WindowDecision,
)
from tagurit.client.config import INFLIGHT_WINDOW, LOOP_INTERVAL_SECONDS
from tagurit.client.frame_scheduler import FrameScheduler
from tagurit.client.gabriel_transport import GabrielTransport
from tagurit.client.rolling_metrics import RollingSnapshot
from tagurit.client.scheduler_datatypes import SchedulerState
from tagurit.protocol import ImageFrame
from tagurit.shared.telemetry import EventLog


# Run the client with a supplied frame source and event reporter
async def run_client(
    frame_source: AsyncIterable[ImageFrame],
    report: Callable[[str, str, FrameScheduler], None],
    *,
    window_size: int | None = None,
    adaptive_window: bool = False,
    send_interval: float | None = None,
    endpoint: str | None = None,
    event_log: EventLog | None = None,
    metrics_window_seconds: float = 10.0,
    metrics_report_interval_seconds: float = 2.0,
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
        - window_size (int | None): Initial limit; None uses four in adaptive mode
          and the configured fixed limit otherwise
        - adaptive_window (bool): Adjust the client limit from recent receipts
        - send_interval (float | None): Minimum seconds between submissions; None uses config
        - endpoint (str | None): Gabriel URL override; None uses config
        - event_log (EventLog | None): Optional caller-owned CSV recorder
        - metrics_window_seconds (float): Recent receipt measurement horizon
        - metrics_report_interval_seconds (float): Time between console measurements

    Return:
        void
    """

    if not math.isfinite(metrics_report_interval_seconds) or metrics_report_interval_seconds <= 0:
        raise ValueError("Measurement report interval must be finite and positive")

    # Create the scheduler and counters for this client run
    if window_size is None:
        window_size = INITIAL_WINDOW if adaptive_window else INFLIGHT_WINDOW
    scheduler = FrameScheduler(window_size=window_size)
    controller = (
        AdaptiveWindowController(time.monotonic(), metrics_window_seconds, window_size)
        if adaptive_window
        else None
    )
    arrivals = 0
    session_id = str(uuid4())

    def apply_window(decision: WindowDecision) -> None:
        previous = scheduler.window_size
        scheduler.set_window_size(decision.window_size)
        if event_log is not None:
            event_log.record(
                "window_change",
                session_id,
                live_count=scheduler.live_count(),
                bank_count=scheduler.stored_count(),
                inflight_count=scheduler.inflight_count(),
                window_size=decision.window_size,
                window_reason=decision.reason,
            )
        report("WINDOW", f"{previous} -> {decision.window_size} | {decision.reason}", scheduler)

    # Attach the scheduler to events reported by the transport
    def report_event(event: str, detail: str) -> None:
        report(event, detail, scheduler)
        if (
            controller is not None
            and event == "LINK"
            and detail.startswith("Connection unavailable:")
        ):
            decision = controller.connection_failed(time.monotonic())
            if decision is not None:
                apply_window(decision)

    # Use actual transport health without a simulated availability restriction
    options = {} if endpoint is None else {"endpoint": endpoint}
    transport = GabrielTransport(
        scheduler,
        session_id,
        report_event,
        send_interval=send_interval,
        event_log=event_log,
        metrics_window_seconds=metrics_window_seconds,
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

    def report_metrics() -> RollingSnapshot:
        snapshot = transport.recent_metrics()
        delay = snapshot.mean_submit_to_receipt_ms
        delay_text = "n/a" if delay is None else f"{delay:.0f}ms"
        window_text = f" | window={scheduler.window_size}" if controller is not None else ""
        report_event(
            "METRICS",
            f"Last {snapshot.elapsed_s:.1f}s | "
            f"mean submit-to-receipt={delay_text} | "
            f"acknowledged={snapshot.acknowledged_fps:.2f} frames/s "
            f"({snapshot.acknowledged} receipts){window_text}",
        )
        return snapshot

    async def report_recent_metrics() -> None:
        while True:
            await asyncio.sleep(metrics_report_interval_seconds)
            snapshot = report_metrics()
            if controller is not None:
                decision = controller.evaluate(
                    snapshot,
                    time.monotonic(),
                    waiting_frames=scheduler.live_count() + scheduler.stored_count(),
                    connected=scheduler.state != SchedulerState.DISCONNECTED,
                )
                if decision is not None:
                    apply_window(decision)

    if event_log is not None:
        event_log.record("start", session_id)
    report_event("START", f"Session {session_id} | Actual transport health controls connectivity")
    if controller is not None:
        if event_log is not None:
            event_log.record(
                "window_change",
                session_id,
                window_size=scheduler.window_size,
                window_reason="adaptive start",
            )
        report_event("WINDOW", f"Adaptive start | window={scheduler.window_size}")

    # Serialize the source, transport, queue timer and measurement reporter.
    tasks = [
        asyncio.create_task(collect_frames()),
        asyncio.create_task(transport.run()),
        asyncio.create_task(maintain_queues()),
        asyncio.create_task(report_recent_metrics()),
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

        report_metrics()

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

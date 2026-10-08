"""
Exercise fixed window capacity, receipt isolation and recovery without sockets.
"""

import asyncio
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from tagurit.client.frame_scheduler import FrameScheduler
from tagurit.client.gabriel_transport import GabrielTransport, ReceiptValidationError
from tagurit.client.scheduler_datatypes import SchedulerState
from tagurit.cloudlet.image_receiver import ImageReceiver
from tagurit.protocol import ImageFrame
from tagurit.shared.image_protocol import decode_image

SESSION = "00000000-0000-0000-0000-000000000001"


def make_transport(window=2):
    success, image = cv2.imencode(".jpg", np.zeros((16, 16, 3), dtype=np.uint8))
    assert success
    scheduler = FrameScheduler(window_size=window)
    transport = GabrielTransport(scheduler, SESSION, lambda *_: None, codec="jpeg", send_interval=0)
    transport._active_attempt = 1
    transport._ready = True
    transport._update_scheduler_connection()
    for frame_id in range(1, 5):
        scheduler.add_frame(ImageFrame(frame_id, 0, image.tobytes(), 0.5))
    return scheduler, transport, ImageReceiver()


def acknowledge(transport, receiver, payload, attempt=1):
    receipt = receiver.handle(payload, None)
    transport._receive_result(SimpleNamespace(string_result=receipt.payload), attempt)


def test_two_outstanding_and_out_of_order_receipt():
    """A full window blocks new reservations; ACK 2 frees only frame 2."""

    async def run():
        scheduler, transport, receiver = make_transport()
        first = await transport._produce(1)
        second = await transport._produce(1)
        assert scheduler.inflight_count() == 2
        blocked = asyncio.create_task(transport._produce(1))
        await asyncio.sleep(0)
        assert not blocked.done()
        assert scheduler.live_count() == 2
        acknowledge(transport, receiver, second)
        third = await asyncio.wait_for(blocked, 1)
        assert decode_image(third.byte_payload).frame_id == 3
        assert set(transport._pending) == {1, 3}
        acknowledge(transport, receiver, first)
        acknowledge(transport, receiver, third)
        assert transport.completed == 3
        assert scheduler.inflight_count() == 0

    asyncio.run(run())


def test_reconnect_replays_all_unresolved_before_new_work():
    """Lost receipts replay exact bytes, ignore stale callbacks and deduplicate."""

    async def run():
        scheduler, transport, receiver = make_transport()
        first = await transport._produce(1)
        second = await transport._produce(1)
        receiver.handle(first, None)
        receiver.handle(second, None)
        transport._active_attempt = 2
        acknowledge(transport, receiver, first, attempt=1)
        assert transport.completed == 0
        retry_first = await transport._produce(2)
        assert retry_first.byte_payload == first.byte_payload
        acknowledge(transport, receiver, retry_first, attempt=2)
        # Even with a free slot, frame 2 must be replayed before frame 3.
        retry_second = await transport._produce(2)
        assert retry_second.byte_payload == second.byte_payload
        acknowledge(transport, receiver, retry_second, attempt=2)
        assert receiver.accepted_count() == 2
        third = await transport._produce(2)
        assert decode_image(third.byte_payload).frame_id == 3
        assert scheduler.inflight_count() == 1

    asyncio.run(run())


def test_bad_or_duplicate_receipt_never_releases_another_slot():
    """Invalid receipts stop production; a duplicate cannot release frame 2."""

    async def run():
        scheduler, transport, receiver = make_transport()
        first = await transport._produce(1)
        await transport._produce(1)
        acknowledge(transport, receiver, first)
        acknowledge(transport, receiver, first)
        assert transport.completed == 1
        assert set(transport._pending) == {2}
        assert scheduler.inflight_count() == 1
        with pytest.raises(ReceiptValidationError):
            await transport._produce(1)

    asyncio.run(run())


def test_bank_inflight_keeps_reintegration_without_empty_heap_pop():
    """All bank items may be outstanding while new live frames are selected."""
    scheduler = FrameScheduler(window_size=4, live_weight=1, stored_weight=2)
    scheduler.set_connected(False)
    scheduler.add_frame(ImageFrame(1, 0, b"x", 0.5))
    scheduler.set_connected(True)
    assert scheduler.reserve_next_frame().frame_id == 1
    for frame_id in (2, 3):
        scheduler.add_frame(ImageFrame(frame_id, 0, b"x", 0.5))
        assert scheduler.reserve_next_frame().frame_id == frame_id
    assert scheduler.state == SchedulerState.REINTEGRATING
    scheduler.acknowledge_frame(2)
    assert scheduler.state == SchedulerState.REINTEGRATING
    scheduler.acknowledge_frame(1)
    assert scheduler.state == SchedulerState.CONNECTED
    assert scheduler.inflight_count() == 1


@pytest.mark.parametrize("window", [0, -1, True, 1.5])
def test_invalid_windows(window):
    with pytest.raises(ValueError, match="Window size"):
        FrameScheduler(window_size=window)


def test_timeout_checks_every_pending_frame(monkeypatch):
    """A newer submission must not reset an older frame's receipt deadline."""

    async def run():
        _, transport, _ = make_transport()
        await transport._produce(1)
        await transport._produce(1)
        transport._pending[1].submitted_at = 0
        transport._pending[2].submitted_at = 15
        monkeypatch.setattr(
            "tagurit.client.gabriel_transport.time", SimpleNamespace(monotonic=lambda: 15)
        )
        task = asyncio.create_task(asyncio.Event().wait())
        try:
            with pytest.raises(TimeoutError, match="frame 1"):
                await transport._watch_connection(task, 0)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


def test_connection_loop_resets_tokens_and_retries_retained_frames(monkeypatch):
    """The real reconnect loop retains both frames across a failed connection."""

    async def run():
        _, transport, receiver = make_transport()
        monkeypatch.setattr("tagurit.client.gabriel_transport.RECONNECT_INTERVAL_SECONDS", 0)
        completed = asyncio.Event()
        payloads = []
        attempts = []

        def create_client(attempt):
            attempts.append(attempt)

            async def launch_async():
                first = await transport._produce(attempt)
                second = await transport._produce(attempt)
                if attempt == 1:
                    payloads.extend([first.byte_payload, second.byte_payload])
                    receiver.handle(first, None)
                    receiver.handle(second, None)
                    raise ConnectionError("Injected disconnect before receipts")
                assert [first.byte_payload, second.byte_payload] == payloads
                acknowledge(transport, receiver, second, attempt)
                acknowledge(transport, receiver, first, attempt)
                completed.set()
                await asyncio.Event().wait()

            return SimpleNamespace(launch_async=launch_async)

        monkeypatch.setattr(transport, "_create_client", create_client)
        task = asyncio.create_task(transport.run())
        try:
            await asyncio.wait_for(completed.wait(), 2)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        assert attempts == [1, 2]
        assert transport.completed == 2
        assert receiver.accepted_count() == 2
        assert not transport._pending

    asyncio.run(run())

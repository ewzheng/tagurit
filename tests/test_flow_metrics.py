"""
Check timing boundaries, retries and incomplete observations without a network.
"""

import asyncio
from types import SimpleNamespace

import pytest

from tagurit.client.frame_scheduler import FrameScheduler
from tagurit.client.gabriel_transport import GabrielTransport
from tagurit.protocol import ImageFrame
from tagurit.shared.image_protocol import ImageReceipt, decode_image, encode_receipt
from tagurit.shared.telemetry import EventLog
from tagurit.sim.flow_metrics import summarize_events

SESSION = "00000000-0000-0000-0000-000000000001"


@pytest.mark.parametrize("retry", [False, True])
def test_timing_excludes_encoding_and_preserves_retry(tmp_path, monkeypatch, retry):
    """Encoding is timed separately; a retry reuses bytes and its arrival time."""
    now = [10.0]

    def clock():
        return now[0]

    monkeypatch.setattr("tagurit.client.gabriel_transport.time", SimpleNamespace(monotonic=clock))
    path = tmp_path / "events.csv"
    log = EventLog(path, clock=clock)
    scheduler = FrameScheduler(clock=clock)
    transport = GabrielTransport(
        scheduler,
        SESSION,
        lambda *_: None,
        send_interval=0,
        event_log=log,
    )
    transport._active_attempt = 1
    transport._ready = True
    transport._update_scheduler_connection()
    scheduler.add_frame(ImageFrame(1, 0, b"source", 0.5))
    log.record("arrive", SESSION, 1, jpeg_bytes=6, live_count=1, bank_count=0)
    now[0] = 20
    calls = []

    def encode(*args):
        calls.append(args)
        now[0] = 22
        return b"encoded", 16, 16

    monkeypatch.setattr("tagurit.client.gabriel_transport.encode_video", encode)
    first = asyncio.run(transport._produce(1))
    assert transport._pending[1].submitted_at == 22
    if retry:
        now[0] = 30
        transport._pending[1].submitted_at = None
        transport._active_attempt = 2
        second = asyncio.run(transport._produce(2))
        assert second.byte_payload == first.byte_payload
    assert len(calls) == 1
    now[0] = 33 if retry else 25
    message = decode_image(first.byte_payload)
    receipt = ImageReceipt("received", SESSION, 1, message.sha256, 7, 16, 16)
    transport._receive_result(
        SimpleNamespace(string_result=encode_receipt(receipt)), 2 if retry else 1
    )
    log.close()
    rows, summary = summarize_events(path)
    assert rows[0]["queue_ms"] == 10000
    assert rows[0]["encode_ms"] == 2000
    assert rows[0]["arrival_to_submit_ms"] == 12000
    assert rows[0]["send_to_receipt_ms"] == 3000
    assert rows[0]["first_submit_to_receipt_ms"] == (11000 if retry else 3000)
    assert rows[0]["attempts"] == (2 if retry else 1)
    assert summary["send_to_receipt_ms"]["count"] == (0 if retry else 1)
    assert scheduler.pending_count() == 0


def test_incomplete_run_is_not_reported_as_success(tmp_path):
    """Pending arrivals survive in the report even if the client is cancelled."""
    path = tmp_path / "events.csv"
    log = EventLog(path, clock=lambda: 100)
    log.record("arrive", SESSION, 1, jpeg_bytes=100, live_count=1)
    log.close()
    rows, summary = summarize_events(path)
    assert rows[0]["status"] == "unfinished"
    assert rows[0]["queue_ms"] is None
    assert summary["acknowledged"] == 0
    assert summary["unfinished"] == 1
    assert summary["max_waiting_frames"] == 1
    with pytest.raises(FileExistsError):
        EventLog(path)


@pytest.mark.parametrize("interval", [-1, float("nan"), float("inf")])
def test_invalid_pacing_rejected(interval):
    """Experiment settings cannot introduce negative or unbounded timers."""
    with pytest.raises(ValueError, match="Send interval"):
        GabrielTransport(FrameScheduler(), SESSION, lambda *_: None, send_interval=interval)

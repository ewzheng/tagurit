"""
Check that adaptive decisions reach the scheduler and client event log.
"""

import asyncio
import csv
import time

from tagurit.client.main import run_client
from tagurit.client.rolling_metrics import RollingReceiptMetrics
from tagurit.protocol import ImageFrame
from tagurit.shared.telemetry import EventLog


def test_adaptive_run_probes_and_records_window_changes(monkeypatch, tmp_path):
    """The client applies controller output while retaining every test frame."""

    class FakeTransport:
        """A serial receipt source that replaces Gabriel for this offline check."""

        def __init__(self, scheduler, session_id, report, **options):
            self.scheduler = scheduler
            self.metrics = RollingReceiptMetrics(
                options["metrics_window_seconds"], time.monotonic()
            )
            self.completed = 0

        def set_link_allowed(self, available):
            self.scheduler.set_connected(available)

        async def run(self):
            while True:
                frame = self.scheduler.reserve_next_frame()
                if frame is None:
                    await asyncio.sleep(0.001)
                    continue
                submitted_at = time.monotonic()
                await asyncio.sleep(0.005)
                received_at = time.monotonic()
                self.metrics.record(submitted_at, received_at)
                self.scheduler.acknowledge_frame(frame.frame_id)
                self.completed += 1

        def recent_metrics(self):
            return self.metrics.snapshot(time.monotonic())

        def stop_accepting(self):
            pass

    monkeypatch.setattr("tagurit.client.main.GabrielTransport", FakeTransport)
    path = tmp_path / "client-events.csv"
    log = EventLog(path)
    reported = []

    async def frames():
        for frame_id in range(1, 101):
            yield ImageFrame(frame_id, 0, b"image", 0.5)

    try:
        asyncio.run(
            run_client(
                frames(),
                lambda event, detail, scheduler: reported.append(
                    (event, detail, scheduler.window_size)
                ),
                adaptive_window=True,
                event_log=log,
                metrics_window_seconds=0.05,
                metrics_report_interval_seconds=0.01,
            )
        )
    finally:
        log.close()

    with path.open(newline="") as stream:
        changes = [row for row in csv.DictReader(stream) if row["event"] == "window_change"]
    assert changes[0]["window_size"] == "4"
    assert any(row["window_size"] == "8" for row in changes)
    assert any(event == "WINDOW" and window == 8 for event, _, window in reported)
    assert reported[-1][0] == "END"
    assert "Acknowledged=100" in reported[-1][1]

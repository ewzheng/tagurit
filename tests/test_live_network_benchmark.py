"""
Check interactive network-profile markers without starting a cloudlet.
"""

import asyncio
import csv
import os
import sys

from scripts import benchmark_flow_control


def test_typed_profile_is_recorded_during_run(tmp_path, monkeypatch):
    """The terminal callback records a marker while the event loop is active."""
    images = tmp_path / "images"
    images.mkdir()
    (images / "frame.jpg").write_bytes(b"sample")
    output = tmp_path / "result"
    master_fd, slave_fd = os.openpty()

    async def fake_run_client(_source, _report, **_kwargs):
        os.write(master_fd, b"strong\n")
        marker = output / "client-events.csv"
        async with asyncio.timeout(1):
            while "strong" not in marker.read_text():
                await asyncio.sleep(0.01)

    with os.fdopen(slave_fd, "r", encoding="utf-8") as terminal:
        try:
            monkeypatch.setattr(sys, "stdin", terminal)
            monkeypatch.setattr(benchmark_flow_control, "run_client", fake_run_client)
            monkeypatch.setattr(
                sys,
                "argv",
                [
                    "benchmark_flow_control.py",
                    "--images",
                    str(images),
                    "--count",
                    "1",
                    "--interactive-markers",
                    "--initial-condition",
                    "off",
                    "--output",
                    str(output),
                ],
            )
            benchmark_flow_control.main()
        finally:
            os.close(master_fd)

    with (output / "client-events.csv").open(newline="") as stream:
        events = list(csv.DictReader(stream))
    assert [event["network_condition"] for event in events] == ["off", "strong"]
    assert "CONDITION |" in (output / "live-console.log").read_text()

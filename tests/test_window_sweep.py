"""
Check that the window sweep reports actual acknowledged-frame measurements.
"""

import csv
import json

from scripts.benchmark_window_sweep import read_result, save_table


def test_sweep_row_uses_acknowledged_frames_and_preserves_incomplete_status(tmp_path):
    """Unfinished frames stay visible and never enter the receipt-delay mean."""
    summary = {
        "arrived": 3,
        "acknowledged": 2,
        "unfinished": 1,
        "retried_frames": 0,
        "acknowledged_frames_per_s": 1.25,
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    with (tmp_path / "frames.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("status", "first_submit_to_receipt_ms"))
        writer.writeheader()
        writer.writerows(
            (
                {"status": "acknowledged", "first_submit_to_receipt_ms": 100},
                {"status": "acknowledged", "first_submit_to_receipt_ms": 300},
                {"status": "unfinished", "first_submit_to_receipt_ms": ""},
            )
        )

    row = read_result(tmp_path, "normal", 1, 2)
    assert row["mean_submit_to_receipt_ms"] == 200
    assert row["acknowledged_frames_per_s"] == 1.25
    assert row["frames_acknowledged"] == 2
    assert row["status"] == "incomplete"

    output = tmp_path / "window-results.csv"
    save_table(output, [row])
    with output.open(newline="") as stream:
        saved = list(csv.DictReader(stream))
    assert saved[0]["mean_submit_to_receipt_ms"] == "200.0"
    assert saved[0]["status"] == "incomplete"


def test_sweep_counts_retry_delay_from_first_submission(tmp_path):
    """A successful retry includes the time spent waiting before reconnecting."""
    summary = {
        "arrived": 2,
        "acknowledged": 2,
        "unfinished": 0,
        "retried_frames": 1,
        "acknowledged_frames_per_s": 1.0,
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    with (tmp_path / "frames.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("status", "send_to_receipt_ms", "first_submit_to_receipt_ms"),
        )
        writer.writeheader()
        writer.writerows(
            (
                {
                    "status": "acknowledged",
                    "send_to_receipt_ms": 100,
                    "first_submit_to_receipt_ms": 100,
                },
                {
                    "status": "acknowledged",
                    "send_to_receipt_ms": 200,
                    "first_submit_to_receipt_ms": 1200,
                },
            )
        )

    row = read_result(tmp_path, "limited", 1, 16)
    assert row["mean_submit_to_receipt_ms"] == 650
    assert row["status"] == "complete; retries"

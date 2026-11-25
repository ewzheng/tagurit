"""
Verify the client receipt window with a controlled monotonic clock.
"""

import pytest

from tagurit.client.rolling_metrics import RollingReceiptMetrics


def test_recent_delay_and_acknowledged_fps_expire_together():
    """Old acknowledgments leave both measurements after the time window."""
    metrics = RollingReceiptMetrics(10.0, started_at=0.0)
    metrics.record(1.0, 2.0)
    metrics.record(3.0, 5.0)

    early = metrics.snapshot(5.0)
    assert early.acknowledged == 2
    assert early.mean_submit_to_receipt_ms == 1500
    assert early.acknowledged_fps == pytest.approx(0.4)

    later = metrics.snapshot(13.0)
    assert later.acknowledged == 1
    assert later.mean_submit_to_receipt_ms == 2000
    assert later.acknowledged_fps == pytest.approx(0.1)

    empty = metrics.snapshot(16.0)
    assert empty.acknowledged == 0
    assert empty.mean_submit_to_receipt_ms is None
    assert empty.acknowledged_fps == 0


def test_measurement_window_rejects_invalid_duration():
    """A zero or unbounded measurement horizon cannot produce valid rates."""
    for duration in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="Measurement window"):
            RollingReceiptMetrics(duration, started_at=0.0)

"""
Track recently acknowledged frames for client-side flow-control measurements.

Only validated receipts enter this time window. The clock is local and monotonic.
"""

import math
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class RollingSnapshot:
    """
    Summarize receipts within the recent measurement interval.

    Parameters:
        - elapsed_s (float): Observed interval, capped at the configured window
        - acknowledged (int): Validated receipts in that interval
        - mean_submit_to_receipt_ms (float | None): Mean delay, or None without receipts
        - acknowledged_fps (float): Validated receipts per observed second
    """

    elapsed_s: float
    acknowledged: int
    mean_submit_to_receipt_ms: float | None
    acknowledged_fps: float


class RollingReceiptMetrics:
    """
    Keep receipt times and submit-to-receipt delays for a sliding time window.

    MUTATES its deque when a receipt arrives or a snapshot expires old receipts.
    Calls MUST be serialized on the client's event loop.

    Parameters:
        - window_seconds (float): Duration of the recent measurement interval
        - started_at (float): Local monotonic time when measurement began
    """

    def __init__(self, window_seconds: float, started_at: float) -> None:
        if not math.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError("Measurement window must be finite and positive")
        self._window_seconds = window_seconds
        self._started_at = started_at
        self._receipts: deque[tuple[float, float]] = deque()

    def record(self, submitted_at: float, received_at: float) -> None:
        """
        Add one validated receipt, measuring from its first submission attempt.

        Parameters:
            - submitted_at (float): Local monotonic first submission time
            - received_at (float): Local monotonic receipt time

        Return: void
        """
        if received_at < submitted_at:
            raise ValueError("Receipt time precedes submission")
        self._receipts.append((received_at, received_at - submitted_at))
        self._expire(received_at)

    def snapshot(self, at: float) -> RollingSnapshot:
        """
        Calculate recent delay and acknowledged throughput at a local time.

        The denominator grows from startup to the full window duration. With
        no receipts, delay is unavailable and throughput is zero.

        Parameters:
            - at (float): Local monotonic time for the snapshot

        Return: RollingSnapshot for the recent interval
        """
        if at < self._started_at:
            raise ValueError("Snapshot time precedes measurement start")
        self._expire(at)
        elapsed_s = min(self._window_seconds, at - self._started_at)
        acknowledged = len(self._receipts)
        mean_delay = (
            1000 * sum(delay for _, delay in self._receipts) / acknowledged
            if acknowledged
            else None
        )
        return RollingSnapshot(
            elapsed_s,
            acknowledged,
            mean_delay,
            acknowledged / elapsed_s if elapsed_s > 0 else 0.0,
        )

    def _expire(self, at: float) -> None:
        cutoff = at - self._window_seconds
        while self._receipts and self._receipts[0][0] < cutoff:
            self._receipts.popleft()

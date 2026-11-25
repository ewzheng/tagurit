"""
Choose a client in-flight limit from recent receipt delay and throughput.

This first controller probes one window level at a time. It uses application
receipts as feedback; it does not claim to measure physical link bandwidth.
"""

import math
from dataclasses import dataclass

from tagurit.client.rolling_metrics import RollingSnapshot

WINDOW_LEVELS = (1, 2, 4, 8)
INITIAL_WINDOW = 4
TARGET_DELAY_MS = 1000.0
MIN_RECEIPTS = 5
MIN_PROBE_GAIN = 0.03
PROBE_COOLDOWN_SECONDS = 30.0


@dataclass(frozen=True)
class WindowDecision:
    """
    Report a new client limit and the observation that caused it.

    Parameters:
        - window_size (int): New maximum number of reserved frames
        - reason (str): Human-readable reason for the change
    """

    window_size: int
    reason: str


class AdaptiveWindowController:
    """
    Probe for throughput gains while limiting receipt delay.

    Calls MUST be serialized with scheduler and transport callbacks on one
    event loop. The caller applies decisions to the scheduler; this class
    never owns or discards frames. A full measurement interval follows every
    change before another decision, and failed probes wait through a cooldown.

    Parameters:
        - started_at (float): Local monotonic time when control began
        - measurement_seconds (float): Duration of the rolling receipt sample
        - initial_window (int): Initial member of WINDOW_LEVELS
    """

    def __init__(
        self,
        started_at: float,
        measurement_seconds: float,
        initial_window: int = INITIAL_WINDOW,
    ) -> None:
        if not math.isfinite(started_at):
            raise ValueError("Controller start time must be finite")
        if not math.isfinite(measurement_seconds) or measurement_seconds <= 0:
            raise ValueError("Controller measurement interval must be finite and positive")
        if initial_window not in WINDOW_LEVELS or type(initial_window) is not int:
            raise ValueError("Adaptive window must be one of 1, 2, 4, or 8")
        self._window_size = initial_window
        self._measurement_seconds = measurement_seconds
        self._next_evaluation_at = started_at + measurement_seconds
        self._next_probe_at = self._next_evaluation_at
        self._probe_from: int | None = None
        self._probe_baseline_fps = 0.0

    @property
    def window_size(self) -> int:
        """
        Return the current target number of in-flight frames.

        Return: Positive client window size
        """
        return self._window_size

    def connection_failed(self, at: float) -> WindowDecision | None:
        """
        Reduce the limit after a connection failure or receipt timeout.

        Parameters:
            - at (float): Local monotonic failure time

        Return: A smaller limit, or None when already at the minimum
        """
        self._probe_from = None
        self._next_probe_at = at + PROBE_COOLDOWN_SECONDS
        self._next_evaluation_at = at + self._measurement_seconds
        index = WINDOW_LEVELS.index(self._window_size)
        if index == 0:
            return None
        return self._change(WINDOW_LEVELS[index - 1], "connection failure or receipt timeout")

    def evaluate(
        self,
        snapshot: RollingSnapshot,
        at: float,
        *,
        waiting_frames: int,
        connected: bool,
    ) -> WindowDecision | None:
        """
        Adjust after enough validated receipts cover one measurement interval.

        Increase only when queued work can use another slot. A probe is kept
        only if receipt throughput improves by at least three percent and mean
        delay remains below one second. Longer delays reduce the current limit.

        Parameters:
            - snapshot (RollingSnapshot): Recent validated receipt measurements
            - at (float): Local monotonic evaluation time
            - waiting_frames (int): Frames waiting outside the in-flight map
            - connected (bool): Whether Gabriel is currently available

        Return: A new limit when changed, otherwise None
        """
        if not connected or at < self._next_evaluation_at:
            return None
        if snapshot.elapsed_s < self._measurement_seconds:
            return None
        if snapshot.acknowledged < MIN_RECEIPTS or snapshot.mean_submit_to_receipt_ms is None:
            return None

        delay = snapshot.mean_submit_to_receipt_ms
        fps = snapshot.acknowledged_fps
        self._next_evaluation_at = at + self._measurement_seconds

        if self._probe_from is not None:
            previous = self._probe_from
            self._probe_from = None
            if delay <= TARGET_DELAY_MS and fps >= self._probe_baseline_fps * (1 + MIN_PROBE_GAIN):
                self._next_probe_at = at + PROBE_COOLDOWN_SECONDS
                return None
            self._next_probe_at = at + PROBE_COOLDOWN_SECONDS
            if delay > TARGET_DELAY_MS:
                reason = f"probe delay {delay:.0f} ms exceeds {TARGET_DELAY_MS:.0f} ms"
            else:
                reason = f"probe throughput {fps:.2f} frames/s did not improve enough"
            return self._change(previous, reason)

        if delay > TARGET_DELAY_MS:
            index = WINDOW_LEVELS.index(self._window_size)
            self._next_probe_at = at + PROBE_COOLDOWN_SECONDS
            if index > 0:
                return self._change(
                    WINDOW_LEVELS[index - 1],
                    f"mean receipt delay {delay:.0f} ms exceeds {TARGET_DELAY_MS:.0f} ms",
                )
            return None

        index = WINDOW_LEVELS.index(self._window_size)
        if waiting_frames > 0 and index < len(WINDOW_LEVELS) - 1 and at >= self._next_probe_at:
            self._probe_from = self._window_size
            self._probe_baseline_fps = fps
            return self._change(WINDOW_LEVELS[index + 1], f"probe above {fps:.2f} frames/s")
        return None

    def _change(self, window_size: int, reason: str) -> WindowDecision:
        self._window_size = window_size
        return WindowDecision(window_size, reason)

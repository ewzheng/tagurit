"""
Write optional CSV events for client and cloudlet timing experiments.

Monotonic timestamps measure durations ONLY within the same process. Unix
timestamps help correlate logs but require synchronized clocks across hosts.
"""

import csv
import time
from collections.abc import Callable
from pathlib import Path

FIELDS = (
    "event",
    "session_id",
    "frame_id",
    "monotonic_s",
    "unix_ns",
    "live_count",
    "bank_count",
    "inflight_count",
    "codec",
    "jpeg_bytes",
    "wire_bytes",
    "attempt",
    "submitted_count",
)


class EventLog:
    """
    Flush each event to an exclusively created file; existing files RAISE.

    Calls MUST come from one process and be serialized by the caller.

    Parameters:
        - path (Path): Destination CSV, including its parent directory
        - clock (Callable): Monotonic clock, replaceable for offline tests
    """

    def __init__(self, path: Path, clock: Callable[[], float] = time.monotonic) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("x", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDS)
        self._writer.writeheader()
        self._file.flush()
        self._clock = clock

    def record(
        self,
        event: str,
        session_id: str,
        frame_id: int = 0,
        *,
        at: float | None = None,
        **values: str | int,
    ) -> None:
        """
        Write one event; frame ID zero denotes an event for the whole run.

        Parameters:
            - event (str): Lifecycle event name
            - session_id (str): Application session UUID
            - frame_id (int): Application frame identity
            - at (float | None): Explicit local monotonic timestamp
            - values (str | int): Additional columns declared in FIELDS

        Return: void
        """
        self._writer.writerow(
            {
                "event": event,
                "session_id": session_id,
                "frame_id": frame_id,
                "monotonic_s": self._clock() if at is None else at,
                "unix_ns": time.time_ns(),
                **values,
            }
        )
        self._file.flush()

    def close(self) -> None:
        """
        Close the CSV after the run, including when cancelled.

        Return: void
        """
        self._file.close()

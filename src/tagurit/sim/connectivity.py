"""
Simulate connection availability using timed events

This module does not inspect the network or the Gabriel connection
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass


# Hold one planned change in connection availability
@dataclass(frozen=True)
class ConnectivityEvent:
    """
    Describe a connection change at a time measured from monitor startup

    Events are immutable and are applied by the simulated connectivity monitor

    Parameters:
        - after_seconds (float): Time since monitor startup
        - available (bool): Whether communication is allowed after this event
    """

    after_seconds: float  # Time since monitor startup
    available: bool      # Whether communication is allowed


# Report connection availability using a fixed schedule
class ConnectivityMonitor:
    """
    Apply scheduled availability changes based on elapsed time

    The schedule MUST start at zero and its event times must strictly increase
    The supplied clock should move forward and can be controlled by tests
    This monitor NEVER changes the actual network connection

    Parameters:
        - schedule (tuple[ConnectivityEvent, ...] | None): Events or the default schedule
        - clock (Callable[[], float]): Clock returning time in seconds
    """

    # Store the schedule and record when the monitor starts
    def __init__(self, schedule: tuple[ConnectivityEvent, ...] | None = None, clock: Callable[[], float] = time.monotonic) -> None:

        # Load defaults after this module has finished defining its types
        if schedule is None:
            from tagurit.sim.config import CONNECTIVITY_SCHEDULE

            schedule = CONNECTIVITY_SCHEDULE

        # Keep our own fixed copy of the supplied schedule
        self._schedule = tuple(schedule)
        self._validate_schedule()

        # Use the supplied clock so tests can control elapsed time
        self._clock = clock
        self._start_time = self._clock()

    # Check that the schedule starts at zero and moves forward in time
    def _validate_schedule(self) -> None:

        # Require an initial connection setting
        if not self._schedule:
            raise ValueError("Connectivity schedule must not be empty")

        if self._schedule[0].after_seconds != 0.0:
            raise ValueError("Connectivity schedule must start at zero")

        # Check each event against the time of the previous event
        previous_time = -1.0

        for event in self._schedule:
            if not math.isfinite(event.after_seconds):
                raise ValueError("Schedule times must be finite")

            if event.after_seconds <= previous_time:
                raise ValueError("Schedule times must strictly increase")

            if not isinstance(event.available, bool):
                raise ValueError("Connection availability must be True or False")

            previous_time = event.after_seconds

    # Return the connection setting for the current elapsed time
    def is_available(self) -> bool:
        """
        Read the availability defined by the latest event whose time has arrived

        The final event remains in effect after the schedule ends
        Calling this method does not consume or change the schedule

        Return:
            True when the schedule allows communication and False otherwise
        """

        # Measure time since this monitor was created
        elapsed_seconds = self._clock() - self._start_time
        available = self._schedule[0].available

        # Apply events up to the current time and stop at the first future event
        for event in self._schedule:
            if event.after_seconds > elapsed_seconds:
                break

            available = event.available

        return available
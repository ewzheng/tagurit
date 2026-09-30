"""
Manage live frames, stored frames and unresolved transmissions

Frames remain owned by the scheduler until acknowledged
This module has no Gabriel dependency
"""

import heapq
import math
import time
from collections import deque
from collections.abc import Callable

from tagurit.client.config import LIVE_FRAME_WEIGHT, LIVE_TO_BANK_INTERVAL_SECONDS, STORED_FRAME_WEIGHT
from tagurit.client.scheduler_datatypes import LiveQueueEntry, PriorityBankEntry, SchedulerState
from tagurit.protocol import ImageFrame


# Hold waiting frames and retain one selected frame until acknowledged
class FrameScheduler:
    """
    Route scored frames and select their transmission order

    Live frames use FIFO order and banked frames use highest priority first
    Equal scores use original scheduler arrival order across both queues
    Calls MUST be serialized by the caller

    Parameters:
        - live_weight (int): Live turns per reintegration cycle
        - stored_weight (int): Stored turns per reintegration cycle
        - migration_interval (float): Seconds between live-to-bank moves during outages
        - clock (Callable): Monotonic clock used for timing and offline tests
    """

    # Create the queues, turn pattern and disconnected timer
    def __init__(self, live_weight: int = LIVE_FRAME_WEIGHT, stored_weight: int = STORED_FRAME_WEIGHT, migration_interval: float = LIVE_TO_BANK_INTERVAL_SECONDS, clock: Callable[[], float] = time.monotonic) -> None:

        # Reject invalid queue weights and timer intervals
        if type(live_weight) is not int or type(stored_weight) is not int:
            raise ValueError("Queue weights must be positive integers")

        if min(live_weight, stored_weight) < 1:
            raise ValueError("Queue weights must be positive integers")

        if isinstance(migration_interval, bool) or not isinstance(migration_interval, (int, float)):
            raise ValueError("Migration interval must be a positive finite number")

        if not math.isfinite(migration_interval) or migration_interval <= 0:
            raise ValueError("Migration interval must be a positive finite number")

        # Start with empty queues and an available connection
        self._connected = True
        self._state = SchedulerState.CONNECTED
        self._live_queue: deque[LiveQueueEntry] = deque()
        self._priority_bank: list[PriorityBankEntry] = []
        self._arrival_order = 0

        # Set the order of live and stored turns
        self._turn_pattern = [True] * live_weight + [False] * stored_weight
        self._turn_index = 0

        # Keep one selected frame until its receipt arrives
        self._inflight_frame: ImageFrame | None = None
        self._inflight_from_bank = False

        # Start the timer only when entering disconnected mode
        self._clock = clock
        self._migration_interval = migration_interval
        self._next_migration_at: float | None = None

    # Return the current mode
    @property
    def state(self) -> SchedulerState:
        """
        Read the current operating state

        Return:
            Current SchedulerState
        """
        return self._state

    # Update connection availability without discarding frames
    def set_connected(self, connected: bool) -> None:
        """
        Update the mode using communication availability and stored work

        Repeated down reports MUST NOT restart the disconnected timer

        Parameters:
            - connected (bool): Whether communication is available

        Return:
            void
        """
        self._connected = connected
        self._update_state()

    # Update the mode and reset timers only on a state change
    def _update_state(self) -> None:
        stored_work_remains = bool(self._priority_bank) or self._inflight_from_bank

        if not self._connected:
            new_state = SchedulerState.DISCONNECTED
        elif stored_work_remains:
            new_state = SchedulerState.REINTEGRATING
        else:
            new_state = SchedulerState.CONNECTED

        # Keep the current timing and turn position when the mode is unchanged
        if new_state == self._state:
            return

        self._state = new_state
        self._turn_index = 0

        # Each outage starts a fresh interval and reconnection cancels it
        if new_state == SchedulerState.DISCONNECTED:
            self._next_migration_at = self._clock() + self._migration_interval
        else:
            self._next_migration_at = None

    # Assign an arrival order before placing the frame in either queue
    def add_frame(self, frame: ImageFrame) -> None:
        """
        Retain an incoming scored frame without changing it

        Disconnected arrivals enter the bank and other arrivals enter the live queue

        Parameters:
            - frame (ImageFrame): Scored image with an ID unique within this run

        Return:
            void
        """
        arrival_order = self._arrival_order

        if self._state == SchedulerState.DISCONNECTED:
            entry = PriorityBankEntry(arrival_order=arrival_order, frame=frame)
            heapq.heappush(self._priority_bank, entry)
        else:
            self._live_queue.append(LiveQueueEntry(arrival_order=arrival_order, frame=frame))

        self._arrival_order += 1

    # Move one oldest waiting live frame when the disconnected timer is due
    def migrate_live_frame(self) -> ImageFrame | None:
        """
        Poll the timer and move at most one waiting live frame into the bank

        The caller MUST poll this method while running the client
        Original arrival order is preserved and the in-flight frame is untouched
        Delayed polls move only one frame rather than catching up in a burst

        Return:
            Moved ImageFrame or None when no move occurs
        """

        # Only disconnected mode has an active migration timer
        if self._state != SchedulerState.DISCONNECTED or self._next_migration_at is None:
            return None

        now = self._clock()

        if now < self._next_migration_at:
            return None

        # Schedule the next check even when there is no waiting live frame
        self._next_migration_at = now + self._migration_interval

        if not self._live_queue:
            return None

        # Preserve the frame and its original order when changing queues
        live_entry = self._live_queue[0]
        bank_entry = PriorityBankEntry(arrival_order=live_entry.arrival_order, frame=live_entry.frame)
        heapq.heappush(self._priority_bank, bank_entry)
        self._live_queue.popleft()
        return live_entry.frame

    # Select one frame while keeping it in the in-flight slot
    def reserve_next_frame(self) -> ImageFrame | None:
        """
        Select a waiting frame and retain it until acknowledgment

        MUTATES the queues and in-flight slot
        No frame is selected during an outage or while a receipt is outstanding

        Return:
            Selected ImageFrame or None
        """
        if not self._connected or self._inflight_frame is not None:
            return None

        if not self._live_queue and not self._priority_bank:
            return None

        # Connected mode uses FIFO and reintegration uses the turn pattern
        from_bank = False

        if self._state == SchedulerState.CONNECTED:
            frame = self._live_queue.popleft().frame
        else:
            use_live_queue = self._turn_pattern[self._turn_index]
            self._turn_index = (self._turn_index + 1) % len(self._turn_pattern)

            if use_live_queue and self._live_queue:
                frame = self._live_queue.popleft().frame
            else:
                frame = heapq.heappop(self._priority_bank).frame
                from_bank = True

        self._inflight_frame = frame
        self._inflight_from_bank = from_bank
        return frame

    # Release the selected frame after its receipt has been checked
    def acknowledge_frame(self, frame_id: int) -> None:
        """
        Release the unresolved frame and update the operating state

        The transport MUST validate receipt identity and contents first
        A mismatched frame ID RAISES ValueError

        Parameters:
            - frame_id (int): ID of the acknowledged image

        Return:
            void
        """
        if self._inflight_frame is None or self._inflight_frame.frame_id != frame_id:
            raise ValueError("Receipt does not match the in-flight frame")

        self._inflight_frame = None
        self._inflight_from_bank = False
        self._update_state()

    # Complete a selection immediately for offline scheduler tests
    def pop_next_frame(self) -> ImageFrame | None:
        """
        Select and acknowledge a frame for tests only

        Real transmission MUST reserve and acknowledge separately

        Return:
            Selected ImageFrame or None
        """
        frame = self.reserve_next_frame()

        if frame is not None:
            self.acknowledge_frame(frame.frame_id)

        return frame

    # Return which queue supplied the in-flight frame
    def inflight_lane(self) -> str:
        """
        Read the origin of the unresolved frame

        Return:
            bank or live with live also returned when no frame is in flight
        """
        return "bank" if self._inflight_from_bank else "live"

    # Count waiting live frames
    def live_count(self) -> int:
        """
        Count live frames excluding the in-flight frame

        Return:
            Number of waiting live frames
        """
        return len(self._live_queue)

    # Count waiting stored frames
    def stored_count(self) -> int:
        """
        Count banked frames excluding the in-flight frame

        Return:
            Number of waiting stored frames
        """
        return len(self._priority_bank)

    # Count unresolved transmissions
    def inflight_count(self) -> int:
        """
        Count frames waiting for acknowledgment

        Return:
            Zero or one
        """
        return int(self._inflight_frame is not None)

    # Count all unfinished work
    def pending_count(self) -> int:
        """
        Count retained frames across both queues and the in-flight slot

        Return:
            Total number of unfinished frames
        """
        return self.live_count() + self.stored_count() + self.inflight_count()
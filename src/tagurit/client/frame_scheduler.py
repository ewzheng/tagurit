"""
Manage live frames, stored frames and unresolved transmissions

Frames remain owned by the scheduler until acknowledged
This module has no Gabriel dependency
"""

import heapq
import math
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import replace

from tagurit.client.config import (
    INFLIGHT_WINDOW,
    LIVE_FRAME_WEIGHT,
    LIVE_TO_BANK_INTERVAL_SECONDS,
    STORED_FRAME_WEIGHT,
)
from tagurit.client.scheduler_datatypes import (
    BankOrder,
    InflightEntry,
    LiveQueueEntry,
    PriorityBankEntry,
    SchedulerState,
)
from tagurit.protocol import ImageFrame


# Hold waiting frames and retain selected frames until acknowledged
class FrameScheduler:
    """
    Route scored frames and select their transmission order

    Live frames use FIFO order and banked frames use highest priority first
    Equal scores use original scheduler arrival order across both queues
    A supplied bank order replaces highest priority first for banked frames
    It sees waiting frames and every reserved or acknowledged frame so repeats can wait
    Calls MUST be serialized by the caller

    Parameters:
        - live_weight (int): Live turns per reintegration cycle
        - stored_weight (int): Stored turns per reintegration cycle
        - migration_interval (float): Seconds between live-to-bank moves during outages
        - window_size (int): Maximum number of reserved frames
        - clock (Callable): Monotonic clock used for timing and offline tests
        - bank_order (BankOrder | None): Chooses the next banked frame or None for highest priority
    """

    # Create the queues, turn pattern and disconnected timer
    def __init__(
        self,
        live_weight: int = LIVE_FRAME_WEIGHT,
        stored_weight: int = STORED_FRAME_WEIGHT,
        migration_interval: float = LIVE_TO_BANK_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        window_size: int = INFLIGHT_WINDOW,
        bank_order: BankOrder | None = None,
    ) -> None:

        if type(window_size) is not int or window_size < 1:
            raise ValueError("Window size must be a positive integer")
        self._window_size = window_size

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

        # Remember capture times and scores of acknowledged frames for the bank order
        self._bank_order = bank_order
        self._delivered: list[tuple[float, float]] = []

        # Set the order of live and stored turns
        self._turn_pattern = [True] * live_weight + [False] * stored_weight
        self._turn_index = 0

        # Keep selected frames until its receipt arrives
        self._inflight: dict[int, InflightEntry] = {}

        # Start the timer only when entering disconnected mode
        self._clock = clock
        self._migration_interval = migration_interval
        self._next_migration_at: float | None = None

    @property
    def window_size(self) -> int:
        """
        Return the current maximum number of reserved frames.

        Return: Positive in-flight window size
        """
        return self._window_size

    def set_window_size(self, window_size: int) -> None:
        """
        Change the reservation limit without releasing outstanding frames.

        If the new limit is below the current in-flight count, new reservations
        wait until enough matching receipts release those frames.

        Parameters:
            - window_size (int): New positive reservation limit

        Return: void
        """
        if type(window_size) is not int or window_size < 1:
            raise ValueError("Window size must be a positive integer")
        self._window_size = window_size

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
        stored_work_remains = bool(self._priority_bank) or any(
            entry.from_bank for entry in self._inflight.values()
        )

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
        bank_entry = PriorityBankEntry(
            arrival_order=live_entry.arrival_order, frame=live_entry.frame
        )
        heapq.heappush(self._priority_bank, bank_entry)
        self._live_queue.popleft()
        return live_entry.frame

    # Select a frame while retaining it in the in-flight window
    def reserve_next_frame(self) -> ImageFrame | None:
        """
        Select a waiting frame and retain it until acknowledgment

        MUTATES the queues and in-flight window
        No frame is selected during an outage or when the window is full

        Return:
            Selected ImageFrame or None
        """
        if not self._connected or len(self._inflight) >= self._window_size:
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

            if self._live_queue and (use_live_queue or not self._priority_bank):
                frame = self._live_queue.popleft().frame
            else:
                frame = self._take_banked_frame()
                from_bank = True

        self._inflight[frame.frame_id] = InflightEntry(frame, from_bank)
        return frame

    # List every frame still waiting in either queue
    def waiting_frames(self) -> list[ImageFrame]:
        """
        Read the frames waiting in the live queue and the bank

        In-flight frames are left out because their transmission has started

        Return:
            Waiting ImageFrames in original arrival order
        """
        entries = [*self._live_queue, *self._priority_bank]
        return [entry.frame for entry in sorted(entries, key=lambda entry: entry.arrival_order)]

    # Replace the scores of waiting frames after the query changes
    def reprioritize(self, priorities: Mapping[int, float]) -> int:
        """
        Give waiting frames new scores while keeping their place and contents

        MUTATES both queues and reorders the bank
        IDs that are not waiting are ignored and in-flight frames are untouched
        Acknowledged-frame history is cleared because its scores used the old query
        An invalid score RAISES ValueError and changes nothing

        Parameters:
            - priorities (Mapping[int, float]): New score between 0 and 1 by frame ID

        Return:
            Number of waiting frames that received a new score
        """

        # Build every replacement first so a bad score leaves the queues unchanged
        def rescored(frame: ImageFrame) -> ImageFrame:
            if frame.frame_id not in priorities:
                return frame

            return replace(frame, priority=priorities[frame.frame_id])

        live = [
            LiveQueueEntry(entry.arrival_order, rescored(entry.frame)) for entry in self._live_queue
        ]
        bank = [
            PriorityBankEntry(arrival_order=entry.arrival_order, frame=rescored(entry.frame))
            for entry in self._priority_bank
        ]
        changed = sum(entry.frame.frame_id in priorities for entry in [*live, *bank])

        self._live_queue = deque(live)
        self._priority_bank = bank
        heapq.heapify(self._priority_bank)
        self._delivered.clear()
        return changed

    # List waiting banked frames in arrival order followed by frames already sent
    def _bank_view(self) -> tuple[list[PriorityBankEntry], list[float], list[float], list[bool]]:
        entries = sorted(self._priority_bank, key=lambda entry: entry.arrival_order)

        # Reserved frames count as sent so their encounters are not chosen twice
        reserved = [
            (entry.frame.timestamp, entry.frame.priority) for entry in self._inflight.values()
        ]
        gone = [*self._delivered, *reserved]
        timestamps = [entry.frame.timestamp for entry in entries] + [t for t, _ in gone]
        priorities = [entry.frame.priority for entry in entries] + [p for _, p in gone]
        sent = [False] * len(entries) + [True] * len(gone)
        return entries, timestamps, priorities, sent

    # Ask the bank order for a waiting frame and reject any other answer
    def _choose(
        self,
        entries: list[PriorityBankEntry],
        timestamps: list[float],
        priorities: list[float],
        sent: list[bool],
    ) -> int:
        choice = self._bank_order(timestamps, priorities, sent)

        if choice is None or not 0 <= choice < len(entries) or sent[choice]:
            raise ValueError("Bank order must choose a waiting banked frame")

        return choice

    # Remove the next banked frame by score or by the supplied order
    def _take_banked_frame(self) -> ImageFrame:
        if self._bank_order is None:
            return heapq.heappop(self._priority_bank).frame

        entries, timestamps, priorities, sent = self._bank_view()
        entry = entries[self._choose(entries, timestamps, priorities, sent)]
        self._priority_bank.remove(entry)
        heapq.heapify(self._priority_bank)
        return entry.frame

    # Preview the next banked frames without selecting them
    def stored_order(self, limit: int) -> list[int]:
        """
        Read the IDs of the next banked frames in the order the bank would send them

        The preview assumes no further arrivals and ignores live turns
        The queues are NOT changed

        Parameters:
            - limit (int): Most frame IDs to return

        Return:
            Frame IDs with the next to send first
        """
        if self._bank_order is None:
            return [entry.frame.frame_id for entry in heapq.nsmallest(limit, self._priority_bank)]

        entries, timestamps, priorities, sent = self._bank_view()
        order: list[int] = []

        # Mark each preview pick as sent so the next pick follows it
        while len(order) < min(limit, len(entries)):
            choice = self._choose(entries, timestamps, priorities, sent)
            sent[choice] = True
            order.append(entries[choice].frame.frame_id)

        return order

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
        if frame_id not in self._inflight:
            raise ValueError("Receipt does not match the in-flight frame")

        # Frames delivered from either queue count toward the bank order
        if self._bank_order is not None:
            frame = self._inflight[frame_id].frame
            self._delivered.append((frame.timestamp, frame.priority))

        del self._inflight[frame_id]
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
    def inflight_lane(self, frame_id: int | None = None) -> str:
        """
        Read the origin of the unresolved frame

        Return:
            bank or live with live also returned when no frame is in flight
        """
        if frame_id is None:
            if not self._inflight:
                return "live"
            if len(self._inflight) > 1:
                raise ValueError("Specify a frame ID when multiple frames are in flight")
            frame_id = next(iter(self._inflight))
        return "bank" if self._inflight[frame_id].from_bank else "live"

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
            Number of reserved frames
        """
        return len(self._inflight)

    # Count all unfinished work
    def pending_count(self) -> int:
        """
        Count retained frames across both queues and the in-flight window

        Return:
            Total number of unfinished frames
        """
        return self.live_count() + self.stored_count() + self.inflight_count()

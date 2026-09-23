"""
Manage live frames, stored frames and unresolved transmissions

Frames remain owned by the scheduler until acknowledged
This module has no Gabriel dependency
"""

import heapq
from collections import deque

from tagurit.client.config import LIVE_FRAME_WEIGHT, STORED_FRAME_WEIGHT
from tagurit.client.scheduler_datatypes import PriorityBankEntry, SchedulerState
from tagurit.protocol import ImageFrame


# Hold waiting frames and retain one selected frame until acknowledged
class FrameScheduler:
    """
    Route incoming frames and select their transmission order

    Live frames use arrival order and stored frames use highest priority first
    Equal priorities use arrival order
    Only one frame can await acknowledgment at a time

    Calls MUST be serialized by the caller
    This class does not provide thread synchronization

    Parameters:
        - live_weight (int): Live turns per reintegration cycle
        - stored_weight (int): Stored turns per reintegration cycle
    """

    # Create the queues and the turn pattern
    def __init__(self, live_weight: int = LIVE_FRAME_WEIGHT, stored_weight: int = STORED_FRAME_WEIGHT) -> None:

        # Require integer weights before checking their values
        if type(live_weight) is not int or type(stored_weight) is not int:
            raise ValueError("Queue weights must be positive integers")

        if min(live_weight, stored_weight) < 1:
            raise ValueError("Queue weights must be positive integers")

        # Start with empty queues and an available connection
        self._connected = True
        self._state = SchedulerState.CONNECTED
        self._live_queue: deque[ImageFrame] = deque()
        self._priority_bank: list[PriorityBankEntry] = []
        self._arrival_order = 0

        # Set the order of live and stored turns
        self._turn_pattern = [True] * live_weight + [False] * stored_weight
        self._turn_index = 0

        # Keep one selected frame until its receipt arrives
        self._inflight_frame: ImageFrame | None = None
        self._inflight_from_bank = False

    # Return the current mode
    @property
    def state(self) -> SchedulerState:
        """
        Read the current operating state

        Return:
            SchedulerState describing the current mode
        """
        return self._state

    # Update connection availability and choose the matching mode
    def set_connected(self, connected: bool) -> None:
        """
        Update communication availability without discarding any frames

        Changes the operating state based on availability and stored work

        Parameters:
            - connected (bool): Whether communication is available

        Return:
            void
        """
        self._connected = connected
        self._update_state()

    # Keep reintegration active until the last stored frame is acknowledged
    def _update_state(self) -> None:

        # Stored work includes a banked frame currently in flight
        stored_work_remains = bool(self._priority_bank) or self._inflight_from_bank

        if not self._connected:
            new_state = SchedulerState.DISCONNECTED
        elif stored_work_remains:
            new_state = SchedulerState.REINTEGRATING
        else:
            new_state = SchedulerState.CONNECTED

        # Restart the turn pattern only when the mode changes
        if new_state != self._state:
            self._state = new_state
            self._turn_index = 0

    # Place an incoming frame in the queue used by the current mode
    def add_frame(self, frame: ImageFrame) -> None:
        """
        Retain an incoming frame without changing it

        Disconnected frames enter the priority bank and MUST have a score
        Other frames enter the live queue in arrival order

        Parameters:
            - frame (ImageFrame): Incoming image and its metadata

        Return:
            void
        """

        # Disconnected frames need a score before entering the bank
        if self._state == SchedulerState.DISCONNECTED:
            entry = PriorityBankEntry(arrival_order=self._arrival_order, frame=frame)
            heapq.heappush(self._priority_bank, entry)
            self._arrival_order += 1
        else:
            self._live_queue.append(frame)

    # Select one frame while keeping it in the in-flight slot
    def reserve_next_frame(self) -> ImageFrame | None:
        """
        Select a waiting frame and retain it until acknowledgment

        MUTATES the queues and in-flight slot
        Returns None during disconnection, while a frame is unresolved,
        or when no frames are waiting

        Return:
            Selected ImageFrame or None
        """

        # Do not select during an outage or while a receipt is outstanding
        if not self._connected or self._inflight_frame is not None:
            return None

        if not self._live_queue and not self._priority_bank:
            return None

        # Connected mode serves live frames in arrival order
        from_bank = False

        if self._state == SchedulerState.CONNECTED:
            frame = self._live_queue.popleft()
        else:
            use_live_queue = self._turn_pattern[self._turn_index]
            self._turn_index = (self._turn_index + 1) % len(self._turn_pattern)

            # Use a stored frame if it is its turn or no live frame is ready
            if use_live_queue and self._live_queue:
                frame = self._live_queue.popleft()
            else:
                frame = heapq.heappop(self._priority_bank).frame
                from_bank = True

        # Keep the selected frame until its matching receipt arrives
        self._inflight_frame = frame
        self._inflight_from_bank = from_bank
        return frame

    # Release the selected frame after its receipt has been checked
    def acknowledge_frame(self, frame_id: int) -> None:
        """
        Release the unresolved frame and update the operating state

        The transport MUST validate the receipt identity and contents first
        A frame ID that does not match the selected frame RAISES ValueError

        Parameters:
            - frame_id (int): ID of the acknowledged image

        Return:
            void
        """

        # Reject acknowledgments for any frame other than the selected one
        if self._inflight_frame is None or self._inflight_frame.frame_id != frame_id:
            raise ValueError("Receipt does not match the in-flight frame")

        self._inflight_frame = None
        self._inflight_from_bank = False
        self._update_state()

    # Keep the immediate-completion helper for existing scheduler tests
    def pop_next_frame(self) -> ImageFrame | None:
        """
        Select a frame and immediately mark it complete for offline tests

        Real transmission MUST reserve and acknowledge separately

        Return:
            Selected ImageFrame or None
        """
        frame = self.reserve_next_frame()

        # Complete the selection immediately for tests
        if frame is not None:
            self.acknowledge_frame(frame.frame_id)

        return frame

    # Return which queue supplied the in-flight frame
    def inflight_lane(self) -> str:
        """
        Describe the origin of the unresolved frame

        Only meaningful while a frame is in flight
        Returns live when no frame is in flight

        Return:
            bank or live
        """
        return "bank" if self._inflight_from_bank else "live"

    # Return the number of live frames waiting
    def live_count(self) -> int:
        """
        Count waiting live frames excluding the in-flight frame

        Return:
            Number of live frames waiting
        """
        return len(self._live_queue)

    # Return the number of stored frames waiting
    def stored_count(self) -> int:
        """
        Count banked frames excluding the in-flight frame

        Return:
            Number of stored frames waiting
        """
        return len(self._priority_bank)

    # Return the number of frames awaiting acknowledgment
    def inflight_count(self) -> int:
        """
        Count unresolved transmissions

        Return:
            Zero or one
        """
        return int(self._inflight_frame is not None)

    # Return all unfinished work including the in-flight frame
    def pending_count(self) -> int:
        """
        Count every retained frame across both queues and the in-flight slot

        Return:
            Total number of unfinished frames
        """
        return self.live_count() + self.stored_count() + self.inflight_count()
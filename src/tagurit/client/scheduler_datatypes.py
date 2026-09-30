"""
Define operating states and entries for the live queue and priority bank
"""

from dataclasses import dataclass, field
from enum import Enum

from tagurit.protocol import ImageFrame


# Name the three modes used by the scheduler
class SchedulerState(Enum):
    """
    Identify whether frames can be sent and which queues receive them
    """

    CONNECTED = "CONNECTED"
    DISCONNECTED = "DISCONNECTED"
    REINTEGRATING = "REINTEGRATING"


# Keep the original arrival order with each waiting live frame
@dataclass(frozen=True)
class LiveQueueEntry:
    """
    Retain a live frame and its original scheduler arrival order

    Parameters:
        - arrival_order (int): Sequence number assigned when the frame arrives
        - frame (ImageFrame): Scored image waiting for transmission
    """

    arrival_order: int  # Original arrival order at the scheduler
    frame: ImageFrame  # Image waiting in the live queue


# Order banked frames by score and then original arrival order
@dataclass(order=True, frozen=True)
class PriorityBankEntry:
    """
    Select higher scores first and earlier arrivals when scores match

    Moving a live frame into the bank MUST preserve its arrival_order

    Parameters:
        - arrival_order (int): Sequence number assigned when the frame arrives
        - frame (ImageFrame): Scored image held in the bank
    """

    sort_priority: float = field(init=False)  # Negative score for heap ordering
    arrival_order: int                        # Original scheduler arrival order
    frame: ImageFrame = field(compare=False)  # Image held by this entry

    # Set the heap sorting value while creating this frozen entry
    def __post_init__(self) -> None:
        object.__setattr__(self, "sort_priority", -self.frame.priority)
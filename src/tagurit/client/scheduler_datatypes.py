"""
Define the operating states and internal priority bank entries
"""

from dataclasses import dataclass, field
from enum import Enum

from tagurit.protocol import ImageFrame


# Name the three modes used by the scheduler
class SchedulerState(Enum):
    """
    Identify the scheduler's current operating mode

    Disconnected mode retains new scored frames
    Reintegrating mode shares transmission between live and stored frames
    Connected mode sends live frames in arrival order
    """

    CONNECTED = "CONNECTED"
    DISCONNECTED = "DISCONNECTED"
    REINTEGRATING = "REINTEGRATING"


# Hold the values used to order a frame in the priority bank
@dataclass(order=True, frozen=True)
class PriorityBankEntry:
    """
    Order stored frames by highest priority and then earliest arrival
    An unscored frame RAISES ValueError

    Parameters:
        - arrival_order (int): Sequence number assigned when the frame is stored
        - frame (ImageFrame): Scored image retained by this entry
    """

    sort_priority: float = field(init=False)  # Negative score for heap ordering
    arrival_order: int                        # Earlier arrivals win equal scores
    frame: ImageFrame = field(compare=False)  # Frame held by this entry

    # Build the sorting value from the frame's score
    def __post_init__(self) -> None:

        # Only scored frames can enter the priority bank
        if self.frame.priority is None:
            raise ValueError("A stored frame must have a priority score")

        # Set the derived field while creating the frozen entry
        object.__setattr__(self, "sort_priority", -self.frame.priority)
"""
Define the scored image frame shared by perception and client scheduling
"""

import math
from dataclasses import dataclass, field


# Hold one scored image for handoff to the scheduler
@dataclass(frozen=True)
class ImageFrame:
    """
    Carry an immutable scored image from its source to the scheduler

    Every frame MUST have a priority and an ID unique within the session
    Invalid scores RAISE ValueError

    Parameters:
        - frame_id (int): Unique image number within the session
        - timestamp (float): Capture time in seconds since the Unix epoch
        - image_bytes (bytes): Encoded image contents
        - priority (float): Score between 0 and 1
        - kept (object | None): What the tagger kept to score this frame again without its
          image when the query changes or None; opaque to the client and never transmitted
    """

    frame_id: int       # Unique image number within the session
    timestamp: float    # Capture time in seconds since the Unix epoch
    image_bytes: bytes  # Encoded JPEG contents
    priority: float     # Required score between 0 and 1
    kept: object | None = field(default=None, compare=False, repr=False)  # Tagger's rescoring state

    # Reject missing scores and values outside the allowed range
    def __post_init__(self) -> None:
        if isinstance(self.priority, bool) or not isinstance(self.priority, (int, float)):
            raise ValueError("Every frame must have a numeric priority score between 0 and 1")

        if not math.isfinite(self.priority) or not 0.0 <= self.priority <= 1.0:
            raise ValueError("Priority must be between 0 and 1")
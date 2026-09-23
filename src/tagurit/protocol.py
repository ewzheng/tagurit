"""
Define the image frame shared by perception and client scheduling

This module contains the handoff datatype and has no model or transport imports
"""

import math
from dataclasses import dataclass


# Hold one image and its metadata for handoff to the scheduler
@dataclass(frozen=True)
class ImageFrame:
    """
    Carry an image from its source to the client scheduler

    The frame is immutable and its ID must be unique within the capture session
    A scored frame has a priority between 0 and 1
    An unscored live frame uses None and cannot enter the priority bank

    Parameters:
        - frame_id (int): Unique image number within the capture session
        - timestamp (float): Capture time in seconds since the Unix epoch
        - image_bytes (bytes): Encoded image contents
        - priority (float | None): Interest score or None when unscored
    """

    frame_id: int                  # Unique image number within the session
    timestamp: float               # Capture time in seconds since the Unix epoch
    image_bytes: bytes             # Encoded JPEG contents
    priority: float | None = None   # Score between 0 and 1 or None when unscored

    # Check the priority after the frame is created
    def __post_init__(self) -> None:

        # Only check the score when one has been supplied
        if self.priority is not None:
            if not math.isfinite(self.priority) or not 0.0 <= self.priority <= 1.0:
                raise ValueError("Priority must be between 0 and 1 or None")
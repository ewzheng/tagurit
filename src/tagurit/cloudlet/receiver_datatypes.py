"""
Define the receiver's internal record of an accepted image
"""

from dataclasses import dataclass

from tagurit.shared.image_protocol import ImageMessage, ImageReceipt


# Keep one accepted image and its original receipt
@dataclass(frozen=True)
class AcceptedImage:
    """
    Retain an accepted image and the receipt returned for matching retries

    The receiver uses this record to avoid accepting the same image twice
    This record is held in memory and does not survive a server restart

    Parameters:
        - message (ImageMessage): Accepted image with its identity and hash
        - receipt (ImageReceipt): Original confirmation returned for the image
    """

    message: ImageMessage  # Identity and image bytes retained by the receiver
    receipt: ImageReceipt  # Original receipt returned on a matching retry
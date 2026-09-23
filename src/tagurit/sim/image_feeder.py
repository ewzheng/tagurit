"""
Read sample images and scores from a manifest for simulated frame arrivals

Image paths are relative to the manifest's folder
"""

import csv
import time
from collections.abc import Iterator
from pathlib import Path

from tagurit.protocol import ImageFrame
from tagurit.sim.config import MANIFEST_PATH


# Read the manifest and yield one image frame at a time in row order
def feed_images(manifest_path: str | Path = MANIFEST_PATH) -> Iterator[ImageFrame]:
    """
    Load images and their sample scores in manifest order

    Images are read when the caller requests the next frame
    Frame IDs start at one for each call and timestamps record loading time
    Missing files and invalid rows RAISE rather than being skipped

    Parameters:
        - manifest_path (str | Path): CSV file containing image and priority columns

    Return:
        Iterator yielding ImageFrame records in row order
    """

    # Get image paths relative to the manifest's folder
    manifest_path = Path(manifest_path)
    image_directory = manifest_path.parent

    # Keep the manifest open while reading its rows
    with manifest_path.open("r", newline="", encoding="utf-8") as manifest_file:
        reader = csv.DictReader(manifest_file)

        # Check that the required columns exist
        required_columns = {"image", "priority"}

        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError("Manifest must contain 'image' and 'priority' columns")

        # Load each image and assign its ID in arrival order
        for frame_id, row in enumerate(reader, start=1):
            image_path = image_directory / row["image"]
            priority = float(row["priority"])
            image_bytes = image_path.read_bytes()

            # Build the shared frame using the image and its test score
            frame = ImageFrame(
                frame_id=frame_id,
                timestamp=time.time(),
                image_bytes=image_bytes,
                priority=priority
            )

            # Give the caller one frame and wait for the next request
            yield frame
"""
Tag captured images and hand them to the client: the edge device's pipeline.

Images arrive as encoded bytes from whatever captures them, a camera on the
scout or a trace replay in ``sim``. Each one is stamped with a frame ID and
its arrival time, scored by a tagger into a ``protocol.ImageFrame``, and
passed to ``client.main.run_client``, which queues, banks and sends it.
Tagging is blocking compute, so it runs in a worker thread and NEVER stalls
Gabriel transport on the event loop. This module never imports ``sim``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterable, AsyncIterator, Callable
from typing import Protocol

from tagurit.client.frame_scheduler import FrameScheduler
from tagurit.client.main import run_client
from tagurit.protocol import ImageFrame


class FrameTagger(Protocol):
    """
    Anything that scores an encoded image into a frame; ``tagging.Tagger`` is one.
    """

    def tag(self, frame_id: int, timestamp: float, image_bytes: bytes) -> ImageFrame:
        """
        Score one encoded image.

        Parameters:
            - frame_id (int): unique image number within the session
            - timestamp (float): capture time in seconds since the Unix epoch
            - image_bytes (bytes): encoded image, e.g. JPEG

        Return: the image with its priority
        """
        ...


async def tag_frames(
    images: AsyncIterable[bytes],
    tagger: FrameTagger,
    clock: Callable[[], float] = time.time,
) -> AsyncIterator[ImageFrame]:
    """
    Stamp and tag each image as it arrives.

    Frame IDs count up from 1 in arrival order. The timestamp is read when
    an image arrives, before tagging, so it is the capture time rather than
    the time scoring finished. The next image is not requested until the
    current one is tagged, so a slow tagger slows the source down rather
    than piling images up. A tagger error RAISES out of the iterator.

    Parameters:
        - images (AsyncIterable[bytes]): encoded images in capture order
        - tagger (FrameTagger): scores each image, run in a worker thread
        - clock (Callable[[], float]): capture-time source, seconds since the epoch

    Return: asynchronous iterator of tagged frames in arrival order
    """
    frame_id = 0
    async for image_bytes in images:
        frame_id += 1
        timestamp = clock()
        yield await asyncio.to_thread(tagger.tag, frame_id, timestamp, image_bytes)


async def run(
    images: AsyncIterable[bytes],
    tagger: FrameTagger,
    report: Callable[[str, str, FrameScheduler], None],
) -> None:
    """
    Tag images and run the client on them until the images end and every frame is acknowledged.

    Cancellation stops the client and may leave unsent frames, as in
    ``client.main.run_client``.

    Parameters:
        - images (AsyncIterable[bytes]): encoded images in capture order
        - tagger (FrameTagger): scores each image
        - report (Callable): receives each client event, its detail, and the scheduler

    Return: void
    """
    await run_client(tag_frames(images, tagger), report)

"""
Tag captured images and hand them to the client: the edge device's pipeline.

Images arrive as encoded bytes from whatever captures them, a camera on the
scout or a trace replay in ``sim``. Each one is stamped with a frame ID and
its arrival time, scored by a tagger into a ``protocol.ImageFrame``, and
passed to ``client.main.run_client``, which queues, banks and sends it.
Tagging is blocking compute, so it runs in a worker thread and NEVER stalls
Gabriel transport on the event loop. Images that arrive while the tagger is
busy do not queue: only the newest waits, as a live camera needs. When the
mission changes, the tagger's query is switched and the frames still
waiting in the client are rescored from what the tagger kept of them, so
the bank reorders for the new query without reading an image. This module
never imports ``sim``.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from collections.abc import AsyncIterable, AsyncIterator, Callable, Mapping, Sequence
from typing import Protocol, runtime_checkable

from tagurit.client.config import SEND_INTERVAL_SECONDS
from tagurit.client.frame_scheduler import FrameScheduler
from tagurit.client.main import run_client
from tagurit.client.scheduler_datatypes import BankOrder, Rescorer
from tagurit.protocol import ImageFrame
from tagurit.shared.telemetry import EventLog


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


@runtime_checkable
class RetargetableTagger(FrameTagger, Protocol):
    """
    A tagger whose query can change mid-mission; ``tagging.Tagger`` with a CLIP scorer is one.
    """

    def set_targets(self, phrases: Sequence[str]) -> None:
        """
        Switch the query for every frame tagged from now on.

        Parameters:
            - phrases (Sequence[str]): the new query phrases

        Return: void
        """
        ...

    def rescore(self, frames: Sequence[ImageFrame]) -> Mapping[int, float]:
        """
        Priorities under the current query for frames tagged earlier.

        Parameters:
            - frames (Sequence[ImageFrame]): frames this tagger tagged

        Return: frame ID to new priority, for the frames it could rescore
        """
        ...


class _OneAtATime:
    """
    Run a tagger's calls one at a time.

    Tagging and retargeting run in worker threads and share the model's
    state, so a query change must wait for the frame being tagged.

    Parameters:
        - tagger (RetargetableTagger): the tagger to guard
    """

    def __init__(self, tagger: RetargetableTagger) -> None:
        self._tagger = tagger
        self._lock = threading.Lock()

    def tag(self, frame_id: int, timestamp: float, image_bytes: bytes) -> ImageFrame:
        with self._lock:
            return self._tagger.tag(frame_id, timestamp, image_bytes)

    def set_targets(self, phrases: Sequence[str]) -> None:
        with self._lock:
            self._tagger.set_targets(phrases)

    def rescore(self, frames: Sequence[ImageFrame]) -> Mapping[int, float]:
        with self._lock:
            return self._tagger.rescore(frames)


async def retarget(
    missions: AsyncIterable[Sequence[str]], tagger: RetargetableTagger
) -> AsyncIterator[Rescorer]:
    """
    Switch the tagger's query at each mission change and hand the client a rescorer for it.

    The query is switched before the rescorer is yielded, so frames tagged
    from then on already use it. The rescorer scores frames in a worker
    thread from what the tagger kept of them. The tagger's calls MUST be
    safe to run from different threads, one at a time.

    Parameters:
        - missions (AsyncIterable[Sequence[str]]): each mission's query phrases, as they change
        - tagger (RetargetableTagger): the tagger scoring new frames

    Return: asynchronous iterator of one rescorer per mission change
    """

    async def rescore(frames: Sequence[ImageFrame]) -> Mapping[int, float]:
        return await asyncio.to_thread(tagger.rescore, list(frames))

    async for phrases in missions:
        await asyncio.to_thread(tagger.set_targets, tuple(phrases))
        yield rescore


async def tag_frames(
    images: AsyncIterable[bytes],
    tagger: FrameTagger,
    clock: Callable[[], float] = time.time,
    drop_stale: bool = True,
    min_interval: float = 0.0,
    dropped: Callable[[int], None] | None = None,
) -> AsyncIterator[ImageFrame]:
    """
    Stamp each image as it arrives and tag the newest one whenever the tagger is free.

    Frame IDs count up from 1 in arrival order, and the timestamp is read
    when an image arrives, so it is the capture time rather than the time
    scoring finished. With ``drop_stale`` the source is read continuously
    and nothing queues behind the tagger: an image still waiting when a
    newer one arrives is dropped, so the tagger always works on the most
    recent view, and an image arriving less than ``min_interval`` seconds
    after the last one tagged is dropped at once. Dropped images leave gaps
    in the frame IDs. Such a source MUST pace itself, as a camera does; one
    that yields without waiting would have all but its last image dropped.
    Without ``drop_stale`` the next image is not requested until the
    current one is tagged, so a slow tagger slows the source down and every
    image is tagged, which suits replaying a trace in full. A tagger or
    source error RAISES out of the iterator. A negative ``min_interval``, or
    a positive one without ``drop_stale``, RAISES ValueError.

    Parameters:
        - images (AsyncIterable[bytes]): encoded images in capture order
        - tagger (FrameTagger): scores each image, run in a worker thread
        - clock (Callable[[], float]): capture-time source, seconds since the epoch
        - drop_stale (bool): keep only the newest waiting image
        - min_interval (float): least seconds between the capture times of tagged images
        - dropped (Callable[[int], None] | None): told the frame ID of each dropped image

    Return: asynchronous iterator of tagged frames in arrival order
    """
    if min_interval < 0 or (min_interval > 0 and not drop_stale):
        raise ValueError("min_interval must be non-negative and needs drop_stale")
    if not drop_stale:
        frame_id = 0
        async for image_bytes in images:
            frame_id += 1
            timestamp = clock()
            yield await asyncio.to_thread(tagger.tag, frame_id, timestamp, image_bytes)
        return

    waiting: list[tuple[int, float, bytes]] = []  # the newest untagged image, if any
    arrived = asyncio.Event()
    ended = False
    last_tagged = -math.inf

    def drop(frame_id: int) -> None:
        if dropped is not None:
            dropped(frame_id)

    async def read() -> None:
        nonlocal ended
        frame_id = 0
        try:
            async for image_bytes in images:
                frame_id += 1
                timestamp = clock()
                if timestamp - last_tagged < min_interval:
                    drop(frame_id)
                    continue
                if waiting:
                    drop(waiting[0][0])
                waiting[:] = [(frame_id, timestamp, image_bytes)]
                arrived.set()
        finally:
            ended = True
            arrived.set()

    reader = asyncio.create_task(read())
    try:
        while True:
            if not waiting:
                if ended:
                    break
                arrived.clear()
                await arrived.wait()
                continue
            frame_id, timestamp, image_bytes = waiting.pop()
            last_tagged = timestamp
            yield await asyncio.to_thread(tagger.tag, frame_id, timestamp, image_bytes)
        await reader  # a source error surfaces here
    finally:
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)


async def run(
    images: AsyncIterable[bytes],
    tagger: FrameTagger,
    report: Callable[[str, str, FrameScheduler], None],
    send_interval: float = SEND_INTERVAL_SECONDS,
    drop_stale: bool = True,
    min_interval: float = 0.0,
    dropped: Callable[[int], None] | None = None,
    bank_order: BankOrder | None = None,
    missions: AsyncIterable[Sequence[str]] | None = None,
    *,
    adaptive_window: bool = False,
    encoder: str = "auto",
    event_log: EventLog | None = None,
) -> None:
    """
    Tag images and run the client on them until the images end and every frame is acknowledged.

    Cancellation stops the client and may leave unsent frames, as in
    ``client.main.run_client``. ``drop_stale``, ``min_interval`` and
    ``dropped`` are ``tag_frames``'s. With ``missions``, each new set of
    query phrases switches the tagger and rescores the frames waiting in
    the client; a tagger that cannot change its query RAISES TypeError.

    Parameters:
        - images (AsyncIterable[bytes]): encoded images in capture order
        - tagger (FrameTagger): scores each image
        - report (Callable): receives each client event, its detail, and the scheduler
        - send_interval (float): minimum seconds between the client's image submissions
        - drop_stale (bool): keep only the newest image waiting for the tagger
        - min_interval (float): least seconds between the capture times of tagged images
        - dropped (Callable[[int], None] | None): told the frame ID of each dropped image
        - bank_order (BankOrder | None): picks the banked frame to send next, such as
          ``tagging.order.next_frame`` with its threshold bound; None sends by priority
        - missions (AsyncIterable[Sequence[str]] | None): query phrases of each new mission
        - adaptive_window (bool): adjust in-flight capacity from receipt feedback
        - encoder (str): auto, software, or hardware H.264 encoding backend
        - event_log (EventLog | None): optional caller-owned client event log

    Return: void
    """
    rescorers = None
    if missions is not None:
        if not isinstance(tagger, RetargetableTagger):
            raise TypeError(f"{type(tagger).__name__} cannot change its query")
        tagger = _OneAtATime(tagger)
        rescorers = retarget(missions, tagger)
    frames = tag_frames(
        images, tagger, drop_stale=drop_stale, min_interval=min_interval, dropped=dropped
    )
    await run_client(
        frames,
        report,
        send_interval=send_interval,
        adaptive_window=adaptive_window,
        encoder=encoder,
        event_log=event_log,
        bank_order=bank_order,
        rescorers=rescorers,
    )

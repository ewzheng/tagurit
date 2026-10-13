"""Tests for the orchestrator: stamping, tagging off the event loop, and the client hand-off."""

import asyncio
import time
from collections.abc import AsyncIterable, AsyncIterator, Iterable

import pytest

from tagurit import orchestrator
from tagurit.protocol import ImageFrame


class LengthTagger:
    """Priority from the image length, so tests can tell frames apart."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay

    def tag(self, frame_id: int, timestamp: float, image_bytes: bytes) -> ImageFrame:
        time.sleep(self.delay)  # blocking on purpose, like real model inference
        return ImageFrame(frame_id, timestamp, image_bytes, len(image_bytes) / 10)


async def images(items: Iterable[bytes]) -> AsyncIterator[bytes]:
    for item in items:
        yield item


async def collect(frames: AsyncIterable[ImageFrame]) -> list[ImageFrame]:
    return [frame async for frame in frames]


def test_frames_are_numbered_stamped_and_tagged_in_order() -> None:
    clock = iter([100.0, 101.5, 103.0])
    frames = asyncio.run(
        collect(
            orchestrator.tag_frames(
                images([b"a", b"bbb", b"cc"]), LengthTagger(), clock=lambda: next(clock)
            )
        )
    )
    assert [f.frame_id for f in frames] == [1, 2, 3]
    assert [f.timestamp for f in frames] == [100.0, 101.5, 103.0]
    assert [f.image_bytes for f in frames] == [b"a", b"bbb", b"cc"]
    assert [f.priority for f in frames] == [0.1, 0.3, 0.2]


def test_slow_tagging_does_not_block_the_event_loop() -> None:
    async def scenario() -> int:
        ticks = 0
        done = asyncio.Event()

        async def ticker() -> None:
            nonlocal ticks
            while not done.is_set():
                ticks += 1
                await asyncio.sleep(0.01)

        task = asyncio.create_task(ticker())
        await collect(orchestrator.tag_frames(images([b"x", b"y"]), LengthTagger(delay=0.15)))
        done.set()
        await task
        return ticks

    # 0.3 s of tagging; a blocked loop would allow about one tick.
    assert asyncio.run(scenario()) >= 10


def test_tagger_errors_propagate() -> None:
    class Broken:
        def tag(self, frame_id: int, timestamp: float, image_bytes: bytes) -> ImageFrame:
            raise ValueError("bad image")

    with pytest.raises(ValueError, match="bad image"):
        asyncio.run(collect(orchestrator.tag_frames(images([b"x"]), Broken())))


def test_run_hands_tagged_frames_to_the_client(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[ImageFrame] = []
    reports: list[object] = []

    async def fake_run_client(frames: AsyncIterable[ImageFrame], report: object) -> None:
        received.extend([frame async for frame in frames])
        reports.append(report)

    def report(event: str, detail: str, scheduler: object) -> None:
        pass

    monkeypatch.setattr(orchestrator, "run_client", fake_run_client)
    asyncio.run(orchestrator.run(images([b"aa", b"b"]), LengthTagger(), report))
    assert [(f.frame_id, f.priority) for f in received] == [(1, 0.2), (2, 0.1)]
    assert reports == [report]

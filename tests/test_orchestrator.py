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
                images([b"a", b"bbb", b"cc"]),
                LengthTagger(),
                clock=lambda: next(clock),
                drop_stale=False,
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
        await collect(
            orchestrator.tag_frames(
                images([b"x", b"y"]), LengthTagger(delay=0.15), drop_stale=False
            )
        )
        done.set()
        await task
        return ticks

    # 0.3 s of tagging; a blocked loop would allow about one tick.
    assert asyncio.run(scenario()) >= 10


async def paced(items: Iterable[bytes], interval: float) -> AsyncIterator[bytes]:
    """Release one image every ``interval`` seconds whatever the consumer does, like a camera."""
    for item in items:
        yield item
        await asyncio.sleep(interval)


def test_images_arriving_while_the_tagger_is_busy_are_dropped_for_the_newest() -> None:
    dropped: list[int] = []
    # Five images 0.02 s apart against a 0.07 s tagger: 2 to 4 arrive while 1 is tagged.
    frames = asyncio.run(
        collect(
            orchestrator.tag_frames(
                paced([b"a", b"bb", b"ccc", b"dddd", b"eeeee"], 0.02),
                LengthTagger(delay=0.07),
                dropped=dropped.append,
            )
        )
    )
    ids = [f.frame_id for f in frames]
    assert ids[0] == 1 and ids[-1] == 5  # the first is tagged at once, the last is never stale
    assert ids == sorted(ids) and len(ids) < 5
    assert sorted(ids + dropped) == [1, 2, 3, 4, 5]
    assert all(len(f.image_bytes) == f.frame_id for f in frames)  # bytes stay with their ID


def test_a_tagger_that_keeps_up_drops_nothing() -> None:
    dropped: list[int] = []
    frames = asyncio.run(
        collect(
            orchestrator.tag_frames(
                paced([b"a", b"b", b"c"], 0.03), LengthTagger(), dropped=dropped.append
            )
        )
    )
    assert [f.frame_id for f in frames] == [1, 2, 3]
    assert dropped == []


def test_min_interval_drops_images_captured_too_soon_after_the_last_tagged() -> None:
    clock = iter([0.0, 0.4, 0.9, 1.0, 1.3, 2.5])
    dropped: list[int] = []
    frames = asyncio.run(
        collect(
            orchestrator.tag_frames(
                paced([b"x"] * 6, 0.01),
                LengthTagger(),
                clock=lambda: next(clock),
                min_interval=1.0,
                dropped=dropped.append,
            )
        )
    )
    assert [f.timestamp for f in frames] == [0.0, 1.0, 2.5]
    assert dropped == [2, 3, 5]


def test_min_interval_is_checked() -> None:
    with pytest.raises(ValueError):
        asyncio.run(
            collect(orchestrator.tag_frames(images([b"x"]), LengthTagger(), min_interval=-1))
        )
    with pytest.raises(ValueError):
        asyncio.run(
            collect(
                orchestrator.tag_frames(
                    images([b"x"]), LengthTagger(), drop_stale=False, min_interval=1.0
                )
            )
        )


def test_source_errors_propagate_after_the_waiting_image_is_tagged() -> None:
    async def failing() -> AsyncIterator[bytes]:
        yield b"x"
        raise OSError("camera unplugged")

    tagged: list[ImageFrame] = []

    async def scenario() -> None:
        async for frame in orchestrator.tag_frames(failing(), LengthTagger()):
            tagged.append(frame)

    with pytest.raises(OSError, match="camera unplugged"):
        asyncio.run(scenario())
    assert [f.frame_id for f in tagged] == [1]


def test_tagger_errors_propagate() -> None:
    class Broken:
        def tag(self, frame_id: int, timestamp: float, image_bytes: bytes) -> ImageFrame:
            raise ValueError("bad image")

    with pytest.raises(ValueError, match="bad image"):
        asyncio.run(collect(orchestrator.tag_frames(images([b"x"]), Broken())))


def test_run_hands_tagged_frames_to_the_client(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[ImageFrame] = []
    reports: list[object] = []

    async def fake_run_client(
        frames: AsyncIterable[ImageFrame],
        report: object,
        send_interval: float,
        bank_order: object,
        rescorers: object,
    ) -> None:
        received.extend([frame async for frame in frames])
        reports.append((report, send_interval, bank_order, rescorers))

    def report(event: str, detail: str, scheduler: object) -> None:
        pass

    monkeypatch.setattr(orchestrator, "run_client", fake_run_client)

    def order(timestamps: object, priorities: object, sent: object) -> int | None:
        return None

    asyncio.run(
        orchestrator.run(
            images([b"aa", b"b"]), LengthTagger(), report, 0.25, drop_stale=False, bank_order=order
        )
    )
    assert [(f.frame_id, f.priority) for f in received] == [(1, 0.2), (2, 0.1)]
    assert reports == [(report, 0.25, order, None)]


class GainTagger(LengthTagger):
    """A tagger whose query is a gain on the image length, and which can rescore its frames."""

    def __init__(self) -> None:
        super().__init__()
        self.gain = 1
        self.calls: list[str] = []

    def tag(self, frame_id: int, timestamp: float, image_bytes: bytes) -> ImageFrame:
        self.calls.append("tag")
        return ImageFrame(frame_id, timestamp, image_bytes, self.gain * len(image_bytes) / 10)

    def set_targets(self, phrases: object) -> None:
        self.calls.append("set_targets")
        self.gain = len(phrases)  # type: ignore[arg-type]

    def rescore(self, frames: list[ImageFrame]) -> dict[int, float]:
        self.calls.append("rescore")
        return {f.frame_id: self.gain * len(f.image_bytes) / 10 for f in frames}


def test_each_mission_switches_the_query_then_yields_a_rescorer() -> None:
    tagger = GainTagger()
    old = [tagger.tag(1, 0.0, b"aa"), tagger.tag(2, 0.0, b"b")]

    async def scenario() -> list[dict[int, float]]:
        results = []
        async for rescorer in orchestrator.retarget(images([("x", "y"), ("x", "y", "z")]), tagger):
            results.append(dict(await rescorer(old)))
        return results

    doubled, tripled = asyncio.run(scenario())
    assert doubled == pytest.approx({1: 0.4, 2: 0.2})
    assert tripled == pytest.approx({1: 0.6, 2: 0.3})
    assert tagger.calls == ["tag", "tag", "set_targets", "rescore", "set_targets", "rescore"]


def test_run_passes_rescorers_only_for_a_tagger_that_can_change_its_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[int, float]] = []

    async def fake_run_client(
        frames: AsyncIterable[ImageFrame],
        report: object,
        send_interval: float,
        bank_order: object,
        rescorers: AsyncIterable,
    ) -> None:
        tagged = [frame async for frame in frames]
        async for rescorer in rescorers:
            seen.append(dict(await rescorer(tagged)))

    monkeypatch.setattr(orchestrator, "run_client", fake_run_client)
    asyncio.run(
        orchestrator.run(
            images([b"aa", b"b"]),
            GainTagger(),
            lambda event, detail, scheduler: None,
            drop_stale=False,
            missions=images([("x", "y", "z")]),
        )
    )
    assert seen == [pytest.approx({1: 0.6, 2: 0.3})]

    with pytest.raises(TypeError, match="cannot change its query"):
        asyncio.run(
            orchestrator.run(
                images([b"aa"]),
                LengthTagger(),
                lambda event, detail, scheduler: None,
                missions=images([("x",)]),
            )
        )

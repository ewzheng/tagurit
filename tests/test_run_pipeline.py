"""Tests for the end-to-end demo's frame selection, replay, and priority recording."""

import asyncio
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pytest

from tagurit.sim import run_pipeline
from tagurit.tagging import Tagger
from tagurit.tagging.priority import PriorityScale

DATASET = "sparse-seadronessee"


def write_sequence(root: Path, name: str, targets: list[int]) -> None:
    folder = root / "data" / DATASET / name
    folder.mkdir(parents=True)
    rows = ["file,target"]
    for i, target in enumerate(targets):
        (folder / f"{i}.jpg").write_bytes(f"{name}-{i}".encode())
        rows.append(f"{i}.jpg,{target}")
    (folder / "frames.csv").write_text("\n".join(rows) + "\n")


@pytest.fixture
def dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_sequence(tmp_path, "a", [0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    write_sequence(tmp_path, "b", [0, 1, 0, 0])
    monkeypatch.chdir(tmp_path)


RECORD = {"dataset": DATASET, "sequences": ["a", "b"], "fit_fraction": 0.5}


@pytest.mark.usefixtures("dataset")
def test_held_out_frames_follow_the_fit_record() -> None:
    frames = run_pipeline.select_frames(RECORD, None, None, all_frames=False, empty_stride=1)
    assert [(f.sequence, f.index) for f in frames] == [
        ("a", 6), ("a", 7), ("a", 8), ("a", 9), ("a", 10), ("b", 3), ("b", 4),
    ]  # fmt: skip


@pytest.mark.usefixtures("dataset")
def test_empty_stride_thins_empty_frames_and_keeps_every_target() -> None:
    frames = run_pipeline.select_frames(RECORD, None, None, all_frames=True, empty_stride=3)
    assert [f.has_target() for f in frames].count(True) == 2
    assert len(frames) == 2 + 4  # 12 empty frames, every third kept


def test_a_dataset_is_required_without_a_record() -> None:
    with pytest.raises(ValueError):
        run_pipeline.select_frames(None, None, None, all_frames=False, empty_stride=1)


@pytest.mark.usefixtures("dataset")
def test_replay_yields_each_frames_bytes_in_order() -> None:
    frames = run_pipeline.select_frames(RECORD, None, ["b"], all_frames=True, empty_stride=1)

    async def drain() -> list[bytes]:
        return [image async for image in run_pipeline.replay(frames, interval=0.0)]

    assert asyncio.run(drain()) == [b"b-0", b"b-1", b"b-2", b"b-3"]


def test_recorder_keeps_priority_thumbnail_and_best_tile() -> None:
    class Brightness:
        def score_tiles(self, tiles: Sequence[np.ndarray]) -> list[float]:
            return [float(tile.mean()) for tile in tiles]

    image = np.zeros((64, 128, 3), dtype=np.uint8)
    image[0:64, 64:128] = 200  # the right-hand tile is the bright one
    ok, png = cv2.imencode(".png", image)
    assert ok
    tagger = Tagger(Brightness(), PriorityScale(reference=100.0), tile_size=64)
    recorder = run_pipeline.Recorder(tagger, targets={3: True})

    frame = recorder.tag(3, 12.5, png.tobytes())
    assert (frame.frame_id, frame.timestamp, frame.image_bytes) == (3, 12.5, png.tobytes())
    assert frame.priority == pytest.approx(200 / 300)
    view = recorder.views[3]
    assert view.target is True and view.priority == frame.priority
    assert view.best_tile == (0.5, 0.0, 0.5, 1.0)
    assert recorder.priorities == {3: frame.priority}
    assert recorder.latest is not None and recorder.latest[0] == 3


def test_a_failing_dashboard_stops_the_pipeline_and_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    cancelled = []

    async def endless_pipeline(images: object, tagger: object, report: object) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    class BrokenVideo:
        def write(self, image: np.ndarray) -> None:
            raise RuntimeError("no display")

    monkeypatch.setattr(run_pipeline.orchestrator, "run", endless_pipeline)
    recorder = run_pipeline.Recorder(object(), targets={})  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="no display"):
        asyncio.run(run_pipeline.run_demo([], 0.0, recorder, {}, False, BrokenVideo()))  # type: ignore[arg-type]
    assert cancelled == [True]

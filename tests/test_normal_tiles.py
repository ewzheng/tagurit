"""Tests for drawing target-free tiles from annotated frames."""

import random
from pathlib import Path

import cv2
import numpy as np
import pytest

from tagurit.sim.normal_tiles import normal_tiles, sample_normal_tiles
from tagurit.sim.trace import Box, TraceFrame

BRIGHT = 250


def frame_with_box(tmp_path: Path, name: str, target: bool | None = None) -> TraceFrame:
    """A 96x64 dark frame with a bright box in its top-left 32px tile."""
    image = np.full((64, 96, 3), 20, dtype=np.uint8)
    image[4:12, 4:12] = BRIGHT
    path = tmp_path / f"{name}.png"
    cv2.imwrite(str(path), image)
    box = Box("person", "person", 4, 4, 8, 8, None, None, None)
    return TraceFrame("seq", 1, 0.0, path, (box,), target)


def test_normal_tiles_avoid_boxes(tmp_path: Path) -> None:
    frame = frame_with_box(tmp_path, "a")
    found = normal_tiles(frame, tile_size=32, margin=0, limit=10, rng=random.Random(0))
    assert len(found) == 5  # six tiles, minus the one with the box
    assert all(tile.max() < BRIGHT for tile in found)


def test_margin_keeps_neighbours_out(tmp_path: Path) -> None:
    frame = frame_with_box(tmp_path, "a")
    # The box ends at x=y=12: a 15px margin stops short of the next tiles at 32,
    # a 25px margin reaches the tiles right, below, and diagonal.
    assert len(normal_tiles(frame, 32, margin=15, limit=10, rng=random.Random(0))) == 5
    assert len(normal_tiles(frame, 32, margin=25, limit=10, rng=random.Random(0))) == 2


def test_target_frames_without_boxes_give_nothing(tmp_path: Path) -> None:
    frame = frame_with_box(tmp_path, "a")
    unplaced = TraceFrame("seq", 1, 0.0, frame.path, (), target=True)
    assert normal_tiles(unplaced, 32, 0, 10, random.Random(0)) == []


def test_sets_fill_in_order_from_disjoint_frames(tmp_path: Path) -> None:
    frames = [frame_with_box(tmp_path, str(i)) for i in range(4)]
    first, second = sample_normal_tiles(frames, [4, 2], 32, 0, tiles_per_frame=2, seed=0)
    assert (len(first), len(second)) == (4, 2)


def test_running_out_of_frames_raises(tmp_path: Path) -> None:
    frames = [frame_with_box(tmp_path, "a")]
    with pytest.raises(ValueError):
        sample_normal_tiles(frames, [3, 3], 32, 0, tiles_per_frame=2, seed=0)

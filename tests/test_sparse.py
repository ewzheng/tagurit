"""Tests for choosing crop windows when building synthetic sparse streams."""

import random

from tagurit.sim.sparse import (
    Window,
    boxes_in_window,
    empty_window,
    target_window,
    thin_targets,
)
from tagurit.sim.trace import Box


def box(left: int, top: int, width: int, height: int, label: str = "person") -> Box:
    return Box(label, label, left, top, width, height, None, None, None)


def inside(window: Window, b: Box) -> bool:
    return (
        window.left <= b.left
        and window.top <= b.top
        and b.left + b.width <= window.left + window.width
        and b.top + b.height <= window.top + window.height
    )


def test_target_window_holds_the_target_and_stays_in_frame() -> None:
    boxes = [box(950, 10, 40, 30)]
    for seed in range(50):
        window = target_window(boxes, 1000, 500, 200, 100, random.Random(seed))
        assert window is not None and inside(window, boxes[0])
        assert window.left + window.width <= 1000 and window.top + window.height <= 500


def test_target_window_skips_dont_care_oversized_and_out_of_frame_boxes() -> None:
    rng = random.Random(0)
    assert target_window([box(10, 10, 20, 20, "ignored")], 1000, 500, 200, 100, rng) is None
    assert target_window([box(10, 10, 300, 20)], 1000, 500, 200, 100, rng) is None
    assert target_window([box(990, 10, 20, 20)], 1000, 500, 200, 100, rng) is None
    assert target_window([box(10, 10, 20, 20)], 100, 500, 200, 100, rng) is None


def clear(window: Window, b: Box, margin: int) -> bool:
    return (
        b.left + b.width + margin <= window.left
        or window.left + window.width <= b.left - margin
        or b.top + b.height + margin <= window.top
        or window.top + window.height <= b.top - margin
    )


def test_empty_window_keeps_clear_of_every_box() -> None:
    boxes = [box(100, 100, 20, 20), box(600, 300, 50, 50, "ignored")]
    for seed in range(50):
        window = empty_window(boxes, 1000, 500, 200, 100, 30, random.Random(seed))
        assert window is not None
        assert all(clear(window, b, 30) for b in boxes)


def test_empty_window_gives_up_on_a_full_frame() -> None:
    assert empty_window([box(0, 0, 1000, 500)], 1000, 500, 200, 100, 0, random.Random(0)) is None


def test_boxes_are_clipped_and_moved_into_the_window() -> None:
    window = Window(left=100, top=50, width=200, height=100)
    moved = boxes_in_window(
        [box(90, 60, 30, 20), box(150, 70, 10, 10), box(400, 400, 5, 5)], window
    )
    assert [(b.left, b.top, b.width, b.height) for b in moved] == [
        (0, 10, 20, 20),
        (50, 20, 10, 10),
    ]


def test_thinning_caps_the_target_share_and_keeps_order() -> None:
    flags = [False] * 99 + [True] * 30
    kept = thin_targets(flags, 0.01, random.Random(0))
    assert kept == sorted(kept)
    assert sum(flags[i] for i in kept) == 1  # round(0.01 * 99 / 0.99)
    assert sum(not flags[i] for i in kept) == 99


def test_thinning_keeps_everything_under_the_rate() -> None:
    flags = [False] * 99 + [True]
    assert thin_targets(flags, 0.05, random.Random(0)) == list(range(100))
    assert thin_targets([True, True], 1.0, random.Random(0)) == [0, 1]
    assert thin_targets([True, True], 0.0, random.Random(0)) == []

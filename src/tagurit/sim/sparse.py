"""
Choose crop windows of annotated frames for synthetic sparse streams.

Real aerial datasets film their targets on purpose, so almost every frame
holds one. Fixed-size windows cut from those frames give genuinely empty
frames alongside a controlled few that hold a whole target, all from real
pixels. Pasting targets into empty frames is avoided on purpose: the seams
would be easy anomalies and flatter any detector. This module only picks
windows and moves boxes into window coordinates; ``scripts/make_sparse.py``
does the file work.
"""

from __future__ import annotations

import random
from collections.abc import Collection, Sequence
from dataclasses import dataclass, replace

from tagurit.sim.trace import Box


@dataclass(frozen=True)
class Window:
    """
    A crop rectangle in frame pixels, in the ``trace.Box`` xywh convention.

    Parameters:
        - left (int): x of the left edge
        - top (int): y of the top edge
        - width (int): window width
        - height (int): window height
    """

    left: int
    top: int
    width: int
    height: int


def target_window(
    boxes: Sequence[Box],
    frame_width: int,
    frame_height: int,
    width: int,
    height: int,
    rng: random.Random,
    dont_care: Collection[str] = ("ignored",),
) -> Window | None:
    """
    A window inside the frame that wholly contains one randomly chosen target.

    The target is drawn from boxes that are not don't-care, lie inside the
    frame, and fit in the window. The window is placed uniformly among the
    positions that keep the target inside, so targets do not always sit in
    the centre.

    Parameters:
        - boxes (Sequence[Box]): ground truth on the frame
        - frame_width (int): frame width, pixels
        - frame_height (int): frame height, pixels
        - width (int): window width, pixels
        - height (int): window height, pixels
        - rng (random.Random): source of the random choices
        - dont_care (Collection[str]): labels that are never targets

    Return: the window, or None when no target fits
    """
    if width > frame_width or height > frame_height:
        return None
    fitting = [
        box
        for box in boxes
        if box.label not in dont_care
        and box.width <= width
        and box.height <= height
        and box.left >= 0
        and box.top >= 0
        and box.left + box.width <= frame_width
        and box.top + box.height <= frame_height
    ]
    if not fitting:
        return None
    box = rng.choice(fitting)
    left = rng.randint(max(0, box.left + box.width - width), min(box.left, frame_width - width))
    top = rng.randint(max(0, box.top + box.height - height), min(box.top, frame_height - height))
    return Window(left=left, top=top, width=width, height=height)


def empty_window(
    boxes: Sequence[Box],
    frame_width: int,
    frame_height: int,
    width: int,
    height: int,
    margin: int,
    rng: random.Random,
    attempts: int = 50,
) -> Window | None:
    """
    A window inside the frame that no box, of any class, comes within ``margin`` of.

    Tries uniformly random positions and gives up after ``attempts``, so a
    crowded frame may yield None even when a narrow gap exists.

    Parameters:
        - boxes (Sequence[Box]): ground truth on the frame
        - frame_width (int): frame width, pixels
        - frame_height (int): frame height, pixels
        - width (int): window width, pixels
        - height (int): window height, pixels
        - margin (int): clearance kept around every box, pixels
        - rng (random.Random): source of the random positions
        - attempts (int): positions to try before giving up

    Return: the window, or None when none was found
    """
    if width > frame_width or height > frame_height:
        return None
    for _ in range(attempts):
        window = Window(
            left=rng.randint(0, frame_width - width),
            top=rng.randint(0, frame_height - height),
            width=width,
            height=height,
        )
        if not any(_intersects(window, box, margin) for box in boxes):
            return window
    return None


def thin_targets(is_target: Sequence[bool], rate: float, rng: random.Random) -> list[int]:
    """
    Indices to keep so targets make up at most ``rate`` of what is kept.

    Only targets are dropped, chosen at random, and the kept indices stay in
    input order. The allowed number of targets is ``rate / (1 - rate)``
    times the number of non-targets, rounded to the nearest whole number. A
    rate outside [0, 1] RAISES ValueError.

    Parameters:
        - is_target (Sequence[bool]): one flag per item
        - rate (float): largest target share to keep
        - rng (random.Random): chooses which targets to drop

    Return: ascending indices of the items to keep
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be between 0 and 1, got {rate}")
    targets = [i for i, flag in enumerate(is_target) if flag]
    empties = len(is_target) - len(targets)
    allowed = len(targets) if rate == 1.0 else round(rate * empties / (1.0 - rate))
    dropped = set(rng.sample(targets, max(0, len(targets) - allowed)))
    return [i for i in range(len(is_target)) if i not in dropped]


def boxes_in_window(boxes: Sequence[Box], window: Window) -> tuple[Box, ...]:
    """
    The boxes that overlap a window, clipped to it and moved into its coordinates.

    Parameters:
        - boxes (Sequence[Box]): ground truth in frame coordinates
        - window (Window): the crop

    Return: boxes in window coordinates, in input order
    """
    moved = []
    for box in boxes:
        if not _intersects(window, box, 0):
            continue
        left = max(box.left, window.left)
        top = max(box.top, window.top)
        right = min(box.left + box.width, window.left + window.width)
        bottom = min(box.top + box.height, window.top + window.height)
        moved.append(
            replace(
                box,
                left=left - window.left,
                top=top - window.top,
                width=right - left,
                height=bottom - top,
            )
        )
    return tuple(moved)


def _intersects(window: Window, box: Box, margin: int) -> bool:
    """
    Whether a box grown by ``margin`` on every side shares a pixel with the window.

    Parameters:
        - window (Window): the crop
        - box (Box): one ground-truth box
        - margin (int): pixels added to each side of the box

    Return: True when they intersect
    """
    return (
        box.left - margin < window.left + window.width
        and window.left < box.left + box.width + margin
        and box.top - margin < window.top + window.height
        and window.top < box.top + box.height + margin
    )

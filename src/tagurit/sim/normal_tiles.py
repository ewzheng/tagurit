"""
Draw target-free tiles from annotated frames to fit and calibrate taggers.

A tile is normal when no ground-truth box of any class comes within a
margin of it. PatchCore fits its bank on such tiles and every scorer
calibrates its priority reference on them, so a target slipping in here
corrupts both. Frames labelled as holding a target but with no boxes to
place it supply nothing, since any of their tiles could hold it.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np

from tagurit.model.tiling import crop_tiles, decode_image, tile_grid
from tagurit.sim.trace import TraceFrame


def normal_tiles(
    frame: TraceFrame, tile_size: int, margin: int, limit: int, rng: random.Random
) -> list[np.ndarray]:
    """
    Draw up to ``limit`` tiles from one frame that no box comes near.

    Crops are copied so the decoded frame can be freed.

    Parameters:
        - frame (TraceFrame): frame with its ground-truth boxes
        - tile_size (int): tile side length, pixels
        - margin (int): pixels a box is grown by before testing overlap
        - limit (int): most tiles to take from this frame
        - rng (random.Random): chooses among the free tiles

    Return: copied uint8 RGB tiles, possibly none
    """
    if limit <= 0 or (frame.target and not frame.boxes):
        return []
    image = decode_image(frame.read())
    height, width = image.shape[:2]
    free = [
        tile
        for tile in tile_grid(width, height, tile_size)
        if not any(tile.overlaps(b.left, b.top, b.width, b.height, margin) for b in frame.boxes)
    ]
    chosen = rng.sample(free, min(limit, len(free)))
    return [crop.copy() for crop in crop_tiles(image, chosen)]


def sample_normal_tiles(
    frames: Sequence[TraceFrame],
    sizes: Sequence[int],
    tile_size: int,
    margin: int,
    tiles_per_frame: int,
    seed: int,
) -> list[list[np.ndarray]]:
    """
    Fill several sets of normal tiles in turn, each from frames the others never used.

    Frames are taken in the given order, so shuffle them first for variety.
    Each frame gives at most ``tiles_per_frame`` tiles, to the first set
    that is not yet full. Running out of frames before every set is full
    RAISES ValueError naming how many tiles were found, rather than
    returning fewer than asked.

    Parameters:
        - frames (Sequence[TraceFrame]): candidate frames
        - sizes (Sequence[int]): how many tiles each set needs, in fill order
        - tile_size (int): tile side length, pixels
        - margin (int): pixels kept clear around every box
        - tiles_per_frame (int): most tiles taken from one frame
        - seed (int): seeds the choice of tiles within each frame

    Return: one list of tiles per entry of ``sizes``
    """
    rng = random.Random(seed)
    sets: list[list[np.ndarray]] = [[] for _ in sizes]
    for frame in frames:
        open_sets = [i for i, size in enumerate(sizes) if len(sets[i]) < size]
        if not open_sets:
            return sets
        current = open_sets[0]
        limit = min(tiles_per_frame, sizes[current] - len(sets[current]))
        sets[current].extend(normal_tiles(frame, tile_size, margin, limit, rng))
    if all(len(found) == size for found, size in zip(sets, sizes, strict=True)):
        return sets
    found = ", ".join(f"{len(s)} of {size}" for s, size in zip(sets, sizes, strict=True))
    raise ValueError(f"found only {found} normal tiles in {len(frames)} frames")

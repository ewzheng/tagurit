"""
Decode frames and cut them into a grid of square tiles for per-tile scoring.

Aerial targets are a few dozen pixels across. Resizing a whole 4K frame to
a model's input size would shrink a swimmer below a single feature patch,
so models score tiles and tagging aggregates them. This module is numpy and
OpenCV only and NEVER imports torch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Tile:
    """
    One square tile's position in its frame.

    Absolute pixels with the origin at the frame's top-left corner, the same
    convention as ``sim.trace.Box``. The tile covers columns ``left`` up to
    but NOT including ``left + size``, and rows likewise from ``top``.

    Parameters:
        - left (int): x of the left edge, pixels
        - top (int): y of the top edge, pixels
        - size (int): side length, pixels
    """

    left: int
    top: int
    size: int

    def overlaps(self, left: int, top: int, width: int, height: int, margin: int = 0) -> bool:
        """
        Whether a box, grown by ``margin`` pixels on every side, shares a pixel with this tile.

        Boxes use the ``sim.trace.Box`` xywh convention. Edges that only touch
        do not overlap. A negative margin RAISES ValueError.

        Parameters:
            - left (int): x of the box's left edge, pixels
            - top (int): y of the box's top edge, pixels
            - width (int): box width, pixels
            - height (int): box height, pixels
            - margin (int): pixels added to each side of the box before testing

        Return: True when the grown box and the tile intersect
        """
        if margin < 0:
            raise ValueError(f"margin must be non-negative, got {margin}")
        return (
            left - margin < self.left + self.size
            and self.left < left + width + margin
            and top - margin < self.top + self.size
            and self.top < top + height + margin
        )


def tile_grid(width: int, height: int, size: int) -> tuple[Tile, ...]:
    """
    Cover a frame with the fewest square tiles of side ``size``.

    Tiles step by ``size`` from the top-left corner. When a dimension is not a
    multiple of ``size``, the last row or column shifts back to end exactly on
    the frame edge, overlapping its neighbour rather than hanging off the
    frame. Every tile is full size and lies inside the frame. A frame smaller
    than one tile in either dimension RAISES ValueError, as does a
    non-positive size.

    Parameters:
        - width (int): frame width, pixels
        - height (int): frame height, pixels
        - size (int): tile side length, pixels

    Return: tiles in row-major order, top row first
    """
    if size <= 0:
        raise ValueError(f"tile size must be positive, got {size}")
    if width < size or height < size:
        raise ValueError(f"a {width}x{height} frame is smaller than one {size}px tile")
    return tuple(
        Tile(left=left, top=top, size=size)
        for top in _starts(height, size)
        for left in _starts(width, size)
    )


def decode_image(image_bytes: bytes) -> np.ndarray:
    """
    Decode an encoded image (JPEG, PNG, anything OpenCV reads) into RGB.

    OpenCV decodes to BGR. This returns RGB because ImageNet backbones expect
    it. Empty or undecodable bytes RAISE ValueError.

    Parameters:
        - image_bytes (bytes): the encoded file contents

    Return: uint8 array of shape (height, width, 3) in RGB order
    """
    if not image_bytes:
        raise ValueError("image bytes are empty")
    bgr = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("image bytes could not be decoded")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def crop_tiles(image: np.ndarray, tiles: Sequence[Tile]) -> list[np.ndarray]:
    """
    Cut each tile out of a decoded frame.

    Crops are numpy VIEWS into ``image``, not copies: writing into a crop
    writes into the frame. A tile that does not lie fully inside the frame
    RAISES ValueError rather than returning a short crop.

    Parameters:
        - image (np.ndarray): decoded frame of shape (height, width, channels)
        - tiles (Sequence[Tile]): tiles to cut, e.g. from ``tile_grid``

    Return: one (size, size, channels) array per tile, in the order given
    """
    height, width = image.shape[:2]
    crops = []
    for tile in tiles:
        if tile.left < 0 or tile.top < 0:
            raise ValueError(f"{tile} starts outside the frame")
        if tile.left + tile.size > width or tile.top + tile.size > height:
            raise ValueError(f"{tile} extends past a {width}x{height} frame")
        crops.append(image[tile.top : tile.top + tile.size, tile.left : tile.left + tile.size])
    return crops


def _starts(length: int, size: int) -> list[int]:
    """
    Tile start offsets along one dimension, the last one flush with the edge.

    Parameters:
        - length (int): frame length in this dimension, at least ``size``
        - size (int): tile side length

    Return: ascending start offsets
    """
    starts = list(range(0, length - size + 1, size))
    if starts[-1] + size < length:
        starts.append(length - size)
    return starts

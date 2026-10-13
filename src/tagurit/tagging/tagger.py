"""
Turn an encoded frame into a priority: decode, tile, score, aggregate, map.

This is the call the orchestrator makes for each captured frame before it
builds an ``ImageFrame`` for the client. A frame's raw score is its most
unusual tile, because one swimmer in one tile is the whole reason to keep
the frame. The tagger works with any ``model.scorer.TileScorer`` and never
learns which model is behind it.

Scoring is blocking compute. Called directly from the client's asyncio
loop it would stall Gabriel transport, so call it through
``asyncio.to_thread``.
"""

from __future__ import annotations

from dataclasses import dataclass

from tagurit.model.scorer import TileScorer
from tagurit.model.tiling import Tile, crop_tiles, decode_image, tile_grid
from tagurit.protocol import ImageFrame
from tagurit.tagging.priority import PriorityScale


@dataclass(frozen=True)
class FrameScore:
    """
    The tagger's verdict on one frame, with the per-tile detail behind it.

    Parameters:
        - priority (float): frame priority in [0, 1]
        - raw_score (float): highest raw tile score, the value that was mapped
        - tiles (tuple[Tile, ...]): the tile grid, row-major
        - tile_scores (tuple[float, ...]): raw score per tile, same order as ``tiles``
    """

    priority: float
    raw_score: float
    tiles: tuple[Tile, ...]
    tile_scores: tuple[float, ...]


class Tagger:
    """
    Score frames by their most unusual tile and map that onto a priority.

    Holds no per-frame state; calls are independent. A non-positive tile
    size RAISES ValueError.

    Parameters:
        - scorer (TileScorer): the model that scores tiles
        - scale (PriorityScale): maps the frame's raw score to a priority
        - tile_size (int): tile side length in frame pixels
    """

    def __init__(self, scorer: TileScorer, scale: PriorityScale, tile_size: int) -> None:
        if tile_size <= 0:
            raise ValueError(f"tile_size must be positive, got {tile_size}")
        self.scorer = scorer
        self.scale = scale
        self.tile_size = tile_size

    def score(self, image_bytes: bytes) -> FrameScore:
        """
        Score one encoded frame.

        Undecodable bytes, or a frame smaller than one tile, RAISE ValueError.

        Parameters:
            - image_bytes (bytes): encoded image, e.g. JPEG

        Return: the frame's priority, raw score, and per-tile scores
        """
        image = decode_image(image_bytes)
        height, width = image.shape[:2]
        tiles = tile_grid(width, height, self.tile_size)
        tile_scores = tuple(self.scorer.score_tiles(crop_tiles(image, tiles)))
        if len(tile_scores) != len(tiles):
            raise ValueError(f"scorer returned {len(tile_scores)} scores for {len(tiles)} tiles")
        raw_score = max(tile_scores)
        return FrameScore(
            priority=self.scale.priority(raw_score),
            raw_score=raw_score,
            tiles=tiles,
            tile_scores=tile_scores,
        )

    def tag(self, frame_id: int, timestamp: float, image_bytes: bytes) -> ImageFrame:
        """
        Score one encoded frame and package it for the client.

        The image bytes pass through unchanged.

        Parameters:
            - frame_id (int): unique image number within the session
            - timestamp (float): capture time in seconds since the Unix epoch
            - image_bytes (bytes): encoded image, e.g. JPEG

        Return: an ImageFrame carrying the frame's priority
        """
        return ImageFrame(
            frame_id=frame_id,
            timestamp=timestamp,
            image_bytes=image_bytes,
            priority=self.score(image_bytes).priority,
        )

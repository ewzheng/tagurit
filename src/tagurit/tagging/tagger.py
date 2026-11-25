"""
Turn an encoded frame into a priority: decode, tile, score, aggregate, map.

This is the call the orchestrator makes for each captured frame before it
builds an ``ImageFrame`` for the client. A frame's raw score is its most
unusual tile, because one swimmer in one tile is the whole reason to keep
the frame. The tagger works with any ``model.scorer.TileScorer`` and never
learns which model is behind it. A ``model.scorer.WindowedTileScorer`` also
gets each frame's capture time, since its scores depend on the frames just
before; such a tagger must be fed frames in capture order. With a
``model.scorer.RetargetableTileScorer`` each tagged frame also carries what
the scorer kept of it, so ``set_targets`` can change the query mid-mission
and ``rescore`` can give frames still waiting in the client a priority
under it, without their images.

Scoring is blocking compute. Called directly from the client's asyncio
loop it would stall Gabriel transport, so call it through
``asyncio.to_thread``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tagurit.model.scorer import RetargetableTileScorer, TileScorer, WindowedTileScorer
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
        - kept (object | None): what a retargetable scorer kept to rescore the frame, else None
    """

    priority: float
    raw_score: float
    tiles: tuple[Tile, ...]
    tile_scores: tuple[float, ...]
    kept: object | None = None


class Tagger:
    """
    Score frames by their most unusual tile and map that onto a priority.

    Holds no per-frame state itself. With a ``WindowedTileScorer`` and a
    timestamp, a frame's scores depend on the frames scored before it;
    without a timestamp, or with a plain ``TileScorer``, calls are
    independent. A non-positive tile size, or a detection threshold outside
    [0, 1], RAISES ValueError.

    Parameters:
        - scorer (TileScorer): the model that scores tiles
        - scale (PriorityScale): maps the frame's raw score to a priority
        - tile_size (int): tile side length in frame pixels
        - detection (float | None): priority at or above which a frame counts as a
          detection for ``tagging.order``; None when the bundle was not calibrated for it
    """

    def __init__(
        self,
        scorer: TileScorer,
        scale: PriorityScale,
        tile_size: int,
        detection: float | None = None,
    ) -> None:
        if tile_size <= 0:
            raise ValueError(f"tile_size must be positive, got {tile_size}")
        if detection is not None and not 0.0 <= detection <= 1.0:
            raise ValueError(f"detection threshold must be in [0, 1], got {detection}")
        self.scorer = scorer
        self.scale = scale
        self.tile_size = tile_size
        self.detection = detection

    def score(self, image_bytes: bytes, timestamp: float | None = None) -> FrameScore:
        """
        Score one encoded frame.

        With a timestamp, a ``WindowedTileScorer`` scores the frame against
        the frames before it and remembers it; otherwise tiles are scored on
        their own. Undecodable bytes, or a frame smaller than one tile,
        RAISE ValueError.

        Parameters:
            - image_bytes (bytes): encoded image, e.g. JPEG
            - timestamp (float | None): capture time in seconds; None scores the frame alone

        Return: the frame's priority, raw score, and per-tile scores
        """
        image = decode_image(image_bytes)
        height, width = image.shape[:2]
        tiles = tile_grid(width, height, self.tile_size)
        crops = crop_tiles(image, tiles)
        kept = None
        if timestamp is not None and isinstance(self.scorer, RetargetableTileScorer):
            scores, kept = self.scorer.score_and_keep(crops, timestamp)
            tile_scores = tuple(scores)
        elif timestamp is not None and isinstance(self.scorer, WindowedTileScorer):
            tile_scores = tuple(self.scorer.score_frame(crops, timestamp))
        else:
            tile_scores = tuple(self.scorer.score_tiles(crops))
        if len(tile_scores) != len(tiles):
            raise ValueError(f"scorer returned {len(tile_scores)} scores for {len(tiles)} tiles")
        raw_score = max(tile_scores)
        return FrameScore(
            priority=self.scale.priority(raw_score),
            raw_score=raw_score,
            tiles=tiles,
            tile_scores=tile_scores,
            kept=kept,
        )

    def tag(self, frame_id: int, timestamp: float, image_bytes: bytes) -> ImageFrame:
        """
        Score one encoded frame and package it for the client.

        The image bytes pass through unchanged.

        Parameters:
            - frame_id (int): unique image number within the session
            - timestamp (float): capture time in seconds since the Unix epoch
            - image_bytes (bytes): encoded image, e.g. JPEG

        Return: an ImageFrame carrying the frame's priority and what the scorer kept of it
        """
        score = self.score(image_bytes, timestamp)
        return ImageFrame(
            frame_id=frame_id,
            timestamp=timestamp,
            image_bytes=image_bytes,
            priority=score.priority,
            kept=score.kept,
        )

    def set_targets(self, phrases: Sequence[str]) -> None:
        """
        Switch the query for every frame tagged from now on.

        The priority scale and detection threshold stay as calibrated. A
        scorer that cannot change its query RAISES TypeError.

        Parameters:
            - phrases (Sequence[str]): the new query phrases

        Return: void
        """
        self._retargetable().set_targets(phrases)

    def rescore(self, frames: Sequence[ImageFrame]) -> dict[int, float]:
        """
        Priorities under the current query for frames tagged earlier.

        Uses what each frame carries in ``kept``; frames without it are left
        out, so they keep the priority they have. A scorer that cannot
        rescore RAISES TypeError.

        Parameters:
            - frames (Sequence[ImageFrame]): frames this tagger tagged

        Return: frame ID to new priority, for the frames that could be rescored
        """
        scorer = self._retargetable()
        known = [frame for frame in frames if frame.kept is not None]
        raw_scores = scorer.rescore([frame.kept for frame in known])
        return {
            frame.frame_id: self.scale.priority(raw)
            for frame, raw in zip(known, raw_scores, strict=True)
        }

    def _retargetable(self) -> RetargetableTileScorer:
        """
        The scorer, if its query can change; otherwise RAISES TypeError.

        Return: the scorer
        """
        if not isinstance(self.scorer, RetargetableTileScorer):
            raise TypeError(f"{type(self.scorer).__name__} cannot change its query")
        return self.scorer

    def reset(self) -> None:
        """
        Make a windowed scorer forget recent frames, as at the start of a new flight.

        Does nothing for a scorer that remembers nothing.

        Return: void
        """
        if isinstance(self.scorer, WindowedTileScorer):
            self.scorer.reset()

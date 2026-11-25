"""
The interfaces every tile-scoring model implements.

Tagging depends on these protocols, never on a concrete model, so a model
can be swapped without touching tagging: PatchCore through anomalib and
CLIP through transformers today, and later a TensorRT engine or a detector.
``TileScorer`` scores tiles on their own; ``WindowedTileScorer`` also
remembers recent frames, so it is given each frame's capture time;
``RetargetableTileScorer`` can also change what it looks for and score
frames it has already seen again without their images. Nothing here
imports torch.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np


class TileScorer(Protocol):
    """
    Score square RGB tiles by how interesting they look.

    Scores are RAW and model-specific: finite, non-negative, higher meaning
    more interesting, on no fixed scale. Turning them into a priority is
    tagging's job (``tagging.priority``), calibrated per fitted model.
    """

    def score_tiles(self, tiles: Sequence[np.ndarray]) -> list[float]:
        """
        Score each tile.

        Parameters:
            - tiles (Sequence[np.ndarray]): uint8 RGB arrays of shape (size, size, 3)

        Return: one raw score per tile, in the order given
        """
        ...


@runtime_checkable
class WindowedTileScorer(TileScorer, Protocol):
    """
    A tile scorer whose scores also depend on the frames it saw shortly before.

    ``score_frame`` scores one frame and remembers it, so frames must be
    given in capture order; ``score_tiles`` still scores tiles on their own.
    """

    def score_frame(self, tiles: Sequence[np.ndarray], timestamp: float) -> list[float]:
        """
        Score one frame's tiles, then remember them.

        Parameters:
            - tiles (Sequence[np.ndarray]): uint8 RGB arrays of shape (size, size, 3)
            - timestamp (float): the frame's capture time in seconds

        Return: one raw score per tile, in the order given
        """
        ...

    def reset(self) -> None:
        """
        Forget every remembered frame, as at the start of a new flight.

        Return: void
        """
        ...


@runtime_checkable
class RetargetableTileScorer(WindowedTileScorer, Protocol):
    """
    A windowed scorer whose query can change, and which can rescore frames it scored before.

    ``score_and_keep`` scores a frame as ``score_frame`` does and also
    returns what it needs to score that frame again later. What is kept is
    the scorer's own business: callers hold it and hand it back, nothing
    more.
    """

    def score_and_keep(
        self, tiles: Sequence[np.ndarray], timestamp: float
    ) -> tuple[list[float], object]:
        """
        Score one frame's tiles, remember them, and return what rescoring the frame needs.

        Parameters:
            - tiles (Sequence[np.ndarray]): uint8 RGB arrays of shape (size, size, 3)
            - timestamp (float): the frame's capture time in seconds

        Return: one raw score per tile in the order given, and the frame's kept state
        """
        ...

    def set_targets(self, phrases: Sequence[str]) -> None:
        """
        Switch what the scorer looks for.

        Parameters:
            - phrases (Sequence[str]): the new query phrases

        Return: void
        """
        ...

    def rescore(self, kept: Sequence[object]) -> list[float]:
        """
        Score frames again under the current query from what ``score_and_keep`` returned.

        Parameters:
            - kept (Sequence[object]): one kept state per frame

        Return: one raw frame score per kept state, in the order given
        """
        ...

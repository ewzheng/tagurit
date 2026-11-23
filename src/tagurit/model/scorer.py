"""
The interfaces every tile-scoring model implements.

Tagging depends on these protocols, never on a concrete model, so a model
can be swapped without touching tagging: PatchCore through anomalib and
CLIP through transformers today, and later a TensorRT engine or a detector.
``TileScorer`` scores tiles on their own; ``WindowedTileScorer`` also
remembers recent frames, so it is given each frame's capture time. Nothing
here imports torch.
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

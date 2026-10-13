"""
The interface every tile-scoring model implements.

Tagging depends on this protocol, never on a concrete model, so a model can
be swapped without touching tagging: PatchCore through anomalib today, and
later a custom PatchCore, a TensorRT engine, or a detector. Nothing here
imports torch.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

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

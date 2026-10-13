"""
Map raw model scores onto the scheduler's priority range.

Raw scores are unbounded and model-specific. The scheduler only needs an
order, with ties broken by arrival, so the mapping must be strictly
increasing and must NEVER saturate: a clamp to 1.0 would turn every strong
detection into a tie. ``s / (s + reference)`` meets both requirements. It
sends 0 to 0 and ``reference`` to 0.5, and approaches 1 without reaching it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PriorityScale:
    """
    Strictly increasing map from raw scores in [0, inf) to priorities in [0, 1).

    ``reference`` is the raw score that maps to 0.5. Calibrated with
    ``from_normal_scores``, 0.5 means "as unusual as the most unusual normal
    tiles", which a later cache inclusion policy can use as a threshold.
    Only floating-point rounding of enormous scores can produce exactly 1.0.
    A reference that is not finite and positive RAISES ValueError.

    Parameters:
        - reference (float): raw score that maps to 0.5
    """

    reference: float

    def __post_init__(self) -> None:
        if not _is_number(self.reference) or not math.isfinite(self.reference):
            raise ValueError(f"reference must be a finite number, got {self.reference!r}")
        if self.reference <= 0:
            raise ValueError(f"reference must be positive, got {self.reference}")

    @classmethod
    def from_normal_scores(cls, scores: Sequence[float], percentile: float = 99.0) -> PriorityScale:
        """
        Calibrate on raw scores of tiles known to contain no target.

        The reference is the given percentile of those scores. No scores, a
        percentile outside (0, 100], or a resulting reference that is not
        positive RAISES ValueError.

        Parameters:
            - scores (Sequence[float]): raw scores of normal tiles
            - percentile (float): which percentile maps to 0.5

        Return: the calibrated scale
        """
        if len(scores) == 0:
            raise ValueError("calibration needs at least one score")
        if not 0.0 < percentile <= 100.0:
            raise ValueError(f"percentile must be in (0, 100], got {percentile}")
        return cls(reference=float(np.percentile(scores, percentile)))

    def priority(self, raw: float) -> float:
        """
        Map one raw score to a priority.

        A score that is not finite and non-negative RAISES ValueError.

        Parameters:
            - raw (float): raw model score

        Return: priority in [0, 1)
        """
        if not _is_number(raw) or not math.isfinite(raw) or raw < 0:
            raise ValueError(f"raw score must be finite and non-negative, got {raw!r}")
        return raw / (raw + self.reference)


def _is_number(value: object) -> bool:
    """
    Whether a value is a real number, excluding bool.

    Parameters:
        - value (object): value to test

    Return: True for int and float values other than bool
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)

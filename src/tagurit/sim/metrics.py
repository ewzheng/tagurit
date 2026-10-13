"""
Ranking metrics for scored, labelled items such as tiles or frames.

The scheduler only orders frames, so a scorer is judged by how well its
scores rank targets above background, not by any threshold. Positive
labels mean "contains a target". numpy only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """
    Area under the ROC curve via the Mann-Whitney U statistic.

    The probability that a random positive outscores a random negative,
    with ties counting half. Mismatched lengths, or labels missing either
    class, RAISE ValueError.

    Parameters:
        - scores (Sequence[float]): one score per item, higher meaning more likely positive
        - labels (Sequence[bool]): one label per item

    Return: AUROC in [0, 1], 0.5 for a scorer no better than chance
    """
    s, y = _arrays(scores, labels)
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC needs at least one positive and one negative")

    order = np.argsort(s, kind="mergesort")
    _, first, counts = np.unique(s[order], return_index=True, return_counts=True)
    ranks = np.empty(len(s))
    ranks[order] = np.repeat(first + (counts + 1) / 2.0, counts)
    u = ranks[y].sum() - positives * (positives + 1) / 2.0
    return float(u / (positives * negatives))


def average_precision(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """
    Mean precision at the rank of each positive, items ordered by descending score.

    Equal scores keep their input order, so ties are not averaged. Unlike
    AUROC this falls with the base rate, which makes it the honest number
    for rare targets. No positives RAISES ValueError.

    Parameters:
        - scores (Sequence[float]): one score per item
        - labels (Sequence[bool]): one label per item

    Return: average precision in (0, 1]
    """
    s, y = _arrays(scores, labels)
    if not y.any():
        raise ValueError("average precision needs at least one positive")
    hits = y[np.argsort(-s, kind="mergesort")]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(precision[hits].mean())


def recall_at_fraction(scores: Sequence[float], labels: Sequence[bool], fraction: float) -> float:
    """
    Share of positives among the highest-scored ``fraction`` of items.

    Answers "if only the top k% were sent, how many targets arrive". The
    cut takes ceil(fraction * n) items, with equal scores in input order.
    No positives, or a fraction outside (0, 1], RAISES ValueError.

    Parameters:
        - scores (Sequence[float]): one score per item
        - labels (Sequence[bool]): one label per item
        - fraction (float): share of items kept, in (0, 1]

    Return: recall in [0, 1]
    """
    s, y = _arrays(scores, labels)
    if not y.any():
        raise ValueError("recall needs at least one positive")
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    kept = math.ceil(fraction * len(s))
    top = np.argsort(-s, kind="mergesort")[:kept]
    return float(y[top].sum() / y.sum())


def _arrays(scores: Sequence[float], labels: Sequence[bool]) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert and check paired scores and labels.

    Empty input, mismatched lengths, or non-finite scores RAISE ValueError.

    Parameters:
        - scores (Sequence[float]): one score per item
        - labels (Sequence[bool]): one label per item

    Return: float scores and boolean labels as numpy arrays
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    if s.ndim != 1 or s.shape != y.shape:
        raise ValueError(f"scores and labels must be flat and equal length: {s.shape} vs {y.shape}")
    if len(s) == 0:
        raise ValueError("no items to rank")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    return s, y

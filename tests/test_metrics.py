"""Tests for the ranking metrics used to evaluate scorers."""

import pytest

from tagurit.sim.metrics import auroc, average_precision, recall_at_fraction


def test_auroc_perfect_reversed_and_tied() -> None:
    labels = [False, False, True, True]
    assert auroc([0.1, 0.2, 0.8, 0.9], labels) == 1.0
    assert auroc([0.9, 0.8, 0.2, 0.1], labels) == 0.0
    assert auroc([0.5, 0.5, 0.5, 0.5], labels) == 0.5


def test_auroc_counts_pairs() -> None:
    # Positives at 0.3 and 0.9 against negatives at 0.1, 0.5, 0.7: 1 + 3 of 6 pairs won.
    labels = [False, True, False, False, True]
    assert auroc([0.1, 0.3, 0.5, 0.7, 0.9], labels) == pytest.approx(4 / 6)


def test_average_precision() -> None:
    # Ranked: P(0.9), N(0.8), P(0.7) -> precisions 1 and 2/3 at the hits.
    assert average_precision([0.9, 0.8, 0.7], [True, False, True]) == pytest.approx((1 + 2 / 3) / 2)


def test_recall_at_fraction_takes_the_ceiling() -> None:
    scores = [0.9, 0.8, 0.7, 0.6, 0.5]
    labels = [True, False, True, False, False]
    assert recall_at_fraction(scores, labels, 0.2) == 0.5  # top 1 of 5
    assert recall_at_fraction(scores, labels, 0.5) == 1.0  # top 3 of 5
    assert recall_at_fraction(scores, labels, 0.01) == 0.5  # still at least one item


def test_metrics_reject_bad_input() -> None:
    with pytest.raises(ValueError):
        auroc([0.1, 0.2], [True, True])
    with pytest.raises(ValueError):
        auroc([0.1], [True, False])
    with pytest.raises(ValueError):
        average_precision([0.1, 0.2], [False, False])
    with pytest.raises(ValueError):
        recall_at_fraction([0.1, 0.2], [True, False], 0.0)
    with pytest.raises(ValueError):
        auroc([float("nan"), 0.2], [True, False])
    with pytest.raises(ValueError):
        auroc([], [])

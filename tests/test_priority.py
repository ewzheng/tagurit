"""Tests for the raw score to priority mapping and its calibration."""

import math

import pytest

from tagurit.tagging.priority import PriorityScale


def test_reference_maps_to_one_half_and_zero_to_zero() -> None:
    scale = PriorityScale(reference=4.0)
    assert scale.priority(4.0) == 0.5
    assert scale.priority(0.0) == 0.0


def test_mapping_is_strictly_increasing_and_never_reaches_one() -> None:
    scale = PriorityScale(reference=1.0)
    values = [scale.priority(s) for s in (0.5, 1.0, 10.0, 1e3, 1e6)]
    assert values == sorted(values) and len(set(values)) == len(values)
    assert all(0.0 <= v < 1.0 for v in values)


@pytest.mark.parametrize("reference", [0.0, -1.0, math.inf, math.nan, True])
def test_rejects_bad_references(reference: float) -> None:
    with pytest.raises(ValueError):
        PriorityScale(reference=reference)


@pytest.mark.parametrize("raw", [-0.1, math.inf, math.nan, False])
def test_rejects_bad_raw_scores(raw: float) -> None:
    with pytest.raises(ValueError):
        PriorityScale(reference=1.0).priority(raw)


def test_calibration_uses_the_requested_percentile() -> None:
    scores = [float(i) for i in range(1, 101)]
    assert PriorityScale.from_normal_scores(scores, percentile=100.0).reference == 100.0
    median = PriorityScale.from_normal_scores(scores, percentile=50.0).reference
    assert median == pytest.approx(50.5)


def test_calibration_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        PriorityScale.from_normal_scores([])
    with pytest.raises(ValueError):
        PriorityScale.from_normal_scores([1.0, 2.0], percentile=0.0)
    with pytest.raises(ValueError):
        PriorityScale.from_normal_scores([0.0, 0.0])

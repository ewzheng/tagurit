"""Tests for novelty against a time window of recent tile embeddings, and its scale."""

import math

import numpy as np
import pytest

from tagurit.model.novelty import NoveltyScale, NoveltyWindow

X, Y, Z = np.eye(3, dtype=np.float32)


def test_nothing_earlier_is_nan_and_same_frame_tiles_never_count() -> None:
    window = NoveltyWindow(10.0)
    first = window.observe(np.stack([X, X]), timestamp=0.0)
    assert np.isnan(first).all()  # its own twin tile does not make it familiar


def test_seen_tiles_are_familiar_and_new_ones_novel() -> None:
    window = NoveltyWindow(10.0)
    window.observe(np.stack([X]), timestamp=0.0)
    novelty = window.observe(np.stack([X, Y, -X]), timestamp=1.0)
    assert novelty == pytest.approx([0.0, 0.5, 1.0])


def test_frames_at_the_same_time_do_not_count() -> None:
    window = NoveltyWindow(10.0)
    window.observe(np.stack([X]), timestamp=3.0)
    assert np.isnan(window.observe(np.stack([X]), timestamp=3.0)).all()
    assert window.observe(np.stack([X]), timestamp=3.5) == pytest.approx([0.0])


def test_frames_older_than_the_window_are_forgotten() -> None:
    window = NoveltyWindow(2.0)
    window.observe(np.stack([X]), timestamp=0.0)
    window.observe(np.stack([Y]), timestamp=1.5)
    novelty = window.observe(np.stack([X, Y]), timestamp=3.0)
    assert novelty[0] == pytest.approx(0.5)  # X at 0 s is out of a 2 s window at 3 s
    assert novelty[1] == pytest.approx(0.0)


def test_a_clock_that_goes_back_starts_a_new_window() -> None:
    window = NoveltyWindow(10.0)
    window.observe(np.stack([X]), timestamp=100.0)
    assert np.isnan(window.observe(np.stack([X]), timestamp=0.0)).all()
    window.reset()
    assert np.isnan(window.observe(np.stack([Z]), timestamp=1.0)).all()


def test_bad_input_is_rejected() -> None:
    with pytest.raises(ValueError):
        NoveltyWindow(0.0)
    with pytest.raises(ValueError):
        NoveltyWindow(math.inf)
    window = NoveltyWindow(1.0)
    with pytest.raises(ValueError):
        window.observe(X, timestamp=0.0)  # not (tiles, dim)
    with pytest.raises(ValueError):
        window.observe(np.stack([X]), timestamp=math.nan)
    window.observe(np.stack([X]), timestamp=0.0)
    with pytest.raises(ValueError):
        window.observe(np.ones((1, 4), dtype=np.float32), timestamp=0.5)


def test_scale_is_neutral_until_enough_novelty_is_seen() -> None:
    scale = NoveltyScale(size=100, minimum=3)
    scale.add(np.array([0.1, 0.2]))
    assert (scale.percentile(np.array([0.0, 1.0])) == 0.5).all()
    scale.add(np.array([np.nan, 0.3]))  # NaN says nothing and is skipped
    assert scale.percentile(np.array([0.0, 0.2, 1.0, np.nan])) == pytest.approx(
        [0.0, 0.5, 1.0, 0.5]
    )


def test_scale_samples_at_most_its_size_and_repeats() -> None:
    values = np.random.default_rng(0).uniform(size=5000)
    first, second = NoveltyScale(size=200, minimum=1), NoveltyScale(size=200, minimum=1)
    for scale in (first, second):
        for chunk in np.split(values, 50):
            scale.add(chunk)
    probe = np.array([0.25, 0.5, 0.75])
    assert first.percentile(probe) == pytest.approx(second.percentile(probe))
    assert first.percentile(probe) == pytest.approx(probe, abs=0.1)  # a fair sample
    first.reset()
    assert (first.percentile(probe) == 0.5).all()


def test_scale_rejects_bad_sizes() -> None:
    with pytest.raises(ValueError):
        NoveltyScale(size=0, minimum=1)
    with pytest.raises(ValueError):
        NoveltyScale(size=5, minimum=6)

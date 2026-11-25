"""Tests for novelty against a window of the last few frames' tile embeddings, and its scale."""

import numpy as np
import pytest

from tagurit.model.novelty import NoveltyScale, NoveltyWindow

X, Y, Z = np.eye(3, dtype=np.float32)


def test_the_first_frame_is_nan_and_same_frame_tiles_never_count() -> None:
    window = NoveltyWindow(10)
    first = window.observe(np.stack([X, X]))
    assert np.isnan(first).all()  # its own twin tile does not make it familiar


def test_seen_tiles_are_familiar_and_new_ones_novel() -> None:
    window = NoveltyWindow(10)
    window.observe(np.stack([X]))
    novelty = window.observe(np.stack([X, Y, -X]))
    assert novelty == pytest.approx([0.0, 0.5, 1.0])


def test_every_frame_after_the_first_has_something_to_compare_with() -> None:
    window = NoveltyWindow(1)
    window.observe(np.stack([X]))
    for _ in range(5):  # however long ago the last frame was, it is still the last frame
        assert window.observe(np.stack([X])) == pytest.approx([0.0])


def test_only_the_last_frames_are_remembered() -> None:
    window = NoveltyWindow(2)
    window.observe(np.stack([X]))
    window.observe(np.stack([Y]))
    window.observe(np.stack([Y]))  # the third frame pushes X out of a two-frame window
    novelty = window.observe(np.stack([X, Y]))
    assert novelty[0] == pytest.approx(0.5)
    assert novelty[1] == pytest.approx(0.0)


def test_a_frame_stays_for_exactly_the_window() -> None:
    window = NoveltyWindow(3)
    window.observe(np.stack([X]))
    window.observe(np.stack([Y]))
    window.observe(np.stack([Y]))
    assert window.observe(np.stack([X])) == pytest.approx([0.0])  # X is three frames back


def test_reset_starts_a_new_flight() -> None:
    window = NoveltyWindow(10)
    window.observe(np.stack([X]))
    window.reset()
    assert np.isnan(window.observe(np.stack([X]))).all()
    assert window.observe(np.stack([X])) == pytest.approx([0.0])


def test_a_frame_without_tiles_takes_a_place_but_adds_nothing() -> None:
    window = NoveltyWindow(2)
    window.observe(np.zeros((0, 3), dtype=np.float32))
    assert np.isnan(window.observe(np.stack([X]))).all()
    assert window.observe(np.zeros((0, 3), dtype=np.float32)).shape == (0,)
    assert window.observe(np.stack([X])) == pytest.approx([0.0])


def test_bad_input_is_rejected() -> None:
    for frames in (0, -1, 2.0, 10.0, True, None):
        with pytest.raises(ValueError):
            NoveltyWindow(frames)  # type: ignore[arg-type]
    window = NoveltyWindow(1)
    with pytest.raises(ValueError):
        window.observe(X)  # not (tiles, dim)
    window.observe(np.stack([X]))
    with pytest.raises(ValueError):
        window.observe(np.ones((1, 4), dtype=np.float32))


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

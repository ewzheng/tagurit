"""Tests for frame tagging and bundle settings, using a fake scorer instead of a model."""

import json
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pytest

from tagurit.protocol import ImageFrame
from tagurit.tagging.bundle import SETTINGS_FILE, load_bundle
from tagurit.tagging.priority import PriorityScale
from tagurit.tagging.tagger import Tagger


class MeanBrightness:
    """Scores a tile by its mean pixel value, so a bright patch is the anomaly."""

    def __init__(self) -> None:
        self.calls = 0

    def score_tiles(self, tiles: Sequence[np.ndarray]) -> list[float]:
        self.calls += 1
        return [float(tile.mean()) for tile in tiles]


class WrongCount:
    """Returns one score too few, which the tagger must reject."""

    def score_tiles(self, tiles: Sequence[np.ndarray]) -> list[float]:
        return [0.0] * (len(tiles) - 1)


class Windowed(MeanBrightness):
    """Counts the frames it was asked to remember, as a scorer with a novelty window would."""

    def __init__(self) -> None:
        super().__init__()
        self.frames = 0
        self.resets = 0

    def score_frame(self, tiles: Sequence[np.ndarray]) -> list[float]:
        self.frames += 1
        return [float(tile.mean()) + 1.0 for tile in tiles]

    def reset(self) -> None:
        self.resets += 1


def encode(image: np.ndarray) -> bytes:
    ok, png = cv2.imencode(".png", image)
    assert ok
    return png.tobytes()


def frame_with_bright_corner() -> bytes:
    image = np.full((64, 96, 3), 10, dtype=np.uint8)
    image[40:64, 70:96] = 250  # inside the bottom-right 32px tile only
    return encode(image)


def test_frame_score_is_its_highest_tile() -> None:
    scorer = MeanBrightness()
    tagger = Tagger(scorer, PriorityScale(reference=10.0), tile_size=32)
    result = tagger.score(frame_with_bright_corner())

    assert len(result.tiles) == 6 and len(result.tile_scores) == 6
    brightest = max(range(6), key=lambda i: result.tile_scores[i])
    assert (result.tiles[brightest].left, result.tiles[brightest].top) == (64, 32)
    assert result.raw_score == result.tile_scores[brightest]
    assert result.priority == pytest.approx(result.raw_score / (result.raw_score + 10.0))
    assert scorer.calls == 1


def test_brighter_frames_rank_higher() -> None:
    tagger = Tagger(MeanBrightness(), PriorityScale(reference=10.0), tile_size=32)
    dull = tagger.score(encode(np.full((64, 96, 3), 10, dtype=np.uint8)))
    bright = tagger.score(frame_with_bright_corner())
    assert bright.priority > dull.priority


def test_tag_builds_an_image_frame_with_the_original_bytes() -> None:
    tagger = Tagger(MeanBrightness(), PriorityScale(reference=10.0), tile_size=32)
    image_bytes = frame_with_bright_corner()
    frame = tagger.tag(frame_id=7, timestamp=123.5, image_bytes=image_bytes)
    assert isinstance(frame, ImageFrame)
    assert frame.frame_id == 7 and frame.timestamp == 123.5
    assert frame.image_bytes == image_bytes
    assert 0.0 <= frame.priority < 1.0


def test_windowed_scorer_remembers_only_when_asked_and_plain_scorer_never() -> None:
    windowed = Windowed()
    tagger = Tagger(windowed, PriorityScale(reference=10.0), tile_size=32)
    tagged = tagger.tag(frame_id=1, timestamp=5.0, image_bytes=frame_with_bright_corner())
    assert tagged.timestamp == 5.0  # carried along, not read by scoring
    alone = tagger.score(frame_with_bright_corner())
    remembered = tagger.score(frame_with_bright_corner(), remember=True)
    assert windowed.frames == 2  # the tagged frame and the remembered one
    assert windowed.calls == 1  # the call without remember scored tiles alone
    assert remembered.raw_score == pytest.approx(alone.raw_score + 1.0)
    tagger.reset()
    assert windowed.resets == 1

    plain = MeanBrightness()
    plain_tagger = Tagger(plain, PriorityScale(reference=10.0), tile_size=32)
    plain_tagger.score(frame_with_bright_corner(), remember=True)
    plain_tagger.reset()  # nothing to forget
    assert plain.calls == 1


class Retargetable(Windowed):
    """A windowed scorer whose query is a brightness gain, and which keeps each frame's peak."""

    def __init__(self) -> None:
        super().__init__()
        self.gain = 1.0

    def score_and_keep(self, tiles: Sequence[np.ndarray]) -> tuple[list[float], object]:
        scores = [self.gain * s for s in self.score_frame(tiles)]
        return scores, max(scores) / self.gain

    def set_targets(self, phrases: Sequence[str]) -> None:
        self.gain = float(len(phrases))

    def rescore(self, kept: Sequence[object]) -> list[float]:
        return [self.gain * peak for peak in kept]  # type: ignore[operator]


def test_a_retargetable_scorer_lets_tagged_frames_be_rescored_without_their_images() -> None:
    tagger = Tagger(Retargetable(), PriorityScale(reference=100.0), tile_size=32)
    first = tagger.tag(1, 10.0, frame_with_bright_corner())
    second = tagger.tag(2, 11.0, encode(np.full((64, 64, 3), 40, dtype=np.uint8)))
    assert first.kept is not None and second.kept is not None
    assert tagger.rescore([first, second]) == pytest.approx({1: first.priority, 2: second.priority})

    tagger.set_targets(["a boat", "a swimmer", "a raft"])  # gain 3
    plain = ImageFrame(3, 12.0, b"elsewhere", 0.5)  # tagged without kept state: left out
    rescored = tagger.rescore([first, plain, second])
    assert set(rescored) == {1, 2}
    assert rescored[1] > first.priority and rescored[2] > second.priority
    assert rescored[1] > rescored[2]
    assert tagger.tag(4, 13.0, frame_with_bright_corner()).priority == pytest.approx(rescored[1])


def test_frames_from_other_scorers_carry_nothing_and_cannot_be_retargeted() -> None:
    tagger = Tagger(Windowed(), PriorityScale(reference=100.0), tile_size=32)
    assert tagger.tag(1, 10.0, frame_with_bright_corner()).kept is None
    with pytest.raises(TypeError):
        tagger.set_targets(["a boat"])
    with pytest.raises(TypeError):
        tagger.rescore([])


def test_tagger_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        Tagger(MeanBrightness(), PriorityScale(reference=1.0), tile_size=0)
    with pytest.raises(ValueError):
        Tagger(MeanBrightness(), PriorityScale(reference=1.0), tile_size=32, detection=1.5)
    with pytest.raises(ValueError):
        Tagger(WrongCount(), PriorityScale(reference=1.0), tile_size=32).score(
            frame_with_bright_corner()
        )
    with pytest.raises(ValueError):
        Tagger(MeanBrightness(), PriorityScale(reference=1.0), tile_size=128).score(
            frame_with_bright_corner()
        )


def test_bundle_with_unknown_model_is_rejected(tmp_path: Path) -> None:
    settings = {"model": "mystery", "model_file": "x", "tile_size": 32, "reference": 1.0}
    (tmp_path / SETTINGS_FILE).write_text(json.dumps(settings))
    with pytest.raises(ValueError):
        load_bundle(tmp_path)

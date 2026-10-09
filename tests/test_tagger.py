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


def test_tagger_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        Tagger(MeanBrightness(), PriorityScale(reference=1.0), tile_size=0)
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

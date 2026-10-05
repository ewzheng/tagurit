"""
Tests for the anomalib-backed PatchCore scorer and its bundle round trip.

Skipped without the ``model`` extra. A small randomly initialised backbone
keeps the tests offline (no weight download) and fast; the scores are
meaningless as detections but follow the same code path.
"""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("anomalib")

from tagurit.model.patchcore import PatchcoreConfig, PatchcoreScorer  # noqa: E402
from tagurit.tagging.bundle import SETTINGS_FILE, load_bundle, save_patchcore_bundle  # noqa: E402
from tagurit.tagging.priority import PriorityScale  # noqa: E402

CONFIG = PatchcoreConfig(backbone="resnet18", input_size=64, coreset_ratio=0.5, pre_trained=False)


def flat_tiles(count: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Near-uniform grey tiles: the normal class."""
    return [
        np.clip(128 + rng.normal(0, 2, (64, 64, 3)), 0, 255).astype(np.uint8) for _ in range(count)
    ]


def checkerboard() -> np.ndarray:
    """A high-contrast pattern unlike anything in the normal tiles."""
    board = np.indices((64, 64)).sum(axis=0) // 8 % 2
    return np.repeat((board * 255).astype(np.uint8)[..., None], 3, axis=2)


@pytest.fixture(scope="module")
def scorer() -> PatchcoreScorer:
    return PatchcoreScorer.fit(
        flat_tiles(6, np.random.default_rng(0)), config=CONFIG, device="cpu", batch_size=4
    )


def test_fit_fills_a_bank_of_the_expected_size(scorer: PatchcoreScorer) -> None:
    # resnet18 layer2 at 64px input is an 8x8 grid: 64 patches per tile, half kept.
    assert scorer.bank_size == 6 * 64 // 2


def test_unusual_tiles_score_higher(scorer: PatchcoreScorer) -> None:
    normal, odd = scorer.score_tiles([flat_tiles(1, np.random.default_rng(1))[0], checkerboard()])
    assert np.isfinite([normal, odd]).all() and normal >= 0
    assert odd > normal


def test_tiles_of_another_size_are_resized(scorer: PatchcoreScorer) -> None:
    big = np.full((128, 128, 3), 128, dtype=np.uint8)
    assert len(scorer.score_tiles([big, big[:32, :32]])) == 2
    assert scorer.score_tiles([]) == []


def test_bad_tiles_are_rejected(scorer: PatchcoreScorer) -> None:
    with pytest.raises(ValueError):
        scorer.score_tiles([np.zeros((64, 32, 3), dtype=np.uint8)])
    with pytest.raises(ValueError):
        scorer.score_tiles([np.zeros((64, 64, 3), dtype=np.float32)])
    with pytest.raises(ValueError):
        PatchcoreScorer.fit([], config=CONFIG, device="cpu")


def test_bundle_round_trip_keeps_bank_and_settings(scorer: PatchcoreScorer, tmp_path: Path) -> None:
    save_patchcore_bundle(tmp_path, scorer, PriorityScale(reference=2.5), tile_size=128)
    tagger = load_bundle(tmp_path, device="cpu")
    loaded = tagger.scorer
    assert isinstance(loaded, PatchcoreScorer)
    assert loaded.config == CONFIG
    assert loaded.bank_size == scorer.bank_size
    assert tagger.scale.reference == 2.5 and tagger.tile_size == 128


def test_config_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        PatchcoreConfig(coreset_ratio=0.0)
    with pytest.raises(ValueError):
        PatchcoreConfig(layers=())
    with pytest.raises(ValueError):
        PatchcoreConfig(num_neighbors=0)


def test_half_precision_matches_full_precision(scorer: PatchcoreScorer, tmp_path: Path) -> None:
    # fit seeds torch before building the backbone, so the same seed and tiles
    # rebuild the fixture's exact model; only the scoring precision differs.
    half = PatchcoreScorer.fit(
        flat_tiles(6, np.random.default_rng(0)),
        config=CONFIG,
        device="cpu",
        batch_size=4,
        half=True,
    )
    tiles = [flat_tiles(1, np.random.default_rng(2))[0], checkerboard()]
    assert half.half and not scorer.half
    full_scores = scorer.score_tiles(tiles)
    # float16 loses relative precision on near-zero distances, so compare
    # against the score range rather than each score.
    tolerance = 0.01 * max(full_scores)
    assert np.allclose(half.score_tiles(tiles), full_scores, rtol=0.0, atol=tolerance)

    save_patchcore_bundle(tmp_path, half, PriorityScale(reference=1.0), tile_size=64)
    assert json.loads((tmp_path / SETTINGS_FILE).read_text())["half"] is True
    assert load_bundle(tmp_path, device="cpu").scorer.half
    assert not load_bundle(tmp_path, device="cpu", half=False).scorer.half

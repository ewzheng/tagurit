"""
Tests for the zero-shot CLIP scorer and its bundle settings.

Skipped without the ``model`` extra. A tiny randomly initialised CLIP and a
stub tokenizer keep the tests offline; the scores are meaningless but
follow the real code path.
"""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from tagurit.model.clip import ClipConfig, ClipScorer  # noqa: E402
from tagurit.tagging.bundle import SETTINGS_FILE, save_clip_bundle  # noqa: E402
from tagurit.tagging.priority import PriorityScale  # noqa: E402

CONFIG = ClipConfig(
    model="tiny-test-clip", targets=("a person", "a boat"), backgrounds=("the sea", "a road")
)


def tiny_clip() -> transformers.CLIPModel:
    torch.manual_seed(0)
    text = dict(
        vocab_size=50,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=16,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    vision = dict(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=32,
        patch_size=8,
    )
    config = transformers.CLIPConfig(text_config=text, vision_config=vision, projection_dim=16)
    return transformers.CLIPModel(config)


def stub_tokenize(prompts: list[str]) -> dict[str, "torch.Tensor"]:
    """Start token, one id per character, end token, zero padding."""
    rows = [[1, *(3 + ord(c) % 40 for c in p[:10]), 2] for p in prompts]
    width = max(len(r) for r in rows)
    ids = torch.tensor([r + [0] * (width - len(r)) for r in rows])
    return {"input_ids": ids, "attention_mask": (ids != 0).long()}


def make_scorer(half: bool = False) -> ClipScorer:
    return ClipScorer(
        tiny_clip(),
        stub_tokenize,
        image_size=32,
        mean=(0.5, 0.5, 0.5),
        std=(0.25, 0.25, 0.25),
        config=CONFIG,
        device="cpu",
        batch_size=3,
        half=half,
    )


def tiles(count: int) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, (48, 48, 3), dtype=np.uint8) for _ in range(count)]


def test_scores_are_target_probabilities() -> None:
    scores = make_scorer().score_tiles(tiles(5))
    assert len(scores) == 5
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert make_scorer().score_tiles([]) == []


def test_half_precision_stays_close() -> None:
    full = make_scorer().score_tiles(tiles(4))
    half = make_scorer(half=True).score_tiles(tiles(4))
    assert np.allclose(full, half, atol=0.02)


def test_bad_tiles_and_configs_are_rejected() -> None:
    with pytest.raises(ValueError):
        make_scorer().score_tiles([np.zeros((32, 16, 3), dtype=np.uint8)])
    with pytest.raises(ValueError):
        ClipConfig(targets=())
    with pytest.raises(ValueError):
        ClipConfig(template="no placeholder")


def test_bundle_records_checkpoint_and_prompts(tmp_path: Path) -> None:
    save_clip_bundle(tmp_path, make_scorer(), PriorityScale(reference=0.3), tile_size=256)
    settings = json.loads((tmp_path / SETTINGS_FILE).read_text())
    assert settings["model"] == "clip"
    assert settings["clip"]["model"] == "tiny-test-clip"
    assert settings["clip"]["targets"] == ["a person", "a boat"]
    assert (settings["tile_size"], settings["reference"], settings["half"]) == (256, 0.3, False)

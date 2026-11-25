"""
Tests for the query-only CLIP scorer with window novelty, and its bundle.

Skipped without the ``model`` extra. A tiny randomly initialised CLIP and a
stub tokenizer keep the tests offline; the scores are meaningless but
follow the real code path.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from tagurit.model.clip import (  # noqa: E402
    ClipConfig,
    ClipScorer,
    RobustReference,
    combine,
    retarget,
)
from tagurit.model.scorer import WindowedTileScorer  # noqa: E402
from tagurit.tagging.bundle import (  # noqa: E402
    CLIP_REFERENCE_FILE,
    SETTINGS_FILE,
    load_bundle,
    save_clip_bundle,
)
from tagurit.tagging.priority import PriorityScale  # noqa: E402

CONFIG = ClipConfig(model="tiny-test-clip", targets=("a person", "a boat"), window_frames=5)


def tiny_clip() -> transformers.CLIPModel:
    torch.manual_seed(0)
    text = dict(
        vocab_size=50,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=32,
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
    """Start token, one id per character of the prompt's tail, end token, zero padding.

    The tail, because every formatted prompt starts with the same template text.
    """
    rows = [[1, *(3 + ord(c) % 40 for c in p[-12:]), 2] for p in prompts]
    width = max(len(r) for r in rows)
    ids = torch.tensor([r + [0] * (width - len(r)) for r in rows])
    return {"input_ids": ids, "attention_mask": (ids != 0).long()}


def tiles(count: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (48, 48, 3), dtype=np.uint8) for _ in range(count)]


def make_scorer(
    half: bool = False,
    config: ClipConfig = CONFIG,
    reference: np.ndarray | None = None,
    calibrate: bool = True,
    min_history: int = 8,
) -> ClipScorer:
    scorer = ClipScorer(
        tiny_clip(),
        stub_tokenize,
        image_size=32,
        mean=(0.5, 0.5, 0.5),
        std=(0.25, 0.25, 0.25),
        config=config,
        reference=reference,
        device="cpu",
        batch_size=3,
        half=half,
        history=64,
        min_history=min_history,
    )
    if calibrate and reference is None:
        scorer.calibrate(tiles(20, seed=1))
    return scorer


def unit_rows(count: int, seed: int) -> np.ndarray:
    rows = np.random.default_rng(seed).normal(size=(count, 16)).astype(np.float32)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def test_the_scorer_remembers_frames() -> None:
    assert isinstance(make_scorer(), WindowedTileScorer)


def test_an_uncalibrated_scorer_refuses_to_score() -> None:
    scorer = make_scorer(calibrate=False)
    with pytest.raises(ValueError):
        scorer.score_tiles(tiles(2))
    with pytest.raises(ValueError):
        scorer.query_term(("a boat",))
    with pytest.raises(ValueError):
        save_clip_bundle(Path("unused"), scorer, PriorityScale(reference=1.0), tile_size=224)


def test_query_is_each_phrase_minus_the_bare_template() -> None:
    scorer = make_scorer()
    model = scorer._model
    prompts = [CONFIG.template.format(p) for p in CONFIG.targets] + [CONFIG.null_prompt]
    with torch.no_grad():
        text = model.get_text_features(**stub_tokenize(prompts))
    text = getattr(text, "pooler_output", text)
    text = (text / text.norm(dim=-1, keepdim=True)).numpy()
    directions = text[:-1] - text[-1]
    assert np.allclose(scorer.query.directions, directions, atol=1e-5)

    embeddings = scorer.embed(tiles(5))
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)
    assert np.allclose(scorer.query.scores(embeddings), (embeddings @ directions.T).max(axis=1))


def test_score_tiles_is_the_query_alone_on_the_reference_scale() -> None:
    scorer = make_scorer()
    batch = tiles(5)
    scores = scorer.score_tiles(batch)
    query_z = scorer.query.z(scorer.embed(batch))
    assert scores == pytest.approx(combine(query_z, np.full(5, 0.5)).tolist())
    assert all(s > 0 and math.isfinite(s) for s in scores)
    assert len({round(s, 6) for s in scores}) > 1  # tiles really differ
    assert scorer.score_tiles([]) == []


def test_novelty_lowers_tiles_just_seen_and_retarget_reproduces_scores() -> None:
    scorer = make_scorer()
    for t in range(4):  # enough recent novelty to set its scale
        scorer.terms(unit_rows(6, seed=t))
    seen = unit_rows(6, seed=3)[0]
    new = unit_rows(1, seed=99)[0]
    terms = scorer.terms(np.stack([seen, new]))
    assert terms.novelty_p[0] < terms.novelty_p[1]
    assert np.allclose(retarget(terms.embeddings, terms.novelty_p, scorer.query), terms.scores)
    assert np.allclose(terms.scores, combine(terms.query_z, terms.novelty_p))

    scorer.reset()  # no recent novelty, so novelty is neutral again
    assert (scorer.terms(np.stack([new])).novelty_p == 0.5).all()


def test_the_window_holds_the_configured_number_of_frames() -> None:
    def novelty_of_repeat(frames_between: int) -> np.ndarray:
        scorer = make_scorer(min_history=1)  # CONFIG's window is five frames
        scorer.terms(unit_rows(4, seed=0))
        for i in range(frames_between):
            scorer.terms(unit_rows(4, seed=100 + i))
        return scorer.terms(unit_rows(4, seed=0)).novelty_p

    remembered, forgotten = novelty_of_repeat(4), novelty_of_repeat(5)
    assert (remembered == 0.0).all()  # five frames back: still in the window, nothing is less novel
    assert (forgotten > 0.0).all()  # six frames back: out of it


def test_only_reset_starts_a_new_flight() -> None:
    scorer = make_scorer(min_history=1)
    scorer.terms(unit_rows(4, seed=0))
    scorer.terms(unit_rows(4, seed=1))
    assert (scorer.terms(unit_rows(4, seed=0)).novelty_p == 0.0).all()  # seen two frames ago
    scorer.reset()
    assert (scorer.terms(unit_rows(4, seed=0)).novelty_p == 0.5).all()  # a flight's first frame


def test_score_frame_matches_terms_on_its_embeddings() -> None:
    first, second = make_scorer(), make_scorer()
    batch = tiles(4)
    for _ in range(3):
        scores = first.score_frame(batch)
        expected = second.terms(second.embed(batch)).scores
        assert scores == pytest.approx(expected.tolist())


def test_set_targets_switches_the_query() -> None:
    scorer = make_scorer()
    batch = tiles(5)
    before = scorer.score_tiles(batch)
    scorer.set_targets(("a swimmer",))
    assert scorer.config.targets == ("a swimmer",)
    assert scorer.query.phrases == ("a swimmer",)
    assert scorer.score_tiles(batch) != pytest.approx(before)
    with pytest.raises(ValueError):
        scorer.set_targets(())


def test_combine_is_increasing_and_never_saturates() -> None:
    z = np.linspace(-50, 200, 101)
    for novelty in (0.0, 0.5, 1.0):
        scores = combine(z, np.full_like(z, novelty))
        assert (np.diff(scores) > 0).all() and (scores > 0).all() and np.isfinite(scores).all()
    p = np.linspace(0, 1, 11)
    assert (np.diff(combine(np.zeros_like(p), p)) > 0).all()
    scale = PriorityScale(reference=1.5)
    for novelty in (0.5, 1.0):
        raw = combine(np.array([100.0, 101.0]), np.full(2, novelty))
        strong = [scale.priority(float(s)) for s in raw]
        assert strong[0] < strong[1] < 1.0
    # the mean percentile orders tiles: top query and median novelty loses to both high
    both = combine(np.array([2.0]), np.array([0.98]))
    one = combine(np.array([6.0]), np.array([0.5]))
    assert both > one
    with pytest.raises(ValueError):
        combine(np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError):
        combine(np.array([np.inf]), np.full(1, 0.5))
    with pytest.raises(ValueError):
        combine(np.zeros(1), np.full(1, 1.5))


def test_robust_reference() -> None:
    reference = RobustReference.of(np.array([1.0, 2.0, 3.0, 4.0, 100.0]))
    assert reference.median == 3.0
    assert reference.spread == pytest.approx(1.4826)
    assert reference.z(np.array([3.0, 4.4826])) == pytest.approx([0.0, 1.0])
    with pytest.raises(ValueError):
        RobustReference.of(np.array([]))
    with pytest.raises(ValueError):
        RobustReference.of(np.ones(5))


def test_half_precision_stays_close() -> None:
    full = make_scorer().embed(tiles(4))
    half = make_scorer(half=True).embed(tiles(4))
    assert np.allclose(full, half, atol=0.01)


def test_bad_tiles_and_configs_are_rejected() -> None:
    with pytest.raises(ValueError):
        make_scorer().score_tiles([np.zeros((32, 16, 3), dtype=np.uint8)])
    with pytest.raises(ValueError):
        make_scorer().terms(np.ones((2, 5), dtype=np.float32))
    with pytest.raises(ValueError):
        ClipConfig(targets=())
    with pytest.raises(ValueError):
        ClipConfig(template="no placeholder")
    with pytest.raises(ValueError):
        ClipConfig(null_prompt="  ")
    for frames in (0, -3, 10.0, True):
        with pytest.raises(ValueError):
            ClipConfig(window_frames=frames)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        make_scorer(min_history=65)
    with pytest.raises(ValueError):
        make_scorer(reference=np.ones((3, 5), dtype=np.float32))


def test_kept_tiles_rescore_frames_under_the_same_and_a_new_query() -> None:
    scorer, twin = make_scorer(), make_scorer()
    kept, firsts = [], []
    for i in range(4):
        frame = tiles(3, seed=10 + i)
        scores, state = scorer.score_and_keep(frame)
        assert scores == twin.score_frame(frame)  # same scoring, same memory
        assert state.embeddings.dtype == np.float16 and state.embeddings.shape == (3, 16)
        kept.append(state)
        firsts.append(max(scores))
    # Half-precision embeddings move a raw score only slightly.
    assert scorer.rescore(kept) == pytest.approx(firsts, rel=1e-2)
    assert scorer.rescore([]) == []

    scorer.set_targets(["a car"])
    expected = [
        retarget(state.embeddings.astype(np.float32), state.novelty_p, scorer.query).max()
        for state in kept
    ]
    assert scorer.rescore(kept) == pytest.approx(expected)
    assert scorer.rescore(kept) != pytest.approx(firsts, rel=1e-2)  # the new query reorders


def test_bundle_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scorer = make_scorer()
    save_clip_bundle(tmp_path, scorer, PriorityScale(reference=0.3), tile_size=224)
    settings = json.loads((tmp_path / SETTINGS_FILE).read_text())
    assert settings["model"] == "clip"
    assert settings["clip"]["model"] == "tiny-test-clip"
    assert settings["clip"]["targets"] == ["a person", "a boat"]
    assert settings["clip"]["window_frames"] == 5
    assert "window" not in settings["clip"]
    assert settings["reference_file"] == CLIP_REFERENCE_FILE
    assert (settings["tile_size"], settings["reference"], settings["half"]) == (224, 0.3, False)
    assert "detection" not in settings

    def fake_load(cls, config, reference=None, device=None, batch_size=32, half=False):
        return make_scorer(half=half, config=config, reference=reference)

    monkeypatch.setattr(ClipScorer, "load", classmethod(fake_load))
    tagger = load_bundle(tmp_path)
    assert np.array_equal(tagger.scorer.reference, scorer.reference)
    assert tagger.scorer.config == CONFIG
    assert (tagger.tile_size, tagger.scale.reference) == (224, 0.3)
    assert tagger.detection is None
    batch = tiles(3)
    assert tagger.scorer.score_tiles(batch) == pytest.approx(scorer.score_tiles(batch))

    save_clip_bundle(tmp_path, scorer, PriorityScale(reference=0.3), tile_size=224, detection=0.8)
    assert load_bundle(tmp_path).detection == 0.8


def test_bundles_with_a_window_in_seconds_are_refused(tmp_path: Path) -> None:
    save_clip_bundle(tmp_path, make_scorer(), PriorityScale(reference=0.3), tile_size=224)
    settings = json.loads((tmp_path / SETTINGS_FILE).read_text())
    del settings["clip"]["window_frames"]
    settings["clip"]["window"] = 10.0  # as bundles were written before the window counted frames
    (tmp_path / SETTINGS_FILE).write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="10 seconds.*calibrate_clip"):
        load_bundle(tmp_path)


def test_old_reject_phrase_bundles_are_refused(tmp_path: Path) -> None:
    clip = {"model": "x", "targets": ["a boat"], "backgrounds": ["the sea"], "template": "{}"}
    settings = {"model": "clip", "clip": clip, "tile_size": 256, "reference": 0.5, "half": True}
    (tmp_path / SETTINGS_FILE).write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="calibrate_clip"):
        load_bundle(tmp_path)

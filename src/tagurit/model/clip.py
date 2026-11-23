"""
Score tiles with CLIP by the text query alone, plus novelty against what was just seen.

PatchCore asks "is this unlike the normal tiles I was fitted on", which
flags anything unusual, target or not. Here the mission's text query says
what is interesting, and novelty says what is new. Both terms come from one
CLIP image embedding per tile:

- Query: for each query phrase, the direction from the bare template
  (``null_prompt``, "an aerial photo.") to the phrase placed in the template
  ("an aerial photo of a boat."). A tile's query score is its best phrase's
  dot product with that direction. Subtracting the bare template cancels
  what every tile shares with any aerial caption, which plain cosine cannot
  rank past. There are no reject phrases: nothing about the scenery is
  written by hand.
- Novelty: (1 - the tile's highest cosine to a tile of an earlier frame in
  the last ``window`` seconds) / 2 (``model.novelty``), so a target that
  stays in view fades while one that has just appeared stands out.

The experiments combined the two terms by averaging their ranks over the
whole stream (``scripts/experiment_clipnn_window.py``), which needs every
frame before ranking any. On arrival each term becomes a percentile
instead, and a tile scores the mean of its two percentiles:

- The query score's robust z-score against its median and spread on
  target-free calibration tiles kept with the scorer (``reference``),
  recomputed whenever the query changes, becomes a percentile through
  ``sigmoid(1.702 z)``, the logistic approximation to the normal CDF. It
  stays smooth past the reference's largest score.
- Novelty becomes its percentile among the novelty of the flight's earlier
  tiles (``model.novelty.NoveltyScale``), which adapts to frame rate and
  terrain. It is neutral, 0.5, when no earlier frame is in the window or
  fewer than ``NOVELTY_MIN_HISTORY`` tiles have been seen.

Percentiles, not z-scores, keep one term's heavy tail from drowning the
other: on open water novelty's z-scores run past 50 while the query's stop
near 7. The raw score is softplus of the mean percentile's log-odds, plus
``TIE_BREAK`` times a smooth ramp of the query z-score. It is finite,
positive, strictly increasing in both terms and unbounded, so
``tagging.priority`` never ties strong detections. Its order is that of the
mean percentile.

``score_tiles`` has no timestamps, so novelty is neutral there; calibration
uses it. ``score_frame`` takes each frame's capture time and keeps the
window, so frames must come in capture order. ``terms`` returns what a
backlog can keep per tile, the embedding and novelty percentile, and
``retarget`` rescores those against a new query with one matrix product,
without the images. Models load through Hugging Face ``transformers``
(``AutoModel``), so any CLIP-style checkpoint with ``get_text_features``
and ``get_image_features`` works by name, and weights download on first
use. torch and transformers are imported inside functions, so importing
this module stays cheap.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from tagurit.model.device import pick_device
from tagurit.model.novelty import NoveltyScale, NoveltyWindow
from tagurit.model.preprocess import tiles_to_tensor

if TYPE_CHECKING:
    import torch

DEFAULT_MODEL = "openai/clip-vit-base-patch16"
DEFAULT_TARGETS = (
    "a person",
    "a swimmer",
    "a boat",
    "a car",
    "a truck",
    "a bicycle",
    "a motorcycle",
    "an animal",
)
# hand-written reject phrases of the earlier softmax scorer; unused here, kept for the experiments
DEFAULT_BACKGROUNDS = (
    "the open sea",
    "waves on water",
    "a road",
    "a parking lot",
    "grass",
    "trees",
    "a field",
    "buildings",
    "rooftops",
    "sand",
    "rocks",
)
NULL_PROMPT = "an aerial photo."
NOVELTY_HISTORY = 10_000  # earlier tiles of the flight sampled to place novelty
NOVELTY_MIN_HISTORY = 200  # tiles seen before novelty counts
MAD_TO_SIGMA = 1.4826  # median absolute deviation to standard deviation, for normal data
LOGISTIC_SLOPE = 1.702  # sigmoid(1.702 z) is within 0.01 of the normal CDF
TIE_BREAK = 1e-6  # share of the query z-score that keeps strong detections apart


@dataclass(frozen=True)
class ClipConfig:
    """
    Everything that determines a CLIP scorer, saved in its bundle.

    Each query phrase is placed into ``template`` before encoding. No query
    phrases, a template without ``{}``, a blank bare template, or a window
    that is not positive and finite RAISE ValueError.

    Parameters:
        - model (str): Hugging Face model id of a CLIP-style checkpoint
        - targets (tuple[str, ...]): the query: phrases for what is worth sending
        - template (str): sentence each phrase is placed into
        - null_prompt (str): the template with no object, subtracted from each phrase
        - window (float): seconds of earlier frames novelty is measured against
    """

    model: str = DEFAULT_MODEL
    targets: tuple[str, ...] = DEFAULT_TARGETS
    template: str = "an aerial photo of {}."
    null_prompt: str = NULL_PROMPT
    window: float = 10.0

    def __post_init__(self) -> None:
        if not self.targets:
            raise ValueError("CLIP needs at least one query phrase")
        if "{}" not in self.template:
            raise ValueError(f"template must contain '{{}}', got {self.template!r}")
        if not self.null_prompt.strip():
            raise ValueError("null_prompt must not be blank")
        if not math.isfinite(self.window) or self.window <= 0:
            raise ValueError(f"window must be positive and finite, got {self.window}")


@dataclass(frozen=True)
class RobustReference:
    """
    A centre and a spread that put scores on a z scale.

    ``of`` takes the median and 1.4826 times the median absolute deviation,
    which equals the standard deviation for normal data but ignores the few
    targets a mostly empty sample holds. A centre that is not finite, or a
    spread that is not positive and finite, RAISES ValueError.

    Parameters:
        - median (float): the reference sample's median
        - spread (float): its robust standard deviation
    """

    median: float
    spread: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.median):
            raise ValueError(f"median must be finite, got {self.median}")
        if not math.isfinite(self.spread) or self.spread <= 0:
            raise ValueError(f"spread must be positive and finite, got {self.spread}")

    @classmethod
    def of(cls, values: np.ndarray) -> RobustReference:
        """
        The robust centre and spread of a sample.

        An empty sample, or one whose median absolute deviation is zero,
        RAISES ValueError.

        Parameters:
            - values (np.ndarray): (N,) reference scores

        Return: the reference
        """
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            raise ValueError("a reference needs at least one value")
        median = float(np.median(values))
        return cls(median, MAD_TO_SIGMA * float(np.median(np.abs(values - median))))

    def z(self, values: np.ndarray) -> np.ndarray:
        """
        Scores as robust z-scores.

        Parameters:
            - values (np.ndarray): scores on the reference's scale

        Return: (values - median) / spread, float64
        """
        return (np.asarray(values, dtype=np.float64) - self.median) / self.spread


@dataclass(frozen=True, eq=False)
class QueryTerm:
    """
    A query compiled for scoring: one direction per phrase, and its scale on target-free tiles.

    Parameters:
        - phrases (tuple[str, ...]): the query phrases, before the template
        - directions (np.ndarray): (phrases, dim) phrase minus bare template, float32
        - reference (RobustReference): the query score's centre and spread on reference tiles
    """

    phrases: tuple[str, ...]
    directions: np.ndarray
    reference: RobustReference

    def scores(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Each tile's best phrase score.

        Parameters:
            - embeddings (np.ndarray): (tiles, dim) unit tile embeddings

        Return: (tiles,) the highest dot product with a phrase direction
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if len(embeddings) == 0:
            return np.zeros(0, dtype=np.float32)
        return (embeddings @ self.directions.T).max(axis=1)

    def z(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Each tile's query score as a robust z-score against the reference tiles.

        Parameters:
            - embeddings (np.ndarray): (tiles, dim) unit tile embeddings

        Return: (tiles,) z-scores, float64
        """
        return self.reference.z(self.scores(embeddings))


@dataclass(frozen=True, eq=False)
class TileTerms:
    """
    One scored frame's per-tile detail: what a backlog keeps to rescore later.

    Parameters:
        - embeddings (np.ndarray): (tiles, dim) unit tile embeddings, float32
        - query_z (np.ndarray): (tiles,) query z-scores
        - novelty_p (np.ndarray): (tiles,) novelty percentiles, 0.5 where novelty had no reference
        - scores (np.ndarray): (tiles,) raw scores, ``combine(query_z, novelty_p)``
    """

    embeddings: np.ndarray
    query_z: np.ndarray
    novelty_p: np.ndarray
    scores: np.ndarray


def combine(query_z: np.ndarray, novelty_p: np.ndarray) -> np.ndarray:
    """
    A tile's raw score: the mean of its query and novelty percentiles, kept apart at the top.

    The query z-score becomes a percentile as ``sigmoid(LOGISTIC_SLOPE z)``.
    The raw score is softplus of the mean percentile's log-odds, plus
    ``TIE_BREAK`` times a smooth ramp of the query z-score that never
    flattens: positive, strictly increasing in each term, and unbounded, so
    strong detections never tie the way a clamp, or a percentile above the
    reference's maximum, would.
    Complements are computed directly, so nothing cancels near 1. Arrays of
    different shapes, a z-score that is not finite, or a percentile outside
    [0, 1] RAISE ValueError.

    Parameters:
        - query_z (np.ndarray): query z-scores
        - novelty_p (np.ndarray): novelty percentiles in [0, 1], the same shape

    Return: raw scores, float64
    """
    query_z = np.asarray(query_z, dtype=np.float64)
    novelty_p = np.asarray(novelty_p, dtype=np.float64)
    if query_z.shape != novelty_p.shape:
        raise ValueError(f"shapes differ: {query_z.shape} and {novelty_p.shape}")
    if not np.isfinite(query_z).all():
        raise ValueError("query z-scores must be finite")
    if not ((novelty_p >= 0) & (novelty_p <= 1)).all():
        raise ValueError("novelty percentiles must lie in [0, 1]")
    slope = LOGISTIC_SLOPE * query_z
    with np.errstate(divide="ignore"):  # log(0) = -inf is exact here
        log_up = np.logaddexp(-np.logaddexp(0.0, -slope), np.log(novelty_p))
        log_down = np.logaddexp(-np.logaddexp(0.0, slope), np.log(1.0 - novelty_p))
    return np.logaddexp(0.0, log_up - log_down) + TIE_BREAK * _ramp(query_z)


def _ramp(z: np.ndarray) -> np.ndarray:
    """
    A smooth ramp: (z + sqrt(z^2 + 4)) / 2, positive and strictly increasing everywhere.

    It is about z for large z and about 1 / |z| for very negative z, so it
    never flattens into a constant the way softplus underflows to zero.
    Written as 2 / (sqrt(z^2 + 4) - z) below zero, which does not cancel.

    Parameters:
        - z (np.ndarray): finite values

    Return: the ramp, float64
    """
    root = np.sqrt(z * z + 4.0)
    return np.where(z > 0, (z + root) / 2.0, 2.0 / (root - z))


def retarget(embeddings: np.ndarray, novelty_p: np.ndarray, query: QueryTerm) -> np.ndarray:
    """
    Rescore stored tiles against a new query, keeping the novelty they had on arrival.

    One matrix product over the stored embeddings, no images: a backlog that
    kept ``TileTerms.embeddings`` and ``TileTerms.novelty_p`` re-ranks with this
    when the query changes. A frame's new raw score is its tiles' maximum.

    Parameters:
        - embeddings (np.ndarray): (tiles, dim) stored unit tile embeddings
        - novelty_p (np.ndarray): (tiles,) their stored novelty percentiles
        - query (QueryTerm): the new query, from ``ClipScorer.query_term``

    Return: (tiles,) raw scores
    """
    return combine(query.z(embeddings), novelty_p)


class ClipScorer:
    """
    Query-only CLIP tile scorer with window novelty; implements ``model.scorer.WindowedTileScorer``.

    Build one with ``load`` for a pretrained checkpoint. The constructor
    takes an already built model so tests can pass a tiny random one. It
    encodes the bare template once and MUTATES the model: moves it to the
    device, casts it to the requested precision, and sets eval mode. It
    scores only once it has reference embeddings, passed in or set by
    ``calibrate``; before that, scoring RAISES ValueError. ``score_frame``
    keeps state across frames, so drive one instance from one thread, in
    capture order. A non-positive batch size or history, or a minimum
    history outside [1, history], RAISES ValueError.

    Novelty is placed among a seeded sample of the flight's earlier
    novelty (``model.novelty.NoveltyScale``), so a run repeats exactly.

    Parameters:
        - model (Any): CLIP-style model with ``get_text_features`` and ``get_image_features``
        - tokenize (Callable): prompts to a mapping of input tensors
        - image_size (int): side length the vision tower expects
        - mean (Sequence[float]): per-channel normalisation mean
        - std (Sequence[float]): per-channel normalisation standard deviation
        - config (ClipConfig): the query, template and window, and the model id for saving
        - reference (np.ndarray | None): (tiles, dim) unit embeddings of target-free tiles
        - device (str | None): torch device, None to pick one
        - batch_size (int): tiles per forward pass
        - half (bool): embed in float16 instead of float32
        - history (int): earlier tiles of the flight sampled to place novelty
        - min_history (int): tiles seen before novelty counts
    """

    def __init__(
        self,
        model: Any,
        tokenize: Callable[[list[str]], Mapping[str, torch.Tensor]],
        image_size: int,
        mean: Sequence[float],
        std: Sequence[float],
        config: ClipConfig,
        reference: np.ndarray | None = None,
        device: str | None = None,
        batch_size: int = 32,
        half: bool = False,
        history: int = NOVELTY_HISTORY,
        min_history: int = NOVELTY_MIN_HISTORY,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        scale = NoveltyScale(history, min_history)  # validates both
        self.config = config
        self.device = pick_device(device)
        self.batch_size = batch_size
        self.half = half
        self._size = image_size
        self._mean = tuple(mean)
        self._std = tuple(std)
        self._tokenize = tokenize

        model = model.to(self.device).eval()
        self._model = model.half() if half else model.float()
        self._null = self._encode([config.null_prompt])[0]
        self._window = NoveltyWindow(config.window)
        self._scale = scale
        self._latest: float | None = None
        self._reference: np.ndarray | None = None
        self._query: QueryTerm | None = None
        if reference is not None:
            self._set_reference(reference)

    @classmethod
    def load(
        cls,
        config: ClipConfig | None = None,
        reference: np.ndarray | None = None,
        device: str | None = None,
        batch_size: int = 32,
        half: bool = False,
    ) -> ClipScorer:
        """
        Build a scorer from a pretrained Hugging Face checkpoint.

        The model, tokenizer and image normalisation all come from
        ``config.model``, downloading on first use.

        Parameters:
            - config (ClipConfig | None): None means the defaults
            - reference (np.ndarray | None): target-free tile embeddings; None until calibrated
            - device (str | None): torch device, None to pick one
            - batch_size (int): tiles per forward pass
            - half (bool): embed in float16 instead of float32

        Return: a scorer, ready to score once it has reference embeddings
        """
        from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

        config = config or ClipConfig()
        model = AutoModel.from_pretrained(config.model)
        tokenizer = AutoTokenizer.from_pretrained(config.model)
        processor = AutoImageProcessor.from_pretrained(config.model)

        def tokenize(prompts: list[str]) -> Mapping[str, torch.Tensor]:
            return tokenizer(prompts, padding=True, return_tensors="pt")

        return cls(
            model,
            tokenize,
            image_size=int(model.config.vision_config.image_size),
            mean=processor.image_mean,
            std=processor.image_std,
            config=config,
            reference=reference,
            device=device,
            batch_size=batch_size,
            half=half,
        )

    @property
    def reference(self) -> np.ndarray | None:
        """
        The target-free tile embeddings the query is scaled against, None before calibration.

        Return: (tiles, dim) float32 array, or None
        """
        return self._reference

    @property
    def query(self) -> QueryTerm:
        """
        The current query, compiled. Before calibration it RAISES ValueError.

        Return: the query term
        """
        if self._query is None:
            raise ValueError("the scorer has no reference tiles yet; calibrate it first")
        return self._query

    def embed(self, tiles: Sequence[np.ndarray]) -> np.ndarray:
        """
        Embed tiles with the image encoder.

        Parameters:
            - tiles (Sequence[np.ndarray]): uint8 RGB square tiles

        Return: (tiles, dim) unit embeddings, float32
        """
        import torch

        out: list[np.ndarray] = []
        with torch.no_grad():
            for batch in _batches(tiles, self.batch_size):
                pixels = tiles_to_tensor(batch, self._size, self._mean, self._std, self.device)
                if self.half:
                    pixels = pixels.half()
                image = _embedding(self._model.get_image_features(pixel_values=pixels)).float()
                out.append((image / image.norm(dim=-1, keepdim=True)).cpu().numpy())
        if not out:
            return np.zeros((0, len(self._null)), dtype=np.float32)
        return np.concatenate(out).astype(np.float32)

    def calibrate(self, tiles: Sequence[np.ndarray]) -> None:
        """
        Set the reference from target-free tiles, which also recompiles the query.

        No tiles, or tiles whose query scores do not vary, RAISE ValueError.

        Parameters:
            - tiles (Sequence[np.ndarray]): uint8 RGB square tiles known to hold no target

        Return: void
        """
        self._set_reference(self.embed(tiles))

    def query_term(self, phrases: Sequence[str]) -> QueryTerm:
        """
        Compile query phrases against the reference tiles, without changing the scorer.

        No phrases, or no reference yet, RAISE ValueError.

        Parameters:
            - phrases (Sequence[str]): e.g. ("a swimmer", "a person in the water")

        Return: the compiled query
        """
        if not phrases:
            raise ValueError("a query needs at least one phrase")
        if self._reference is None:
            raise ValueError("the scorer has no reference tiles yet; calibrate it first")
        text = self._encode([self.config.template.format(p) for p in phrases])
        directions = (text - self._null).astype(np.float32)
        partial = QueryTerm(tuple(phrases), directions, RobustReference(0.0, 1.0))
        return replace(partial, reference=RobustReference.of(partial.scores(self._reference)))

    def set_targets(self, phrases: Sequence[str]) -> None:
        """
        Switch to a new query; the novelty window is kept.

        Parameters:
            - phrases (Sequence[str]): the new query phrases

        Return: void
        """
        self._query = self.query_term(phrases)
        self.config = replace(self.config, targets=tuple(phrases))

    def score_tiles(self, tiles: Sequence[np.ndarray]) -> list[float]:
        """
        Score tiles by the query alone, novelty neutral; nothing is remembered.

        Parameters:
            - tiles (Sequence[np.ndarray]): uint8 RGB square tiles

        Return: one raw score per tile, in the order given
        """
        query_z = self.query.z(self.embed(tiles))
        return combine(query_z, np.full_like(query_z, 0.5)).tolist()

    def score_frame(self, tiles: Sequence[np.ndarray], timestamp: float) -> list[float]:
        """
        Score one frame's tiles by query and novelty, then remember them.

        Parameters:
            - tiles (Sequence[np.ndarray]): uint8 RGB square tiles of one frame
            - timestamp (float): the frame's capture time in seconds

        Return: one raw score per tile, in the order given
        """
        return self.terms(self.embed(tiles), timestamp).scores.tolist()

    def terms(self, embeddings: np.ndarray, timestamp: float) -> TileTerms:
        """
        Score one frame's already embedded tiles, then remember them.

        The core of ``score_frame``, for embeddings from elsewhere (a cache, a
        TensorRT engine). Novelty is measured before this frame joins the
        window, and placed among the novelty of tiles seen before it. A
        timestamp earlier than the last one starts afresh, as ``reset``
        does. Embeddings whose width differs from the reference's RAISE
        ValueError.

        Parameters:
            - embeddings (np.ndarray): (tiles, dim) unit tile embeddings of one frame
            - timestamp (float): the frame's capture time in seconds

        Return: the frame's per-tile embeddings, query z-scores, novelty percentiles, raw scores
        """
        query = self.query
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != query.directions.shape[1]:
            raise ValueError(
                f"embeddings must be (tiles, {query.directions.shape[1]}), "
                f"got shape {embeddings.shape}"
            )
        if self._latest is not None and timestamp < self._latest:
            self.reset()
        novelty = self._window.observe(embeddings, timestamp)
        self._latest = timestamp
        novelty_p = self._scale.percentile(novelty)
        self._scale.add(novelty)
        query_z = query.z(embeddings)
        return TileTerms(embeddings, query_z, novelty_p, combine(query_z, novelty_p))

    def reset(self) -> None:
        """
        Forget the novelty window and its scale, as at the start of a new flight.

        Return: void
        """
        self._window.reset()
        self._scale.reset()
        self._latest = None

    def _set_reference(self, reference: np.ndarray) -> None:
        """
        Keep target-free tile embeddings and compile the current query against them.

        A reference that is not a non-empty 2-D array as wide as the text
        embeddings RAISES ValueError.

        Parameters:
            - reference (np.ndarray): (tiles, dim) unit embeddings of target-free tiles

        Return: void
        """
        reference = np.asarray(reference, dtype=np.float32)
        if reference.ndim != 2 or len(reference) == 0 or reference.shape[1] != len(self._null):
            raise ValueError(
                f"reference must be (tiles, {len(self._null)}) with tiles > 0, "
                f"got shape {reference.shape}"
            )
        self._reference = reference
        self._query = self.query_term(self.config.targets)

    def _encode(self, prompts: list[str]) -> np.ndarray:
        """
        Embed prompts with the text encoder.

        Parameters:
            - prompts (list[str]): whole sentences

        Return: (prompts, dim) unit embeddings, float32
        """
        import torch

        inputs = {key: value.to(self.device) for key, value in self._tokenize(prompts).items()}
        with torch.no_grad():
            text = _embedding(self._model.get_text_features(**inputs)).float()
        return (text / text.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32)


def _embedding(output: Any) -> torch.Tensor:
    """
    The projected embedding from a ``get_*_features`` call.

    transformers 5 returns an output object whose ``pooler_output`` is the
    projected embedding; earlier versions return the tensor itself.

    Parameters:
        - output (Any): what ``get_text_features`` or ``get_image_features`` returned

    Return: tensor of shape (batch, projection dimension)
    """
    import torch

    return output if isinstance(output, torch.Tensor) else output.pooler_output


def _batches(tiles: Sequence[np.ndarray], size: int) -> Iterator[Sequence[np.ndarray]]:
    """
    Slice tiles into consecutive batches of at most ``size``.

    Parameters:
        - tiles (Sequence[np.ndarray]): tiles to slice
        - size (int): maximum tiles per batch

    Return: iterator of non-empty slices
    """
    for start in range(0, len(tiles), size):
        yield tiles[start : start + size]

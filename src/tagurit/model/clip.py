"""
Score tiles zero-shot with CLIP: how much a tile looks like a target described in words.

PatchCore asks "is this unlike the normal tiles I was fitted on", which
breaks wherever the drone meets terrain its bank never saw. CLIP asks "does
this look more like a person or a boat than like sea, road or forest",
which needs no fitting and nothing to go stale. A tile's score is the
softmax probability mass the model puts on the target prompts against the
background prompts, at the model's own learned temperature. Scores lie in
[0, 1]; priorities are still mapped by ``tagging.priority``.

Prompts are the whole model here: an environment with no fitting
background prompt pushes probability onto the targets. Models load through
Hugging Face ``transformers`` (``AutoModel``), so any CLIP-style checkpoint
with ``get_text_features`` and ``get_image_features`` works by name, and
weights download on first use. torch and transformers are imported inside
functions, so importing this module stays cheap.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from tagurit.model.device import pick_device
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


@dataclass(frozen=True)
class ClipConfig:
    """
    Everything that determines a CLIP scorer, saved in its bundle.

    Each prompt is placed into ``template`` before encoding. Empty target or
    background lists, or a template without ``{}``, RAISE ValueError.

    Parameters:
        - model (str): Hugging Face model id of a CLIP-style checkpoint
        - targets (tuple[str, ...]): things worth sending
        - backgrounds (tuple[str, ...]): scenery that is not
        - template (str): sentence each prompt is placed into
    """

    model: str = DEFAULT_MODEL
    targets: tuple[str, ...] = DEFAULT_TARGETS
    backgrounds: tuple[str, ...] = DEFAULT_BACKGROUNDS
    template: str = "an aerial photo of {}."

    def __post_init__(self) -> None:
        if not self.targets or not self.backgrounds:
            raise ValueError("CLIP needs at least one target and one background prompt")
        if "{}" not in self.template:
            raise ValueError(f"template must contain '{{}}', got {self.template!r}")


class ClipScorer:
    """
    Zero-shot CLIP tile scorer; implements ``model.scorer.TileScorer``.

    Build one with ``load`` for a pretrained checkpoint. The constructor
    takes an already built model so tests can pass a tiny random one. It
    encodes the prompts once and MUTATES the model: moves it to the device,
    casts it to the requested precision, and sets eval mode. Drive one
    instance from one thread.

    Parameters:
        - model (Any): CLIP-style model with ``get_text_features``,
          ``get_image_features`` and ``logit_scale``
        - tokenize (Callable): prompts to a mapping of input tensors
        - image_size (int): side length the vision tower expects
        - mean (Sequence[float]): per-channel normalisation mean
        - std (Sequence[float]): per-channel normalisation standard deviation
        - config (ClipConfig): the prompts, and the model id for saving
        - device (str | None): torch device, None to pick one
        - batch_size (int): tiles per forward pass
        - half (bool): score in float16 instead of float32
    """

    def __init__(
        self,
        model: Any,
        tokenize: Callable[[list[str]], Mapping[str, torch.Tensor]],
        image_size: int,
        mean: Sequence[float],
        std: Sequence[float],
        config: ClipConfig,
        device: str | None = None,
        batch_size: int = 32,
        half: bool = False,
    ) -> None:
        import torch

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.config = config
        self.device = pick_device(device)
        self.batch_size = batch_size
        self.half = half
        self._size = image_size
        self._mean = tuple(mean)
        self._std = tuple(std)

        model = model.to(self.device).eval()
        self._model = model.half() if half else model.float()
        prompts = [config.template.format(p) for p in (*config.targets, *config.backgrounds)]
        inputs = {key: value.to(self.device) for key, value in tokenize(prompts).items()}
        with torch.no_grad():
            text = _embedding(self._model.get_text_features(**inputs)).float()
            self._scale = float(self._model.logit_scale.exp())
        self._text = text / text.norm(dim=-1, keepdim=True)

    @classmethod
    def load(
        cls,
        config: ClipConfig | None = None,
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
            - device (str | None): torch device, None to pick one
            - batch_size (int): tiles per forward pass
            - half (bool): score in float16 instead of float32

        Return: a scorer ready to score
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
            device=device,
            batch_size=batch_size,
            half=half,
        )

    def score_tiles(self, tiles: Sequence[np.ndarray]) -> list[float]:
        """
        Score each tile by the probability the model gives the target prompts.

        Parameters:
            - tiles (Sequence[np.ndarray]): uint8 RGB square tiles

        Return: one score in [0, 1] per tile, in the order given
        """
        import torch

        targets = len(self.config.targets)
        scores: list[float] = []
        with torch.no_grad():
            for batch in _batches(tiles, self.batch_size):
                pixels = tiles_to_tensor(batch, self._size, self._mean, self._std, self.device)
                if self.half:
                    pixels = pixels.half()
                image = _embedding(self._model.get_image_features(pixel_values=pixels)).float()
                image = image / image.norm(dim=-1, keepdim=True)
                probs = (self._scale * image @ self._text.T).softmax(dim=-1)
                scores.extend(float(p) for p in probs[:, :targets].sum(dim=-1).cpu())
        return scores


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

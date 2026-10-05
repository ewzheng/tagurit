"""
Score tiles with PatchCore, using anomalib's implementation.

Fitting runs a frozen ImageNet backbone over normal tiles, keeps every patch
embedding, and shrinks them by greedy coreset selection into a memory bank.
A tile's score is its largest nearest-neighbour distance to that bank,
reweighted by anomalib by how dense the bank is around the match. No
gradient training happens anywhere.

Fitting is the expensive part: every embedding stays in memory until the
coreset is chosen. Fit on a laptop or the cloudlet and ship the saved bank;
the scout only loads and scores. Only the bank and the configuration are
saved; the backbone's pretrained weights come from timm's cache on load.

Half precision is a scoring choice, not part of the bank: fitting always
runs in float32, and ``half`` casts the backbone and bank afterwards. Scores
move by well under one percent, and scoring speeds up on GPUs with fast
float16 paths, such as the Jetson's tensor cores.

Scores are RAW distances. anomalib's Lightning wrapper and its min-max
normaliser are bypassed on purpose: clamping into [0, 1] would turn strong
detections into ties and erase the ranking the scheduler relies on.

torch and anomalib are imported inside functions, so importing this module
is cheap and works without the ``model`` extra installed.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from tagurit.model.device import pick_device
from tagurit.model.preprocess import IMAGENET_MEAN, IMAGENET_STD, tiles_to_tensor

if TYPE_CHECKING:
    import torch
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel

FORMAT_VERSION = 1


@dataclass(frozen=True)
class PatchcoreConfig:
    """
    Everything that determines a PatchCore bank, saved alongside it.

    The defaults follow anomalib's: a WideResNet-50 backbone, layer2 and
    layer3 features, 256px inputs, and a 10% coreset. Invalid values RAISE
    ValueError.

    Parameters:
        - backbone (str): timm model name
        - layers (tuple[str, ...]): backbone stages whose features are concatenated
        - input_size (int): side length tiles are resized to before the backbone
        - coreset_ratio (float): fraction of patch embeddings kept in the bank, in (0, 1]
        - num_neighbors (int): bank neighbours used to reweight the top patch;
          1 makes the score the plain largest patch distance
        - pre_trained (bool): load ImageNet weights; False only in offline tests
    """

    backbone: str = "wide_resnet50_2"
    layers: tuple[str, ...] = ("layer2", "layer3")
    input_size: int = 256
    coreset_ratio: float = 0.1
    num_neighbors: int = 9
    pre_trained: bool = True

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("at least one backbone layer is required")
        if self.input_size <= 0:
            raise ValueError(f"input_size must be positive, got {self.input_size}")
        if not 0.0 < self.coreset_ratio <= 1.0:
            raise ValueError(f"coreset_ratio must be in (0, 1], got {self.coreset_ratio}")
        if self.num_neighbors < 1:
            raise ValueError(f"num_neighbors must be at least 1, got {self.num_neighbors}")


class PatchcoreScorer:
    """
    A fitted PatchCore model that scores tiles; implements ``model.scorer.TileScorer``.

    Build one with ``fit`` or ``load``. The constructor only wraps a model
    whose bank is already filled, and RAISES ValueError on an empty bank. It
    MUTATES the model: casts it to the requested precision and sets eval
    mode. Drive one instance from one thread: torch work on a device should
    not interleave from several.

    Parameters:
        - model (PatchcoreModel): anomalib model with a filled memory bank
        - config (PatchcoreConfig): the configuration the bank was fitted with
        - device (str): torch device the model lives on
        - batch_size (int): tiles per forward pass
        - half (bool): score in float16 instead of float32
    """

    def __init__(
        self,
        model: PatchcoreModel,
        config: PatchcoreConfig,
        device: str,
        batch_size: int = 16,
        half: bool = False,
    ) -> None:
        if model.memory_bank.numel() == 0:
            raise ValueError("the model's memory bank is empty; fit or load one first")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self._model = (model.half() if half else model.float()).eval()
        self.config = config
        self.device = device
        self.batch_size = batch_size
        self.half = half

    @classmethod
    def fit(
        cls,
        tiles: Iterable[np.ndarray],
        config: PatchcoreConfig | None = None,
        device: str | None = None,
        batch_size: int = 16,
        seed: int = 0,
        half: bool = False,
    ) -> PatchcoreScorer:
        """
        Fit a memory bank on tiles that are ALL normal.

        Greedy coreset selection keeps the most distant embeddings first, so a
        single target that slips into ``tiles`` lands in the bank and scores
        as normal from then on. Curate the input. Every patch embedding is
        held in device memory until the coreset is chosen: about 6 MB per
        256px tile with the default backbone. No tiles RAISES ValueError.

        Parameters:
            - tiles (Iterable[np.ndarray]): uint8 RGB square tiles, consumed once
            - config (PatchcoreConfig | None): None means the defaults
            - device (str | None): torch device, None to pick one
            - batch_size (int): tiles per forward pass
            - seed (int): seeds the coreset's random starting point
            - half (bool): score in float16 once fitted; fitting stays float32

        Return: a scorer holding the fitted bank
        """
        import torch

        config = config or PatchcoreConfig()
        device = pick_device(device)
        torch.manual_seed(seed)
        model = _build_model(config).to(device)
        model.train()

        count = 0
        with torch.no_grad():
            for batch in _batches(tiles, batch_size):
                model(_imagenet_tensor(batch, config.input_size, device))
                count += len(batch)
            if count == 0:
                raise ValueError("no tiles to fit on")
            model.subsample_embedding(config.coreset_ratio)
        return cls(model, config, device, batch_size, half)

    @classmethod
    def load(
        cls, path: Path, device: str | None = None, batch_size: int = 16, half: bool = False
    ) -> PatchcoreScorer:
        """
        Load a bank written by ``save`` and rebuild the backbone around it.

        With ``pre_trained`` set, the backbone's ImageNet weights come from
        timm's cache, downloading on first use. A file of another format
        version RAISES ValueError.

        Parameters:
            - path (Path): file written by ``save``
            - device (str | None): torch device, None to pick one
            - batch_size (int): tiles per forward pass
            - half (bool): score in float16 instead of float32

        Return: a scorer ready to score
        """
        import torch

        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("format") != FORMAT_VERSION:
            raise ValueError(f"{path} is format {saved.get('format')}, expected {FORMAT_VERSION}")
        fields = dict(saved["config"])
        fields["layers"] = tuple(fields["layers"])
        config = PatchcoreConfig(**fields)
        device = pick_device(device)
        model = _build_model(config)
        model.memory_bank = saved["memory_bank"]
        return cls(model.to(device), config, device, batch_size, half)

    def save(self, path: Path) -> None:
        """
        Write the bank and configuration, NOT the backbone weights.

        The bank is always written as float32, so one file serves either precision.

        Parameters:
            - path (Path): destination file, conventionally ``*.pt``

        Return: void
        """
        import torch

        torch.save(
            {
                "format": FORMAT_VERSION,
                "config": asdict(self.config),
                "memory_bank": self._model.memory_bank.detach().float().cpu(),
            },
            path,
        )

    @property
    def bank_size(self) -> int:
        """
        Number of patch embeddings in the memory bank.

        Return: row count of the bank
        """
        return int(self._model.memory_bank.shape[0])

    def score_tiles(self, tiles: Sequence[np.ndarray]) -> list[float]:
        """
        Score each tile against the bank.

        Parameters:
            - tiles (Sequence[np.ndarray]): uint8 RGB square tiles

        Return: one raw, non-negative score per tile, in the order given
        """
        import torch

        scores: list[float] = []
        with torch.no_grad():
            for batch in _batches(tiles, self.batch_size):
                result = self._model(_imagenet_tensor(batch, self.config.input_size, self.device))
                scores.extend(float(score) for score in result.pred_score.cpu())
        return scores


def _build_model(config: PatchcoreConfig) -> PatchcoreModel:
    """
    Construct anomalib's PatchCore module with an empty bank.

    Parameters:
        - config (PatchcoreConfig): backbone, layers, and neighbour count to use

    Return: an untrained PatchcoreModel on the CPU
    """
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel

    return PatchcoreModel(
        layers=list(config.layers),
        backbone=config.backbone,
        pre_trained=config.pre_trained,
        num_neighbors=config.num_neighbors,
    )


def _batches(tiles: Iterable[np.ndarray], size: int) -> Iterator[list[np.ndarray]]:
    """
    Group tiles into lists of at most ``size``, the last possibly shorter.

    Parameters:
        - tiles (Iterable[np.ndarray]): tiles to group, consumed once
        - size (int): maximum tiles per group

    Return: iterator of non-empty lists
    """
    batch: list[np.ndarray] = []
    for tile in tiles:
        batch.append(tile)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _imagenet_tensor(tiles: Sequence[np.ndarray], size: int, device: str) -> torch.Tensor:
    """
    Resize tiles and normalise them with ImageNet statistics, as the backbone expects.

    Parameters:
        - tiles (Sequence[np.ndarray]): uint8 RGB square tiles
        - size (int): backbone input side length
        - device (str): torch device for the result

    Return: float tensor of shape (len(tiles), 3, size, size)
    """
    return tiles_to_tensor(tiles, size, IMAGENET_MEAN, IMAGENET_STD, device)

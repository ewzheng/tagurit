"""
Experiment: how background prompts hold up when the terrain is not what the list expects.

Usage:
    uv run python scripts/experiment_clipnn_robust.py
    uv run python scripts/experiment_clipnn_robust.py --model openai/clip-vit-base-patch32

Tiles are scored by the softmax share of the target prompts against a set of
background prompts, as the pipeline's CLIP tagger does. Three background sets:

- full: the pipeline's 11 generic background phrases
- ablated: the same list with the dataset's "home terrain" phrases removed
  (sea and waves for SeaDronesSee; road, parking lot, buildings and rooftops
  for VisDrone), standing in for terrain the list did not anticipate
- selected: a broad scenery vocabulary (``VOCABULARY``, about 100 phrases,
  written without reference to these datasets) shipped once; before each
  frame the scout picks the ``--select`` phrases that best describe its own
  recent tiles, by votes of each tile's nearest phrase

Each set is also scored with neighbour contrast ("+knn"): the tile's score
minus the mean score of its ``--k`` nearest tiles in a rolling bank of the
last ``--window`` frames, so terrain that always scores a little high cancels.
Frames replay in time order with the bank warm-started from just before the
held-out part. A frame scores its best tile.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

from tagurit.model.clip import ClipConfig
from tagurit.model.device import pick_device
from tagurit.model.preprocess import tiles_to_tensor
from tagurit.model.tiling import crop_tiles, decode_image, tile_grid
from tagurit.sim import dataloader
from tagurit.sim.metrics import auroc, average_precision, recall_at_fraction

DEFAULT_MODEL = "openai/clip-vit-base-patch16"
DONT_CARE = "ignored"
TOP_FRACTIONS = (0.01, 0.05, 0.10)
HOME_TERRAIN = {
    "sparse-seadronessee": {"the open sea", "waves on water"},
    "sparse-visdrone": {"a road", "a parking lot", "buildings", "rooftops"},
}
VOCABULARY = (
    # water and coast
    "the open sea", "waves on water", "calm water", "a lake", "a river", "a pond",
    "a canal", "a coastline", "a beach", "wet sand", "foam on water", "sun glare on water",
    "a swamp", "a marsh", "reeds", "ice", "snow", "a frozen lake", "a glacier",
    # natural ground
    "grass", "a meadow", "a field", "farmland", "crops", "plowed soil", "dirt", "mud",
    "gravel", "rocks", "boulders", "a cliff", "a mountain slope", "a hillside", "a valley",
    "a canyon", "sand", "sand dunes", "a desert", "dry scrubland", "bushes", "shrubs",
    "trees", "a forest", "a pine forest", "tree canopy", "fallen leaves", "moss",
    "a dry riverbed", "volcanic rock",
    # built environment
    "a road", "an asphalt road", "a highway", "a street", "a sidewalk", "a parking lot",
    "a crosswalk", "road markings", "a roundabout", "a bridge", "a railway",
    "train tracks", "a building", "buildings", "rooftops", "a flat roof", "solar panels",
    "a construction site", "concrete", "a plaza", "a courtyard", "a sports field",
    "a basketball court", "a tennis court", "a running track", "a stadium", "a garden",
    "a park", "a lawn", "a fence", "a wall", "a playground", "an industrial area",
    "a warehouse", "a pier", "a dam", "power lines", "a tower", "a residential area",
    "houses", "a village", "a suburb", "an airport runway", "a gas station",
    # image conditions
    "shadows", "a blurry image", "darkness", "fog", "haze", "an overexposed image",
    "a black area",
)  # fmt: skip


class Encoder:
    """
    CLIP image and text embeddings, unit length, as float32 arrays.

    Parameters:
        - model_id (str): Hugging Face CLIP checkpoint
        - device (str): torch device
        - batch_size (int): tiles per forward pass
    """

    def __init__(self, model_id: str, device: str, batch_size: int = 64) -> None:
        self.device, self.batch_size = device, batch_size
        self.model = AutoModel.from_pretrained(model_id).to(device).half().eval()
        processor = AutoImageProcessor.from_pretrained(model_id)
        self.mean, self.std = processor.image_mean, processor.image_std
        self.size = int(self.model.config.vision_config.image_size)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        with torch.no_grad():
            self.scale = float(self.model.logit_scale.exp())

    @staticmethod
    def _unit(output: object) -> np.ndarray:
        t = output if isinstance(output, torch.Tensor) else output.pooler_output  # type: ignore[attr-defined]
        t = t.float()
        return (t / t.norm(dim=-1, keepdim=True)).cpu().numpy()

    def images(self, tiles: list[np.ndarray]) -> np.ndarray:
        """
        Embed tiles.

        Parameters:
            - tiles (list[np.ndarray]): uint8 RGB square tiles

        Return: (len(tiles), dim) unit embeddings
        """
        out = []
        with torch.no_grad():
            for start in range(0, len(tiles), self.batch_size):
                batch = tiles[start : start + self.batch_size]
                pixels = tiles_to_tensor(batch, self.size, self.mean, self.std, self.device)
                out.append(self._unit(self.model.get_image_features(pixel_values=pixels.half())))
        return np.concatenate(out)

    def texts(self, prompts: list[str]) -> np.ndarray:
        """
        Embed prompts.

        Parameters:
            - prompts (list[str]): sentences

        Return: (len(prompts), dim) unit embeddings
        """
        tokens = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            return self._unit(self.model.get_text_features(**tokens))


def target_share(
    tiles: np.ndarray, targets: np.ndarray, background: np.ndarray, scale: float
) -> np.ndarray:
    """
    Softmax share of the target prompts, against the background prompts, per tile.

    Parameters:
        - tiles (np.ndarray): (T, dim) tile embeddings
        - targets (np.ndarray): (P, dim) target prompt embeddings
        - background (np.ndarray): (Q, dim) background prompt embeddings
        - scale (float): CLIP's logit scale

    Return: (T,) scores in [0, 1]
    """
    logits = scale * tiles @ np.concatenate([targets, background]).T
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs[:, : len(targets)].sum(axis=1)


def report(name: str, dataset: str, scores: list[float], labels: list[bool]) -> None:
    """
    Print one metrics row.

    Parameters:
        - name (str): row label
        - dataset (str): dataset name
        - scores (list[float]): one score per frame
        - labels (list[bool]): whether each frame holds a target

    Return: void
    """
    recalls = "  ".join(f"{recall_at_fraction(scores, labels, f):6.3f}" for f in TOP_FRACTIONS)
    print(
        f"  {name:14s} {dataset:20s} {len(labels):5d} {sum(labels):4d}  "
        f"{auroc(scores, labels):.3f}  {average_precision(scores, labels):.3f}  {recalls}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    """
    Replay held-out frames and compare background sets with and without neighbour contrast.

    Parameters:
        - argv (list[str] | None): arguments without the program name; None means sys.argv

    Return: void
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--datasets", nargs="+", default=["sparse-seadronessee", "sparse-visdrone"])
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face CLIP checkpoint")
    parser.add_argument("--tile", type=int, default=192, help="tile side in frame pixels")
    parser.add_argument("--window", type=int, default=50, help="frames in the rolling bank")
    parser.add_argument("--k", type=int, default=10, help="neighbours for the contrast")
    parser.add_argument("--select", type=int, default=12, help="phrases the scout selects")
    parser.add_argument("--fit-fraction", type=float, default=0.7)
    parser.add_argument("--device")
    args = parser.parse_args(argv)

    encoder = Encoder(args.model, pick_device(args.device))
    config = ClipConfig()
    say = config.template.format
    targets = encoder.texts([say(p) for p in config.targets])
    vocabulary = encoder.texts([say(p) for p in VOCABULARY])
    print(
        f"{args.model}, {args.tile}px tiles, bank {args.window} frames, k={args.k}, "
        f"{args.select} of {len(VOCABULARY)} phrases selected"
    )

    def tiles_of(frame: object) -> np.ndarray:
        image = decode_image(frame.read())  # type: ignore[attr-defined]
        grid = tile_grid(image.shape[1], image.shape[0], args.tile)
        return encoder.images(crop_tiles(image, grid))

    header = "  ".join(f"top{100 * f:g}%" for f in TOP_FRACTIONS)
    print(f"  {'background':14s} {'dataset':20s} frames tgts  AUROC     AP  {header}")
    for dataset in args.datasets:
        full = [p for p in config.backgrounds]
        ablated = [p for p in full if p not in HOME_TERRAIN.get(dataset, set())]
        fixed = {"full": encoder.texts([say(p) for p in full]),
                 "ablated": encoder.texts([say(p) for p in ablated])}  # fmt: skip
        scores: dict[str, list[float]] = {}
        labels: list[bool] = []
        picked: Counter[str] = Counter()
        for sequence in dataloader.sequences(dataset):
            lead, trail = dataloader.load(dataset, sequence).split(args.fit_fraction)
            bank: deque[np.ndarray] = deque(maxlen=args.window)
            for frame in lead[-args.window :]:
                bank.append(tiles_of(frame))
            for frame in trail:
                label = frame.has_target((DONT_CARE,))
                tiles = tiles_of(frame)
                if label is not None and bank:
                    reference = np.concatenate(bank)
                    votes = Counter((reference @ vocabulary.T).argmax(axis=1).tolist())
                    chosen = [i for i, _ in votes.most_common(args.select)]
                    picked.update(VOCABULARY[i] for i in chosen)
                    sets = {**fixed, "selected": vocabulary[chosen]}
                    k = min(args.k, len(reference))
                    nearest = np.argpartition(-(tiles @ reference.T), k - 1, axis=1)[:, :k]
                    for name, background in sets.items():
                        tile_score = target_share(tiles, targets, background, encoder.scale)
                        ref_score = target_share(reference, targets, background, encoder.scale)
                        contrast = tile_score - ref_score[nearest].mean(axis=1)
                        scores.setdefault(name, []).append(float(tile_score.max()))
                        scores.setdefault(f"{name}+knn", []).append(float(contrast.max()))
                    labels.append(label)
                bank.append(tiles)
        for name, values in scores.items():
            report(name, dataset, values, labels)
        common = ", ".join(phrase for phrase, _ in picked.most_common(8))
        print(f"  most often selected for {dataset}: {common}", flush=True)


if __name__ == "__main__":
    main()

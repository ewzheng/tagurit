"""
Experiment: score tiles against target text with the scout's own recent tiles as background.

Usage:
    uv run python scripts/experiment_clipnn_selfbg.py
    uv run python scripts/experiment_clipnn_selfbg.py --tile 256 --window 100 --k 20
    uv run python scripts/experiment_clipnn_selfbg.py --model openai/clip-vit-base-patch32

Frames of each sequence are replayed in time order. Every tile is embedded
once with a CLIP model (``--model``, ViT-B/16 by default); a rolling bank
holds the tile embeddings of the last ``--window`` frames, warm-started from
the frames just before the held-out part, as if the scout had flown the
earlier stretch. Only target prompts are used.
For each tile, with s = its cosine similarity to a target prompt (the best
prompt is taken at the end):

- raw: s, with no normalisation
- standardised: (s - bank mean of s) / bank standard deviation of s
- knn: s minus the mean s of the tile's ``--k`` nearest bank tiles, found by
  image similarity, so a tile is judged against tiles that look like it
- prompts: the pipeline's softmax over target and hand-written background
  prompts, for reference

Image-to-image similarities are far higher than image-to-text ones in CLIP
(the "modality gap"), so bank tiles are never mixed into the text softmax;
they only define what is normal for the text score. A frame scores its best
tile. Prints frame-level AUROC, average precision and top-fraction recall.
"""

from __future__ import annotations

import argparse
from collections import deque

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


def frame_tiles(encoder: Encoder, frame: object, size: int) -> np.ndarray:
    """
    Embed every tile of one frame.

    Parameters:
        - encoder (Encoder): the CLIP model
        - frame (TraceFrame): the frame
        - size (int): tile side in frame pixels

    Return: (tiles, dim) unit embeddings
    """
    image = decode_image(frame.read())  # type: ignore[attr-defined]
    return encoder.images(crop_tiles(image, tile_grid(image.shape[1], image.shape[0], size)))


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
    Replay each dataset's held-out frames and compare the background references.

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
    parser.add_argument("--window", type=int, default=50, help="frames in the background bank")
    parser.add_argument("--k", type=int, default=10, help="neighbours for the knn score")
    parser.add_argument("--fit-fraction", type=float, default=0.7)
    parser.add_argument("--device")
    args = parser.parse_args(argv)

    encoder = Encoder(args.model, pick_device(args.device))
    print(f"{args.model}, {args.tile}px tiles, bank of {args.window} frames, k={args.k}")
    config = ClipConfig()
    targets = encoder.texts([config.template.format(p) for p in config.targets])
    everything = encoder.texts(
        [config.template.format(p) for p in (*config.targets, *config.backgrounds)]
    )
    n_targets = len(config.targets)

    header = "  ".join(f"top{100 * f:g}%" for f in TOP_FRACTIONS)
    print(f"  {'score':14s} {'dataset':20s} frames tgts  AUROC     AP  {header}")
    for dataset in args.datasets:
        scores: dict[str, list[float]] = {"raw": [], "standardised": [], "knn": [], "prompts": []}
        labels: list[bool] = []
        for sequence in dataloader.sequences(dataset):
            lead, trail = dataloader.load(dataset, sequence).split(args.fit_fraction)
            bank: deque[np.ndarray] = deque(maxlen=args.window)
            for frame in lead[-args.window :]:
                bank.append(frame_tiles(encoder, frame, args.tile))
            for frame in trail:
                label = frame.has_target((DONT_CARE,))
                tiles = frame_tiles(encoder, frame, args.tile)
                if label is not None and bank:
                    reference = np.concatenate(bank)  # (B, dim)
                    s = tiles @ targets.T  # (T, P) tile-to-target similarity
                    ref_s = reference @ targets.T  # (B, P)
                    standard = (s - ref_s.mean(axis=0)) / (ref_s.std(axis=0) + 1e-6)
                    k = min(args.k, len(reference))
                    nearest = np.argpartition(-(tiles @ reference.T), k - 1, axis=1)[:, :k]
                    knn = s - ref_s[nearest].mean(axis=1)  # (T, P)
                    logits = encoder.scale * tiles @ everything.T
                    logits -= logits.max(axis=1, keepdims=True)
                    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
                    scores["raw"].append(float(s.max()))
                    scores["standardised"].append(float(standard.max()))
                    scores["knn"].append(float(knn.max()))
                    scores["prompts"].append(float(probs[:, :n_targets].sum(axis=1).max()))
                    labels.append(label)
                bank.append(tiles)
        for name, values in scores.items():
            report(name, dataset, values, labels)


if __name__ == "__main__":
    main()

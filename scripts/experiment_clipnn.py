"""
Experiment: score frames by patch-level text similarity ("clipnn", phase 1).

Usage:
    uv run python scripts/experiment_clipnn.py
    uv run python scripts/experiment_clipnn.py --methods dense224 owlv2 --max-empty 300

Compares, on the held-out part of the sparse streams (``Trace.split`` at
``--fit-fraction``, as every tagger bundle uses), several ways of asking
"does this frame contain one of the target prompts":

- tile: the pipeline's zero-shot CLIP, one embedding per 256px tile
- dense224 / dense448: CLIP ViT-B/16 with a ClearCLIP-style last block
  (self-self attention, no residual or feed-forward) and SegEarth-OV's
  subtraction of 0.3 x the [CLS] embedding, scoring every 16px patch of
  224px or 448px tiles; the 448px tiles interpolate CLIP's position embeddings
- owlv2: OWLv2 base/16 at 960px, scoring every patch with its class head
- siglip2: SigLIP 2 base/16 at 256px, one embedding per 256px tile
- tips_tile: TIPSv2 B/14, one global embedding per 224px tile
- tips224 / tips448: TIPSv2 B/14 patch tokens, trained to align with text,
  scoring every 14px patch of 224px or 448px tiles

Tile models (clip, siglip2, tips_tile) run at every ``--tiles`` size, cut at
native scale and resized to the model's input; smaller tiles dilute a small
target less. Every method also reports "+frame": the best tile's score minus
the frame's median tile, which asks whether one tile stands out from the rest
of its frame. ``--max-empty N`` keeps every target frame and N empty frames
spread evenly across all sequences, for quick screening; absolute numbers
then differ from a full run because targets are a larger share.

Patch methods also report two other ways to turn patch scores into a tile
score, from the same forward pass: the mean of the top 3 patches ("+top3")
and the best patch minus the tile's median patch ("+contrast"), which
cancels effects such as sun glint that lift every patch of a tile at once.

Each method is a ``TileScorer``, so ``tagging.Tagger`` tiles the frame and a
frame scores its best tile, as in the pipeline; for the patch methods a tile
scores its best patch. Text prompts are ``ClipConfig``'s defaults, the same
targets and backgrounds the pipeline's CLIP tagger uses. Prints frame-level
AUROC, average precision, top-fraction recall and milliseconds per frame.
Nothing in the package changes.
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Sequence

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer, Owlv2ForObjectDetection

from tagurit.model.clip import ClipConfig, ClipScorer
from tagurit.model.device import pick_device
from tagurit.model.preprocess import tiles_to_tensor
from tagurit.sim import dataloader
from tagurit.sim.metrics import auroc, average_precision, recall_at_fraction
from tagurit.sim.trace import TraceFrame
from tagurit.tagging import Tagger
from tagurit.tagging.priority import PriorityScale

CLIP_MODEL = "openai/clip-vit-base-patch16"
OWL_MODEL = "google/owlv2-base-patch16-ensemble"
DONT_CARE = "ignored"
TOP_FRACTIONS = (0.01, 0.05, 0.10)
CLS_WEIGHT = 0.3  # SegEarth-OV's global-bias subtraction


def _tensor(output: object) -> torch.Tensor:
    """
    The embedding tensor from a transformers feature call, which may return an output object.

    Parameters:
        - output (object): a tensor or an output with ``pooler_output``

    Return: the tensor
    """
    return output if isinstance(output, torch.Tensor) else output.pooler_output  # type: ignore[attr-defined]


def aggregate(target: torch.Tensor) -> dict[str, list[float]]:
    """
    Tile scores from per-patch target probabilities, three ways.

    Parameters:
        - target (torch.Tensor): (tiles, patches) probability that each patch is a target

    Return: {"max", "top3", "contrast"} -> one score per tile
    """
    top3 = target.topk(min(3, target.shape[1]), dim=1).values.mean(dim=1)
    contrast = target.amax(dim=1) - target.median(dim=1).values
    return {
        "max": [float(v) for v in target.amax(dim=1).cpu()],
        "top3": [float(v) for v in top3.cpu()],
        "contrast": [float(v) for v in contrast.cpu()],
    }


class TipsScorer:
    """
    Score tiles with TIPSv2 B/14: per patch (dense) or by its global embedding.

    TIPSv2 takes pixels in [0, 1] without normalisation. Its output tokens are
    one global token, one register token, then the patches; patches and text
    share one embedding space, compared at the model's temperature.

    Parameters:
        - config (ClipConfig): prompts and template
        - size (int): input side length; the model was trained at 448
        - dense (bool): score patches rather than the global embedding
        - device (str): torch device
        - batch_size (int): tiles per forward pass
    """

    MODEL = "google/tipsv2-b14"
    PREFIX = 2  # global token and one register token

    def __init__(
        self, config: ClipConfig, size: int, dense: bool, device: str, batch_size: int = 16
    ) -> None:
        self.targets = len(config.targets)
        self.size, self.dense, self.device, self.batch_size = size, dense, device, batch_size
        self.model = AutoModel.from_pretrained(self.MODEL).to(device).half().eval()
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL)
        prompts = [config.template.format(p) for p in (*config.targets, *config.backgrounds)]
        tokens = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            text = _tensor(self.model.get_text_features(**tokens)).float()
            self.scale = 1.0 / float(self.model.temperature)
        self.text = text / text.norm(dim=-1, keepdim=True)
        self.extra: dict[str, float] = {}

    def score_tiles(self, tiles: Sequence[np.ndarray]) -> list[float]:
        """
        Score each tile; for dense scoring also keep the other aggregations in ``extra``.

        Parameters:
            - tiles (Sequence[np.ndarray]): uint8 RGB square tiles

        Return: one score in [0, 1] per tile, by the best patch when dense
        """
        tile_scores: dict[str, list[float]] = {"max": [], "top3": [], "contrast": []}
        with torch.no_grad():
            for start in range(0, len(tiles), self.batch_size):
                batch = tiles[start : start + self.batch_size]
                pixels = tiles_to_tensor(batch, self.size, (0, 0, 0), (1, 1, 1), self.device)
                out = self.model.get_image_features(pixel_values=pixels.half())
                if self.dense:
                    feats = out.last_hidden_state[:, self.PREFIX :].float()
                else:
                    feats = out.pooler_output.float()[:, None, :]
                feats = feats / feats.norm(dim=-1, keepdim=True)
                probs = (self.scale * feats @ self.text.T).softmax(dim=-1)
                for name, values in aggregate(probs[..., : self.targets].sum(dim=-1)).items():
                    tile_scores[name].extend(values)
        if self.dense:
            self.extra = {name: max(values) for name, values in tile_scores.items() if values}
        return tile_scores["max"]


class Siglip2TileScorer:
    """
    Score each tile by SigLIP 2's global embedding against the prompts.

    By default a softmax over target and background prompts is used, as for
    CLIP. With ``sigmoid`` it uses SigLIP's own scoring instead: each target
    prompt gets an independent probability from the model's learned scale and
    bias, a tile scores its most likely target, and background prompts play
    no part, so the query list can change without shifting other scores.

    Parameters:
        - config (ClipConfig): prompts and template
        - device (str): torch device
        - sigmoid (bool): score targets independently with SigLIP's sigmoid
        - batch_size (int): tiles per forward pass
    """

    MODEL = "google/siglip2-base-patch16-256"

    def __init__(
        self, config: ClipConfig, device: str, sigmoid: bool = False, batch_size: int = 32
    ) -> None:
        self.targets = len(config.targets)
        self.device, self.sigmoid, self.batch_size = device, sigmoid, batch_size
        self.model = AutoModel.from_pretrained(self.MODEL).to(device).half().eval()
        processor = AutoImageProcessor.from_pretrained(self.MODEL)
        self.mean, self.std = processor.image_mean, processor.image_std
        self.size = int(self.model.config.vision_config.image_size)
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL)
        prompts = [config.template.format(p) for p in (*config.targets, *config.backgrounds)]
        tokens = tokenizer(prompts, padding="max_length", max_length=64, return_tensors="pt")
        with torch.no_grad():
            text = _tensor(self.model.get_text_features(**tokens.to(device))).float()
            self.scale = float(self.model.logit_scale.exp())
            self.bias = float(self.model.logit_bias)
        self.text = text / text.norm(dim=-1, keepdim=True)

    def score_tiles(self, tiles: Sequence[np.ndarray]) -> list[float]:
        """
        Score each tile by its probability of showing a target.

        Parameters:
            - tiles (Sequence[np.ndarray]): uint8 RGB square tiles

        Return: one score in [0, 1] per tile
        """
        scores: list[float] = []
        with torch.no_grad():
            for start in range(0, len(tiles), self.batch_size):
                batch = tiles[start : start + self.batch_size]
                pixels = tiles_to_tensor(batch, self.size, self.mean, self.std, self.device)
                image = _tensor(self.model.get_image_features(pixel_values=pixels.half())).float()
                image = image / image.norm(dim=-1, keepdim=True)
                logits = self.scale * image @ self.text.T
                if self.sigmoid:
                    target = (logits[:, : self.targets] + self.bias).sigmoid().amax(dim=-1)
                else:
                    target = logits.softmax(dim=-1)[:, : self.targets].sum(dim=-1)
                scores.extend(float(v) for v in target.cpu())
        return scores


class DenseClipScorer:
    """
    Score each tile by its most target-like patch, using CLIP with a ClearCLIP last block.

    Parameters:
        - config (ClipConfig): prompts and template
        - size (int): input side length; above 224 interpolates position embeddings
        - device (str): torch device
        - batch_size (int): tiles per forward pass
    """

    def __init__(self, config: ClipConfig, size: int, device: str, batch_size: int = 16) -> None:
        self.config = config
        self.size = size
        self.device = device
        self.batch_size = batch_size
        self.model = AutoModel.from_pretrained(CLIP_MODEL).to(device).half().eval()
        processor = AutoImageProcessor.from_pretrained(CLIP_MODEL)
        self.mean, self.std = processor.image_mean, processor.image_std
        tokenizer = AutoTokenizer.from_pretrained(CLIP_MODEL)
        prompts = [config.template.format(p) for p in (*config.targets, *config.backgrounds)]
        tokens = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            text = _tensor(self.model.get_text_features(**tokens)).float()
            self.scale = float(self.model.logit_scale.exp())
        self.text = text / text.norm(dim=-1, keepdim=True)

    def patch_embeddings(self, pixels: torch.Tensor) -> torch.Tensor:
        """
        Text-aligned embeddings for every patch of a batch of tiles.

        Parameters:
            - pixels (torch.Tensor): normalised tiles, (B, 3, size, size), float16

        Return: unit-length patch embeddings, (B, patches, dim), float32
        """
        vision = self.model.vision_model
        out = vision(
            pixel_values=pixels,
            interpolate_pos_encoding=self.size != 224,
            output_hidden_states=True,
        )
        x = out.hidden_states[-2]  # input to the last encoder layer
        layer = vision.encoder.layers[-1]
        attn = layer.self_attn
        h = layer.layer_norm1(x)
        b, n, d = h.shape
        q = attn.q_proj(h).view(b, n, attn.num_heads, attn.head_dim).transpose(1, 2)
        v = attn.v_proj(h).view(b, n, attn.num_heads, attn.head_dim).transpose(1, 2)
        weights = torch.softmax((q @ q.transpose(-1, -2)) * attn.scale, dim=-1)  # self-self
        mixed = (weights @ v).transpose(1, 2).reshape(b, n, d)
        dense = attn.out_proj(mixed)  # no residual, no feed-forward
        patches = self.model.visual_projection(vision.post_layernorm(dense[:, 1:]))
        cls = self.model.visual_projection(vision.post_layernorm(out.last_hidden_state[:, 0]))
        patches = patches.float() - CLS_WEIGHT * cls.float()[:, None, :]
        return patches / patches.norm(dim=-1, keepdim=True)

    def score_tiles(self, tiles: Sequence[np.ndarray]) -> list[float]:
        """
        Score each tile by the highest target probability of any of its patches.

        Parameters:
            - tiles (Sequence[np.ndarray]): uint8 RGB square tiles

        Return: one score in [0, 1] per tile
        """
        targets = len(self.config.targets)
        scores: list[float] = []
        with torch.no_grad():
            for start in range(0, len(tiles), self.batch_size):
                batch = tiles[start : start + self.batch_size]
                pixels = tiles_to_tensor(batch, self.size, self.mean, self.std, self.device)
                patches = self.patch_embeddings(pixels.half())
                probs = (self.scale * patches @ self.text.T).softmax(dim=-1)
                target = probs[..., :targets].sum(dim=-1)  # (B, patches)
                scores.extend(float(s) for s in target.amax(dim=-1).cpu())
        return scores


class Owlv2Scorer:
    """
    Score each tile by OWLv2's highest per-patch probability for any target prompt.

    OWLv2 scores each query independently with a sigmoid, so only the target
    prompts are used; background prompts have no role.

    Parameters:
        - config (ClipConfig): prompts and template; only targets are used
        - device (str): torch device
        - batch_size (int): tiles per forward pass
    """

    def __init__(self, config: ClipConfig, device: str, batch_size: int = 4) -> None:
        self.device = device
        self.batch_size = batch_size
        self.model = Owlv2ForObjectDetection.from_pretrained(OWL_MODEL).to(device).half().eval()
        processor = AutoImageProcessor.from_pretrained(OWL_MODEL)
        self.mean, self.std = processor.image_mean, processor.image_std
        self.size = int(self.model.config.vision_config.image_size)
        tokenizer = AutoTokenizer.from_pretrained(OWL_MODEL)
        prompts = [config.template.format(p) for p in config.targets]
        tokens = tokenizer(prompts, padding="max_length", return_tensors="pt").to(device)
        with torch.no_grad():
            self.queries = _tensor(self.model.owlv2.get_text_features(**tokens))[None]  # (1, Q, D)

    def score_tiles(self, tiles: Sequence[np.ndarray]) -> list[float]:
        """
        Score each tile by its highest patch-and-query probability.

        Parameters:
            - tiles (Sequence[np.ndarray]): uint8 RGB square tiles

        Return: one score in [0, 1] per tile
        """
        scores: list[float] = []
        with torch.no_grad():
            for start in range(0, len(tiles), self.batch_size):
                batch = tiles[start : start + self.batch_size]
                pixels = tiles_to_tensor(batch, self.size, self.mean, self.std, self.device)
                feats, _ = self.model.image_embedder(pixel_values=pixels.half())
                feats = feats.reshape(feats.shape[0], -1, feats.shape[-1])
                queries = self.queries.expand(feats.shape[0], -1, -1)
                logits, _ = self.model.class_predictor(feats, queries)
                scores.extend(float(s) for s in logits.float().sigmoid().amax(dim=(1, 2)).cpu())
        return scores


def held_out(dataset: str, fit_fraction: float, max_empty: int | None) -> list[TraceFrame]:
    """
    The held-out frames of every sequence that have a clear label.

    Parameters:
        - dataset (str): a key of ``dataloader.DATASETS``
        - fit_fraction (float): leading share of each recording that is not held out
        - max_empty (int | None): keep every target frame and this many empty ones,
          spread evenly across all sequences

    Return: frames in sequence order
    """
    labelled = [
        frame
        for sequence in dataloader.sequences(dataset)
        for frame in dataloader.load(dataset, sequence).split(fit_fraction)[1]
        if frame.has_target((DONT_CARE,)) is not None
    ]
    empty = [i for i, frame in enumerate(labelled) if not frame.has_target((DONT_CARE,))]
    if max_empty is None or len(empty) <= max_empty:
        return labelled
    step = len(empty) / max_empty
    keep = {empty[int(k * step)] for k in range(max_empty)}
    return [f for i, f in enumerate(labelled) if i in keep or f.has_target((DONT_CARE,))]


def evaluate(
    name: str, scorer: object, tile_size: int, frames: list[TraceFrame], dataset: str
) -> None:
    """
    Score frames with one method and print one result row.

    Parameters:
        - name (str): row label
        - scorer (object): a TileScorer
        - tile_size (int): tile side in frame pixels
        - frames (list[TraceFrame]): labelled frames to score
        - dataset (str): dataset name for the row

    Return: void
    """
    tagger = Tagger(scorer, PriorityScale(reference=1.0), tile_size=tile_size)  # type: ignore[arg-type]
    rows: dict[str, list[float]] = {name: []}
    labels, millis = [], []
    for frame in frames:
        labels.append(bool(frame.has_target((DONT_CARE,))))
        image_bytes = frame.read()
        started = time.perf_counter()
        result = tagger.score(image_bytes)
        millis.append(1000 * (time.perf_counter() - started))
        rows[name].append(result.raw_score)
        frame_contrast = result.raw_score - statistics.median(result.tile_scores)
        rows.setdefault(f"{name}+frame", []).append(frame_contrast)
        for kind, value in getattr(scorer, "extra", {}).items():
            if kind != "max":
                rows.setdefault(f"{name}+{kind}", []).append(value)
    for row, scores in rows.items():
        recalls = "  ".join(f"{recall_at_fraction(scores, labels, f):6.3f}" for f in TOP_FRACTIONS)
        print(
            f"  {row:24s} {dataset:20s} {len(labels):5d} {sum(labels):4d}  "
            f"{auroc(scores, labels):.3f}  {average_precision(scores, labels):.3f}  {recalls}  "
            f"{statistics.median(millis):7.1f}",
            flush=True,
        )


def main(argv: list[str] | None = None) -> None:
    """
    Run each method on each dataset and print the comparison.

    Parameters:
        - argv (list[str] | None): arguments without the program name; None means sys.argv

    Return: void
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--datasets", nargs="+", default=["sparse-seadronessee", "sparse-visdrone"])
    parser.add_argument("--methods", nargs="+", default=["clip", "siglip2", "tips_tile"])
    parser.add_argument(
        "--tiles", type=int, nargs="+", default=[256, 192, 128], help="tile sizes, tile models"
    )
    parser.add_argument("--fit-fraction", type=float, default=0.7)
    parser.add_argument("--max-empty", type=int, help="keep all target frames, this many empty")
    parser.add_argument("--device")
    args = parser.parse_args(argv)

    device = pick_device(args.device)
    config = ClipConfig()
    tile_models = {
        "clip": lambda: ClipScorer.load(config, device=device, half=True),
        "siglip2": lambda: Siglip2TileScorer(config, device),
        "siglip2_sig": lambda: Siglip2TileScorer(config, device, sigmoid=True),
        "tips_tile": lambda: TipsScorer(config, 224, False, device),
    }
    builders = {
        "dense224": lambda: (DenseClipScorer(config, 224, device), 224),
        "dense448": lambda: (DenseClipScorer(config, 448, device), 448),
        "owlv2": lambda: (Owlv2Scorer(config, device), None),
        "tips224": lambda: (TipsScorer(config, 224, True, device), 224),
        "tips448": lambda: (TipsScorer(config, 448, True, device), 448),
    }
    header = "  ".join(f"top{100 * f:g}%" for f in TOP_FRACTIONS)
    print(f"  {'method':24s} {'dataset':20s} frames tgts  AUROC     AP  {header}  ms/frame")
    frames = {d: held_out(d, args.fit_fraction, args.max_empty) for d in args.datasets}
    for method in args.methods:
        if method in tile_models:
            scorer = tile_models[method]()
            for tile in args.tiles:
                for dataset in args.datasets:
                    evaluate(f"{method}@{tile}", scorer, tile, frames[dataset], dataset)
            del scorer
            if device == "mps":
                torch.mps.empty_cache()
            continue
        scorer, tile = builders[method]()
        for dataset in args.datasets:
            # OWLv2 takes the largest square that fits: 720px on 1280x720 crops, 512px on 512px.
            size = tile or (720 if dataset == "sparse-seadronessee" else 512)
            evaluate(method, scorer, size, frames[dataset], dataset)
        del scorer
        if device == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()

"""
Experiment: does a unified VLM embedding make plain text-to-tile search work?

Usage:
    uv run python scripts/experiment_clipnn_unified.py
    uv run python scripts/experiment_clipnn_unified.py --empty 30

Qwen3-VL-Embedding-2B runs its whole network, vision encoder and language
model, over every tile, and the language model over every phrase, so tiles
and phrases come out of one model into one space. CLIP B/16 runs its image
encoder once per tile and its text encoder once per phrase. Both see the
same 224px tiles from a small sample of held-out frames: every target frame
plus ``--empty`` random empty ones per dataset.

Scores per tile, for each model:

- raw: best cosine similarity to a target phrase, i.e. plain nearest-neighbour search
- margin: raw minus the best cosine similarity to a background phrase
- softmax (CLIP only): the pipeline's score, for reference

Tile metrics count a tile as a target tile when a target box's centre lies
in it; tiles that touch a box without holding a centre are left out. Frame
metrics score a frame by its best tile. Targets are far more common in this
sample than in the stream, so AP is only comparable between rows here.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, AutoTokenizer

from tagurit.model.clip import ClipConfig
from tagurit.model.device import pick_device
from tagurit.model.preprocess import tiles_to_tensor
from tagurit.model.tiling import crop_tiles, decode_image, tile_grid
from tagurit.sim import dataloader
from tagurit.sim.metrics import auroc, average_precision

CLIP_MODEL = "openai/clip-vit-base-patch16"
QWEN_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
QUERY_INSTRUCTION = "Retrieve aerial images that show the given object or scene."
TILE_INSTRUCTION = "Represent the user's input."
DONT_CARE = ("ignored",)
TILE = 224


class ClipEmbedder:
    """
    CLIP image and text embeddings, unit length, as float32 arrays.

    Parameters:
        - device (str): torch device
        - batch_size (int): tiles per forward pass
    """

    def __init__(self, device: str, batch_size: int = 64) -> None:
        self.device, self.batch_size = device, batch_size
        self.model = AutoModel.from_pretrained(CLIP_MODEL).to(device).half().eval()
        processor = AutoImageProcessor.from_pretrained(CLIP_MODEL)
        self.mean, self.std = processor.image_mean, processor.image_std
        self.tokenizer = AutoTokenizer.from_pretrained(CLIP_MODEL)
        self.scale = float(self.model.logit_scale.exp())

    @staticmethod
    def _unit(output: object) -> np.ndarray:
        t = output if isinstance(output, torch.Tensor) else output.pooler_output  # type: ignore[attr-defined]
        return F.normalize(t.float(), dim=-1).cpu().numpy()

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
                pixels = tiles_to_tensor(batch, TILE, self.mean, self.std, self.device)
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


class QwenEmbedder:
    """
    Qwen3-VL-Embedding vectors: the last token's final hidden state, unit length.

    Follows the model's reference embedder: a system turn holds the
    instruction, the user turn holds the image or text, and the embedding is
    the hidden state of the last real token. Tiles are already a multiple of
    32 pixels, so they enter at native size (49 visual tokens at 224px).

    Parameters:
        - device (str): torch device
        - batch_size (int): inputs per forward pass
    """

    def __init__(self, device: str, batch_size: int = 16) -> None:
        from transformers import Qwen3VLModel

        self.device, self.batch_size = device, batch_size
        self.model = Qwen3VLModel.from_pretrained(QWEN_MODEL, dtype=torch.bfloat16)
        self.model = self.model.to(device).eval()
        self.processor = AutoProcessor.from_pretrained(QWEN_MODEL, padding_side="right")

    def _embed(
        self, contents: list[list[dict]], instruction: str, images: list | None
    ) -> np.ndarray:
        conversations = [
            [
                {"role": "system", "content": [{"type": "text", "text": instruction}]},
                {"role": "user", "content": content},
            ]
            for content in contents
        ]
        text = self.processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )
        inputs = self.processor(
            text=text, images=images, padding=True, do_resize=False, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            hidden = self.model(**inputs).last_hidden_state
        mask = inputs["attention_mask"]
        last = mask.shape[1] - 1 - mask.flip(1).argmax(dim=1)
        pooled = hidden[torch.arange(len(hidden), device=hidden.device), last]
        return F.normalize(pooled.float(), dim=-1).cpu().numpy()

    def images(self, tiles: list[np.ndarray]) -> np.ndarray:
        """
        Embed tiles.

        Parameters:
            - tiles (list[np.ndarray]): uint8 RGB square tiles

        Return: (len(tiles), dim) unit embeddings
        """
        out = []
        for start in range(0, len(tiles), self.batch_size):
            batch = [Image.fromarray(t) for t in tiles[start : start + self.batch_size]]
            out.append(self._embed([[{"type": "image"}]] * len(batch), TILE_INSTRUCTION, batch))
        return np.concatenate(out)

    def texts(self, prompts: list[str]) -> np.ndarray:
        """
        Embed prompts as search queries.

        Parameters:
            - prompts (list[str]): sentences

        Return: (len(prompts), dim) unit embeddings
        """
        contents = [[{"type": "text", "text": p}] for p in prompts]
        return self._embed(contents, QUERY_INSTRUCTION, None)


def tile_labels(frame: object, grid: tuple, frame_label: bool) -> list[bool | None]:
    """
    Label each tile by whether a target box's centre lies in it.

    Parameters:
        - frame (TraceFrame): the frame and its boxes
        - grid (tuple[Tile, ...]): the frame's tiles
        - frame_label (bool): whether the frame holds a target

    Return: True, False, or None (touches a box without holding a centre,
        or a target frame with no target boxes) per tile
    """
    boxes = frame.boxes  # type: ignore[attr-defined]
    targets = [b for b in boxes if b.label not in DONT_CARE]
    if frame_label and not targets:
        return [None] * len(grid)
    labels: list[bool | None] = []
    for t in grid:
        if any(_centre_in(b, t) for b in targets):
            labels.append(True)
        else:
            labels.append(None if any(_overlaps(b, t) for b in boxes) else False)
    return labels


def _centre_in(box: object, tile: object) -> bool:
    x = box.left + box.width / 2  # type: ignore[attr-defined]
    y = box.top + box.height / 2  # type: ignore[attr-defined]
    return tile.left <= x < tile.left + tile.size and tile.top <= y < tile.top + tile.size  # type: ignore[attr-defined]


def _overlaps(box: object, tile: object) -> bool:
    return (
        box.left < tile.left + tile.size  # type: ignore[attr-defined]
        and box.left + box.width > tile.left  # type: ignore[attr-defined]
        and box.top < tile.top + tile.size  # type: ignore[attr-defined]
        and box.top + box.height > tile.top  # type: ignore[attr-defined]
    )


def sample(dataset: str, empty: int, fit_fraction: float, rng: np.random.Generator) -> list:
    """
    Every held-out target frame plus ``empty`` random held-out empty frames.

    Parameters:
        - dataset (str): dataset name
        - empty (int): empty frames to draw
        - fit_fraction (float): leading share of each sequence left out
        - rng (np.random.Generator): draws the empty frames

    Return: (frame, frame label) pairs
    """
    held_out = []
    for sequence in dataloader.sequences(dataset):
        _, trail = dataloader.load(dataset, sequence).split(fit_fraction)
        held_out.extend(trail)
    positive = [f for f in held_out if f.has_target(DONT_CARE) is True]
    negative = [f for f in held_out if f.has_target(DONT_CARE) is False]
    picks = rng.choice(len(negative), size=min(empty, len(negative)), replace=False)
    return [(f, True) for f in positive] + [(negative[i], False) for i in sorted(picks)]


def main(argv: list[str] | None = None) -> None:
    """
    Embed a small sample of tiles with both models and compare search scores.

    Parameters:
        - argv (list[str] | None): arguments without the program name; None means sys.argv

    Return: void
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--datasets", nargs="+", default=["sparse-seadronessee", "sparse-visdrone"])
    parser.add_argument("--empty", type=int, default=60, help="empty frames per dataset")
    parser.add_argument("--fit-fraction", type=float, default=0.7)
    parser.add_argument("--device")
    args = parser.parse_args(argv)

    device = pick_device(args.device)
    rng = np.random.default_rng(0)
    config = ClipConfig()
    targets = [config.template.format(p) for p in config.targets]
    backgrounds = [config.template.format(p) for p in config.backgrounds]

    tiles: list[np.ndarray] = []
    tile_truth: list[bool | None] = []
    owner: list[tuple[str, int]] = []  # (dataset, frame index) per tile
    frame_truth: dict[str, list[bool]] = {}
    for dataset in args.datasets:
        frames = sample(dataset, args.empty, args.fit_fraction, rng)
        frame_truth[dataset] = [label for _, label in frames]
        for index, (frame, label) in enumerate(frames):
            image = decode_image(frame.read())
            grid = tile_grid(image.shape[1], image.shape[0], TILE)
            tiles.extend(crop_tiles(image, grid))
            tile_truth.extend(tile_labels(frame, grid, label))
            owner.extend([(dataset, index)] * len(grid))
        print(
            f"{dataset}: {len(frames)} frames ({sum(frame_truth[dataset])} with targets), "
            f"{sum(1 for d, _ in owner if d == dataset)} tiles",
            flush=True,
        )

    results: dict[str, dict[str, np.ndarray]] = {}
    for name, make in (
        ("clip-b16", lambda: ClipEmbedder(device)),
        ("qwen3vl-2b", lambda: QwenEmbedder(device)),
    ):
        model = make()
        start = time.perf_counter()
        embedded = model.images(tiles)
        ms = 1000 * (time.perf_counter() - start) / len(tiles)
        target_sims = embedded @ model.texts(targets).T
        background_sims = embedded @ model.texts(backgrounds).T
        raw = target_sims.max(axis=1)
        results[name] = {"raw": raw, "margin": raw - background_sims.max(axis=1)}
        if isinstance(model, ClipEmbedder):
            logits = model.scale * np.concatenate([target_sims, background_sims], axis=1)
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
            results[name]["softmax"] = probs[:, : len(targets)].sum(axis=1)
        print(f"{name}: {ms:.1f} ms/tile on {device}, {embedded.shape[1]} dims", flush=True)
        del model
        if device == "mps":
            torch.mps.empty_cache()

    print(f"\n  {'model':11s} {'score':8s} {'dataset':20s}  tileAUROC  tileAP  frameAUROC  frameAP")
    for dataset in args.datasets:
        rows = [i for i, (d, _) in enumerate(owner) if d == dataset]
        labelled = [i for i in rows if tile_truth[i] is not None]
        truth = [bool(tile_truth[i]) for i in labelled]
        for name, scores in results.items():
            for score, values in scores.items():
                best: dict[int, float] = {}
                for i in rows:
                    best[owner[i][1]] = max(best.get(owner[i][1], -np.inf), float(values[i]))
                frames = [best[k] for k in sorted(best)]
                tile_scores = [float(values[i]) for i in labelled]
                print(
                    f"  {name:11s} {score:8s} {dataset:20s}  "
                    f"{auroc(tile_scores, truth):9.3f}  "
                    f"{average_precision(tile_scores, truth):6.3f}  "
                    f"{auroc(frames, frame_truth[dataset]):10.3f}  "
                    f"{average_precision(frames, frame_truth[dataset]):7.3f}"
                )
        print(f"  ({sum(truth)} target tiles of {len(truth)} labelled)")


if __name__ == "__main__":
    main()

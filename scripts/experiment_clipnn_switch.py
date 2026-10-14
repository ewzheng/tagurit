"""
Experiment: switch the clipnn query mid-mission and re-rank the stored backlog.

Usage:
    uv run --with onnx --with onnxscript python scripts/experiment_clipnn_switch.py prepare
    (copy data/clipnn-switch/stage to the Jetson and run scripts/jetson/build_engines.py
    and scripts/jetson/clipnn_scout.py there, then copy the results back)
    uv run python scripts/experiment_clipnn_switch.py analyse

The mission per dataset is the whole sparse stream in time order, thinned to
every target frame plus every ``--empty-stride``th empty frame; CLIP fits
nothing, so no frames are held back. Query A runs for the first half. At the
midpoint the cloudlet sends query B, and because the scout kept every tile
embedding it re-ranks the first half (the backlog) by query B with dot
products alone, without touching an image.

A query's phrases compete against a reject set shipped once before the mission:

- handwritten: the pipeline's 11 background phrases
- vocabulary: the ~100 generic scenery phrases of ``experiment_clipnn_robust.py``, used whole

``prepare`` writes, under ``data/clipnn-switch/stage``: the mission frames and
their tile grids, calibration tiles from empty frames outside the mission,
the CLIP B/16 image encoder as ONNX, and the text embeddings the cloudlet
would send. ``analyse`` reads the Jetson's embeddings and timings from
``data/clipnn-switch/results`` and prints how closely each engine matches a
float32 reference, how well the backlog ranks for query B when re-ranked from
stored embeddings against keeping query A's order, and the device costs.
A tile holds a class when a box of that class has its centre in it; tiles
that only clip such a box, or touch an ignored region, are left out.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from experiment_clipnn_robust import VOCABULARY  # noqa: E402

from tagurit.model.clip import ClipConfig  # noqa: E402
from tagurit.model.device import pick_device  # noqa: E402
from tagurit.model.preprocess import tiles_to_tensor  # noqa: E402
from tagurit.model.tiling import Tile, crop_tiles, decode_image, tile_grid  # noqa: E402
from tagurit.sim import dataloader  # noqa: E402
from tagurit.sim.metrics import average_precision  # noqa: E402

CLIP_MODEL = "openai/clip-vit-base-patch16"
TILE = 224
WORK = Path("data/clipnn-switch")
DONT_CARE = "ignored"
ENGINES = ("fp16", "int8")
# per dataset: query A, then query B; each is (name, phrases, COCO labels it means)
QUERIES = {
    "sparse-seadronessee": (
        ("boats", ("a boat", "a jet ski"), {"boat"}),
        ("swimmers", ("a swimmer", "a person in the water"), {"person"}),
    ),
    "sparse-visdrone": (
        ("cars", ("a car", "a van"), {"car"}),
        ("people", ("a person", "a pedestrian"), {"person"}),
    ),
}


def mission(dataset: str, stride: int) -> tuple[list, list]:
    """
    The mission's frames in time order, and the empty frames it skipped.

    Parameters:
        - dataset (str): dataset name
        - stride (int): keep every stride-th empty frame

    Return: (kept frames, skipped empty frames)
    """
    kept, skipped, empties = [], [], 0
    for sequence in dataloader.sequences(dataset):
        for frame in dataloader.load(dataset, sequence):
            label = frame.has_target((DONT_CARE,))
            if label is None:
                continue
            if label:
                kept.append(frame)
            else:
                (kept if empties % stride == 0 else skipped).append(frame)
                empties += 1
    return kept, skipped


def frame_key(frame: object) -> str:
    """
    A frame's path relative to the dataset root, stable across machines.

    Parameters:
        - frame (TraceFrame): the frame

    Return: "<sequence>/<file>"
    """
    return f"{frame.sequence}/{frame.path.name}"  # type: ignore[attr-defined]


def prepare(args: argparse.Namespace) -> None:
    """
    Write the mission, calibration tiles, ONNX encoder and query messages to the stage.

    Parameters:
        - args (argparse.Namespace): parsed arguments

    Return: void
    """
    from transformers import (
        AutoImageProcessor,
        AutoModel,
        AutoTokenizer,
        CLIPVisionModelWithProjection,
    )

    stage = WORK / "stage"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    rng = np.random.default_rng(0)
    processor = AutoImageProcessor.from_pretrained(CLIP_MODEL)
    manifest: dict = {"tile": TILE, "mean": processor.image_mean, "std": processor.image_std}
    calibration: list[np.ndarray] = []
    for dataset in QUERIES:
        kept, skipped = mission(dataset, args.empty_stride)
        frames = []
        for frame in kept:
            key = frame_key(frame)
            target = stage / "frames" / dataset / key
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(frame.path, target)
            image = decode_image(frame.read())
            grid = tile_grid(image.shape[1], image.shape[0], TILE)
            tiles = [[t.left, t.top] for t in grid]
            frames.append({"path": f"frames/{dataset}/{key}", "tiles": tiles})
        manifest[dataset] = {"frames": frames, "switch": len(frames) // 2}
        for i in rng.choice(len(skipped), size=args.calibration // (2 * 4), replace=False):
            image = decode_image(skipped[i].read())
            grid = tile_grid(image.shape[1], image.shape[0], TILE)
            picks = rng.choice(len(grid), size=min(4, len(grid)), replace=False)
            calibration.extend(crop_tiles(image, [grid[p] for p in picks]))
        tiles = sum(len(f["tiles"]) for f in frames)
        print(f"{dataset}: {len(frames)} mission frames, {tiles} tiles", flush=True)
    (stage / "manifest.json").write_text(json.dumps(manifest))
    np.save(stage / "calibration.npy", np.stack(calibration))
    print(f"calibration: {len(calibration)} tiles from empty frames outside the missions")

    model = AutoModel.from_pretrained(CLIP_MODEL).eval()
    tokenizer = AutoTokenizer.from_pretrained(CLIP_MODEL)
    config = ClipConfig()
    say = config.template.format

    def texts(phrases: list[str]) -> np.ndarray:
        tokens = tokenizer(phrases, padding=True, return_tensors="pt")
        with torch.no_grad():
            out = model.get_text_features(**tokens)
        out = out if isinstance(out, torch.Tensor) else out.pooler_output
        return F.normalize(out.float(), dim=-1).numpy().astype(np.float16)

    messages = {
        "scale": np.float32(model.logit_scale.exp().item()),
        "handwritten": texts([say(p) for p in config.backgrounds]),
        "vocabulary": texts([say(p) for p in VOCABULARY]),
    }
    for dataset, queries in QUERIES.items():
        for name, phrases, _ in queries:
            messages[f"{dataset}/{name}"] = texts([say(p) for p in phrases])
    np.savez(stage / "messages.npz", **messages)
    sizes = {k: v.nbytes for k, v in messages.items() if k != "scale"}
    print("message bytes (float16): " + ", ".join(f"{k} {v}" for k, v in sizes.items()))

    class Encoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.clip = CLIPVisionModelWithProjection.from_pretrained(
                CLIP_MODEL, attn_implementation="eager"
            ).eval()

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            embeds = self.clip(pixel_values=pixel_values).image_embeds
            return embeds / embeds.norm(dim=-1, keepdim=True)

    torch.onnx.export(
        Encoder(),
        (torch.randn(2, 3, TILE, TILE),),
        str(stage / "clip-b16-vision.onnx"),
        input_names=["pixel_values"],
        output_names=["embeds"],
        dynamic_axes={"pixel_values": {0: "batch"}, "embeds": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"exported {stage / 'clip-b16-vision.onnx'}")


def tile_classes(frame: object, grid: tuple, labels: set[str]) -> list[bool | None]:
    """
    Whether each tile holds a box of the given classes.

    Parameters:
        - frame (TraceFrame): the frame and its boxes
        - grid (tuple[Tile, ...]): the frame's tiles
        - labels (set[str]): COCO labels that count

    Return: True, False, or None (clips such a box without its centre, or
        touches an ignored region) per tile
    """
    out: list[bool | None] = []
    for t in grid:
        right, bottom = t.left + t.size, t.top + t.size
        holds = clips = False
        for b in frame.boxes:  # type: ignore[attr-defined]
            overlap = (
                b.left < right
                and b.left + b.width > t.left
                and b.top < bottom
                and b.top + b.height > t.top
            )
            x, y = b.left + b.width / 2, b.top + b.height / 2
            if b.label in labels and t.left <= x < right and t.top <= y < bottom:
                holds = True
            elif overlap and (b.label in labels or b.label == DONT_CARE):
                clips = True
        out.append(True if holds else (None if clips else False))
    return out


def share(tiles: np.ndarray, query: np.ndarray, rejects: np.ndarray, scale: float) -> np.ndarray:
    """
    Softmax share of the query phrases against the reject phrases, per tile.

    Parameters:
        - tiles (np.ndarray): (T, dim) tile embeddings
        - query (np.ndarray): (P, dim) query phrase embeddings
        - rejects (np.ndarray): (Q, dim) reject phrase embeddings
        - scale (float): CLIP's logit scale

    Return: (T,) scores in [0, 1]
    """
    logits = (
        scale * tiles.astype(np.float32) @ np.concatenate([query, rejects]).astype(np.float32).T
    )
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    return probs[:, : len(query)].sum(axis=1) / probs.sum(axis=1)


def to_int8(embeddings: np.ndarray) -> np.ndarray:
    """
    Store embeddings as int8 with one scale per vector, and read them back.

    Parameters:
        - embeddings (np.ndarray): (T, dim) embeddings

    Return: (T, dim) float32 embeddings after the int8 round trip
    """
    x = embeddings.astype(np.float32)
    scale = np.abs(x).max(axis=1, keepdims=True) / 127
    return np.round(x / scale).astype(np.int8).astype(np.float32) * scale


def reference(manifest: dict, dataset: str, device: str) -> np.ndarray:
    """
    Float32 CLIP B/16 embeddings of every mission tile, as the accuracy reference.

    Parameters:
        - manifest (dict): the staged manifest
        - dataset (str): dataset name
        - device (str): torch device

    Return: (tiles, 512) float32 unit embeddings in mission order
    """
    from transformers import CLIPVisionModelWithProjection

    cache = WORK / "results" / f"reference-{dataset}.npy"
    if cache.exists():
        return np.load(cache)
    model = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL).to(device).eval()
    out = []
    with torch.no_grad():
        for entry in manifest[dataset]["frames"]:
            image = decode_image((WORK / "stage" / entry["path"]).read_bytes())
            tiles = [image[t : t + TILE, s : s + TILE] for s, t in entry["tiles"]]
            pixels = tiles_to_tensor(tiles, TILE, manifest["mean"], manifest["std"], device)
            out.append(F.normalize(model(pixel_values=pixels).image_embeds, dim=-1).cpu().numpy())
    embeddings = np.concatenate(out)
    np.save(cache, embeddings)
    return embeddings


class Truth:
    """
    Per-tile and per-frame ground truth for named classes over one mission.

    Parameters:
        - kept (list[TraceFrame]): the mission frames, in order
        - frames (list[dict]): their manifest entries
        - classes (dict[str, set[str]]): class name -> COCO labels it means
    """

    def __init__(self, kept: list, frames: list[dict], classes: dict[str, set[str]]) -> None:
        owner: list[int] = []
        self.tiles: dict[str, list[bool | None]] = {name: [] for name in classes}
        self.frames: dict[str, list[bool]] = {name: [] for name in classes}
        for index, (frame, entry) in enumerate(zip(kept, frames, strict=True)):
            grid = tuple(Tile(left=s, top=t, size=TILE) for s, t in entry["tiles"])
            owner.extend([index] * len(grid))
            for name, labels in classes.items():
                self.tiles[name].extend(tile_classes(frame, grid, labels))
                self.frames[name].append(any(b.label in labels for b in frame.boxes))
        self.owner = np.array(owner)
        self.count = len(frames)

    def ap(self, scores: np.ndarray, name: str, span: range) -> tuple[float, float]:
        """
        Tile-level and frame-level average precision for one class over a span of frames.

        Parameters:
            - scores (np.ndarray): one score per mission tile
            - name (str): class name
            - span (range): frame indices that count

        Return: (tile AP, frame AP); a frame scores its best tile
        """
        rows = np.flatnonzero(np.isin(self.owner, list(span)))
        labelled = [i for i in rows if self.tiles[name][i] is not None]
        tile_ap = average_precision(
            [float(scores[i]) for i in labelled], [bool(self.tiles[name][i]) for i in labelled]
        )
        best = np.full(self.count, -np.inf)
        np.maximum.at(best, self.owner, scores)
        frame_ap = average_precision(
            [float(best[k]) for k in span], [self.frames[name][k] for k in span]
        )
        return tile_ap, frame_ap

    def frames_with(self, name: str, span: range) -> int:
        """
        Count the frames in a span that hold a class.

        Parameters:
            - name (str): class name
            - span (range): frame indices

        Return: number of frames
        """
        return sum(self.frames[name][k] for k in span)


def analyse(args: argparse.Namespace) -> None:
    """
    Score the Jetson's embeddings against the box labels and print the tables.

    Parameters:
        - args (argparse.Namespace): parsed arguments

    Return: void
    """
    stage, results = WORK / "stage", WORK / "results"
    manifest = json.loads((stage / "manifest.json").read_text())
    messages = np.load(stage / "messages.npz")
    scale = float(messages["scale"])
    device = pick_device(args.device)
    reject_sets = {"handwritten": messages["handwritten"], "vocabulary": messages["vocabulary"]}

    for dataset, ((name_a, _, labels_a), (name_b, _, labels_b)) in QUERIES.items():
        kept, _ = mission(dataset, args.empty_stride)
        frames = manifest[dataset]["frames"]
        if [f"frames/{dataset}/{frame_key(f)}" for f in kept] != [f["path"] for f in frames]:
            raise ValueError(f"{dataset}: the staged mission does not match --empty-stride")
        truth = Truth(kept, frames, {name_a: labels_a, name_b: labels_b})
        switch = manifest[dataset]["switch"]
        query_a = messages[f"{dataset}/{name_a}"]
        query_b = messages[f"{dataset}/{name_b}"]
        ref = reference(manifest, dataset, device)
        stores = {"float32 reference": ref}
        for engine in ENGINES:
            stores[f"TensorRT {engine}"] = np.load(results / f"{engine}-{dataset}.npy")
        whole, backlog = range(len(frames)), range(switch)

        print(
            f"\n{dataset}: {len(frames)} frames, "
            f"query A = {name_a}, B = {name_b} from frame {switch}"
        )
        print(
            f"  {'embeddings':18s} {'cos':>6s}  {'rejects':11s}  A tile/frame AP  B tile/frame AP"
        )
        for store, embeddings in stores.items():
            cos = float(np.mean(np.sum(embeddings.astype(np.float32) * ref, axis=1)))
            for rejects, reject in reject_sets.items():
                a = truth.ap(share(embeddings, query_a, reject, scale), name_a, whole)
                b = truth.ap(share(embeddings, query_b, reject, scale), name_b, whole)
                print(
                    f"  {store:18s} {cos:6.4f}  {rejects:11s}  "
                    f"{a[0]:6.3f} / {a[1]:.3f}  {b[0]:6.3f} / {b[1]:.3f}"
                )
        print(
            f"  ({truth.frames_with(name_a, whole)} {name_a} frames, "
            f"{truth.frames_with(name_b, whole)} {name_b} frames)"
        )

        deployed = stores["TensorRT int8"]
        print(f"\n  backlog (frames 0-{switch - 1}, int8 engine) ranked for {name_b}")
        print(f"  {'backlog order':32s} {'rejects':11s}  tileAP  frameAP")
        for rejects, reject in reject_sets.items():
            orders = {
                "re-ranked, float16 store": share(deployed, query_b, reject, scale),
                "re-ranked, int8 store": share(to_int8(deployed), query_b, reject, scale),
                "kept from query A": share(deployed, query_a, reject, scale),
            }
            for label, scores in orders.items():
                tile_ap, frame_ap = truth.ap(scores, name_b, backlog)
                print(f"  {label:32s} {rejects:11s}  {tile_ap:6.3f}  {frame_ap:7.3f}")
        print(f"  ({truth.frames_with(name_b, backlog)} {name_b} frames in the backlog)")

    print("\ndevice timings (Jetson)")
    print(json.dumps(json.loads((results / "timings.json").read_text()), indent=2))


def main(argv: list[str] | None = None) -> None:
    """
    Run one stage of the switch experiment.

    Parameters:
        - argv (list[str] | None): arguments without the program name; None means sys.argv

    Return: void
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("stage", choices=["prepare", "analyse"])
    parser.add_argument("--empty-stride", type=int, default=5, help="keep every nth empty frame")
    parser.add_argument("--calibration", type=int, default=512, help="calibration tiles")
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    {"prepare": prepare, "analyse": analyse}[args.stage](args)


if __name__ == "__main__":
    main()

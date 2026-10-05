"""
Build a synthetic sparse stream by cropping windows out of an annotated dataset.

Usage:
    uv run python scripts/make_sparse.py seadronessee
    uv run python scripts/make_sparse.py visdrone --size 512x512

For every source frame, in order, ``--crops-per-frame`` windows of
``--size`` are planned. Each is a target window with probability
``--rate``, holding one whole target (``tagurit.sim.sparse.target_window``),
and otherwise an empty window that no box of any class comes within
``--margin`` pixels of. When the chosen kind does not fit in a frame, that
crop is skipped and counted. Crowded frames rarely fit an empty window, so
before anything is written, target crops are thinned at random until each
sequence's target share is at most ``--rate``. Without that, a crowded
sequence would be mostly targets and a scorer could score well by
recognising the sequence rather than the target. Crops keep the source
order, so a time split on the result is a time split on the source and
fitting never sees the frames that are scored.

Every frame of a source sequence must share the first frame's size; a
frame that does not RAISES SystemExit.

Output goes to ``data/sparse-<source>/<sequence>/`` in the
``tagurit.sim.framelist`` format, crops re-encoded as JPEG, with
``make_sparse.json`` at the top recording the settings and counts. Slashes
in source sequence names become dashes. An existing output folder RAISES
SystemExit unless ``--force`` deletes it first.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

import cv2

from tagurit.sim import dataloader
from tagurit.sim.framelist import BOX_COLUMNS, BOXES_FILE, FRAMES_FILE
from tagurit.sim.sparse import (
    Window,
    boxes_in_window,
    empty_window,
    target_window,
    thin_targets,
)
from tagurit.sim.trace import Trace, TraceFrame

SOURCES = ("seadronessee", "visdrone")
RECORD = "make_sparse.json"


def parse_size(text: str) -> tuple[int, int]:
    """
    Parse a ``WIDTHxHEIGHT`` crop size.

    Parameters:
        - text (str): e.g. "1280x720"

    Return: (width, height), both positive
    """
    try:
        width, height = (int(part) for part in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected WIDTHxHEIGHT, got {text!r}") from None
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError(f"crop size must be positive, got {text!r}")
    return width, height


@dataclass(frozen=True)
class Crop:
    """
    One planned crop of one source frame.

    Parameters:
        - frame (TraceFrame): the source frame
        - slot (int): which of the frame's crops this is, from 0
        - window (Window): the crop rectangle in frame pixels
        - target (bool): whether the window was chosen to hold a target
    """

    frame: TraceFrame
    slot: int
    window: Window
    target: bool


def plan(trace: Trace, args: argparse.Namespace, rng: random.Random) -> tuple[list[Crop], Counter]:
    """
    Choose every crop window of a sequence, then thin targets to the asked rate.

    Parameters:
        - trace (Trace): the source sequence, all frames the same size
        - args (argparse.Namespace): crop size, rate, crops per frame, margin
        - rng (random.Random): source of every random choice

    Return: kept crops in source order, and counts of kept, skipped and dropped crops
    """
    first = cv2.imread(str(trace.frames[0].path))
    if first is None:
        raise SystemExit(f"could not read {trace.frames[0].path}")
    frame_height, frame_width = first.shape[:2]
    width, height = args.size

    crops: list[Crop] = []
    counts: Counter[str] = Counter()
    for frame in trace:
        for slot in range(args.crops_per_frame):
            wants_target = rng.random() < args.rate
            if wants_target:
                window = target_window(frame.boxes, frame_width, frame_height, width, height, rng)
            else:
                window = empty_window(
                    frame.boxes, frame_width, frame_height, width, height, args.margin, rng
                )
            if window is None:
                counts["skipped target" if wants_target else "skipped empty"] += 1
            else:
                crops.append(Crop(frame, slot, window, wants_target))

    kept = [crops[i] for i in thin_targets([crop.target for crop in crops], args.rate, rng)]
    counts["dropped target"] = len(crops) - len(kept)
    counts["empty"] = sum(not crop.target for crop in kept)
    counts["target"] = len(kept) - counts["empty"]
    return kept, counts


def write_crops(crops: list[Crop], folder: Path, quality: int) -> None:
    """
    Encode planned crops into ``folder`` with their ``frames.csv`` and ``boxes.csv``.

    Each source frame is decoded once.

    Parameters:
        - crops (list[Crop]): kept crops in source order
        - folder (Path): sequence output folder, created here
        - quality (int): JPEG quality

    Return: void
    """
    folder.mkdir(parents=True)
    frames: list[dict[str, object]] = []
    boxes: list[dict[str, object]] = []
    size = None
    for frame, group in groupby(crops, key=lambda crop: crop.frame):
        image = cv2.imread(str(frame.path))
        if image is None:
            raise SystemExit(f"could not read {frame.path}")
        size = size or image.shape[:2]
        if image.shape[:2] != size:
            raise SystemExit(f"{frame.path} is {image.shape[:2]}, not the sequence's {size}")
        for crop in group:
            w = crop.window
            file = f"{frame.index:07d}_{crop.slot}.jpg"
            pixels = image[w.top : w.top + w.height, w.left : w.left + w.width]
            ok, encoded = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                raise SystemExit(f"could not encode a crop of {frame.path}")
            (folder / file).write_bytes(encoded.tobytes())
            frames.append({"file": file, "target": int(crop.target), "timestamp": frame.timestamp})
            for box in boxes_in_window(frame.boxes, w):
                boxes.append(
                    {
                        "file": file,
                        "label": box.label,
                        "dataset_label": box.dataset_label,
                        "left": box.left,
                        "top": box.top,
                        "width": box.width,
                        "height": box.height,
                    }
                )
    write_csv(folder / FRAMES_FILE, ("file", "target", "timestamp"), frames)
    write_csv(folder / BOXES_FILE, BOX_COLUMNS, boxes)


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    """
    Write rows to a CSV file with a header, OVERWRITING any existing file.

    Parameters:
        - path (Path): destination file
        - columns (tuple[str, ...]): header, in order
        - rows (list[dict[str, object]]): one dict per row keyed by column

    Return: void
    """
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    """
    Parse arguments and write the cropped sparse dataset.

    Parameters:
        - argv (list[str] | None): arguments without the program name; None means sys.argv

    Return: void
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", choices=SOURCES)
    parser.add_argument("--sequences", nargs="+", help="source sequence names; default all")
    parser.add_argument("--size", type=parse_size, default=(1280, 720), help="crop WIDTHxHEIGHT")
    parser.add_argument("--rate", type=float, default=0.01, help="share of target crops asked for")
    parser.add_argument("--crops-per-frame", type=int, default=4)
    parser.add_argument("--margin", type=int, default=32, help="clearance around boxes, pixels")
    parser.add_argument("--quality", type=int, default=90, help="JPEG quality of the crops")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="delete an existing output first")
    args = parser.parse_args(argv)
    if not 0.0 <= args.rate <= 1.0:
        parser.error("--rate must be between 0 and 1")
    if args.crops_per_frame <= 0:
        parser.error("--crops-per-frame must be positive")

    out = dataloader.data_root(f"sparse-{args.source}")
    if out.exists():
        if not args.force:
            raise SystemExit(f"{out} exists; pass --force to replace it")
        shutil.rmtree(out)
    names = dataloader.sequences(args.source)
    if not names:
        raise SystemExit(f"no {args.source} sequences on disk; run scripts/fetch_data.py first")

    rng = random.Random(args.seed)
    totals: Counter[str] = Counter()
    for name in args.sequences or names:
        crops, counts = plan(dataloader.load(args.source, name), args, rng)
        folder = out / name.replace("/", "-")
        write_crops(crops, folder, args.quality)
        totals.update(counts)
        print(
            f"{folder.name}: {counts['empty']} empty and {counts['target']} target crops; "
            f"{counts['skipped empty']} empty and {counts['skipped target']} target did not fit, "
            f"{counts['dropped target']} target dropped to keep the rate"
        )

    kept = totals["empty"] + totals["target"]
    share = totals["target"] / kept if kept else 0.0
    print(f"total: {kept} crops, {totals['target']} with a target ({100 * share:.2f}%)")
    record = {
        "source": args.source,
        "size": list(args.size),
        "rate": args.rate,
        "crops_per_frame": args.crops_per_frame,
        "margin": args.margin,
        "quality": args.quality,
        "seed": args.seed,
        "sequences": args.sequences or names,
        "counts": dict(totals),
        "target_share": share,
    }
    (out / RECORD).write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()

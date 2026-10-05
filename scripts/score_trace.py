"""
Score held-out frames with a fitted tagger, report ranking quality, and
optionally write a replay manifest for the client demo.

Usage:
    uv run python scripts/score_trace.py data/models/seadronessee
    uv run python scripts/score_trace.py BUNDLE --stride 5 --manifest data/demo/manifest.csv

By default this scores the frames ``scripts/fit_patchcore.py`` held out:
the trailing part of each sequence it fitted on, as recorded in the
bundle's ``fit.json``. ``--sequences`` scores other sequences instead,
still from the same point in time onward.

Metrics are per tile, because nearly every frame of these datasets holds a
target, so a frame-level split would have no negatives. A tile is positive
when a ground-truth box overlaps it, negative when no box comes within the
fit margin, and left out when a box is near but not on it, or when a
don't-care region (VisDrone's "ignored" class) comes within the margin. The report
gives AUROC, average precision, and the share of positive tiles that land
in the top 1, 5 and 10 percent of scores, overall and per class.

``--manifest`` writes ``image,priority`` rows in replay order, plus
diagnostic columns, which ``tagurit.sim.image_feeder`` reads as-is, so the
client demo replays the frames with real priorities. Image paths are
relative to the manifest's folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from tagurit.sim import dataloader
from tagurit.sim.metrics import auroc, average_precision, recall_at_fraction
from tagurit.sim.trace import TraceFrame
from tagurit.tagging.bundle import load_bundle
from tagurit.tagging.tagger import FrameScore

FIT_RECORD = "fit.json"
TOP_FRACTIONS = (0.01, 0.05, 0.10)
DONT_CARE = "ignored"


def tile_labels(frame: TraceFrame, result: FrameScore, margin: int) -> list[tuple[float, set[str]]]:
    """
    Pair each scoreable tile's raw score with the labels of boxes on it.

    An empty label set is a negative tile. Tiles with a box within
    ``margin`` but not on them are dropped as ambiguous, as are tiles near
    a don't-care region, which is neither target nor background.

    Parameters:
        - frame (TraceFrame): frame with its ground-truth boxes
        - result (FrameScore): the tagger's per-tile scores for this frame
        - margin (int): clearance a negative tile must have from every box

    Return: (raw score, labels on the tile) for each kept tile
    """
    kept = []
    for tile, score in zip(result.tiles, result.tile_scores, strict=True):
        near = [b for b in frame.boxes if tile.overlaps(b.left, b.top, b.width, b.height, margin)]
        if any(b.label == DONT_CARE for b in near):
            continue
        on = {b.label for b in near if tile.overlaps(b.left, b.top, b.width, b.height)}
        if on or not near:
            kept.append((score, on))
    return kept


def report(tiles: list[tuple[float, set[str]]]) -> None:
    """
    Print ranking metrics over all positive tiles and per class.

    Parameters:
        - tiles (list[tuple[float, set[str]]]): raw score and box labels per tile

    Return: void
    """
    scores = [score for score, _ in tiles]
    positive = [bool(labels) for _, labels in tiles]
    count = sum(positive)
    if count == 0 or count == len(tiles):
        print("tile metrics need both positive and negative tiles")
        return
    print(f"\ntiles: {len(tiles)} scored, {count} positive ({100 * count / len(tiles):.1f}%)")
    header = "".join(f"  top{100 * f:g}%" for f in TOP_FRACTIONS)
    print(f"{'class':24s} {'tiles':>6s}  AUROC     AP{header}")

    classes = sorted({label for _, labels in tiles for label in labels})
    for name in ["any target", *classes]:
        if name == "any target":
            subset = list(zip(scores, positive, strict=True))
        else:
            subset = [(s, name in labels) for s, labels in tiles if name in labels or not labels]
        sub_scores = [s for s, _ in subset]
        sub_labels = [y for _, y in subset]
        recalls = "".join(
            f"  {recall_at_fraction(sub_scores, sub_labels, f):6.3f}" for f in TOP_FRACTIONS
        )
        print(
            f"{name:24s} {sum(sub_labels):6d}  {auroc(sub_scores, sub_labels):.3f}  "
            f"{average_precision(sub_scores, sub_labels):.3f}{recalls}"
        )
        if name == "any target":
            ceiling = "".join(
                f"  {recall_at_fraction(positive, positive, f):6.3f}" for f in TOP_FRACTIONS
            )
            print(f"{'  (perfect ranking)':24s} {'':6s}  {'':5s}  {'':5s}{ceiling}")


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    """
    Write replay rows with image paths relative to the manifest's folder.

    An existing file is OVERWRITTEN.

    Parameters:
        - path (Path): CSV file to write
        - rows (list[dict[str, object]]): per-frame records with an absolute ``image`` path

    Return: void
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    folder = path.parent.resolve()
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            image = Path(str(row["image"])).relative_to(folder, walk_up=True)
            writer.writerow({**row, "image": image.as_posix()})
    print(f"wrote {len(rows)} rows to {path}")


def main(argv: list[str] | None = None) -> None:
    """
    Parse arguments, score the held-out frames, print metrics, write the manifest.

    Parameters:
        - argv (list[str] | None): arguments without the program name; None means sys.argv

    Return: void
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("bundle", type=Path, help="directory written by fit_patchcore.py")
    parser.add_argument("--sequences", nargs="+", help="override the fitted sequences")
    parser.add_argument("--stride", type=int, default=1, help="score every Nth held-out frame")
    parser.add_argument("--limit", type=int, help="stop after this many frames")
    parser.add_argument("--manifest", type=Path, help="write a replay manifest CSV here")
    parser.add_argument("--device", help="torch device; default cuda, then mps, then cpu")
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        help="score in float16 (--half) or float32 (--no-half); default the bundle's setting",
    )
    args = parser.parse_args(argv)
    if args.stride <= 0:
        parser.error("--stride must be positive")

    record = json.loads((args.bundle / FIT_RECORD).read_text())
    names = args.sequences or record["sequences"]
    frames: list[TraceFrame] = []
    for name in names:
        trace = dataloader.load(record["dataset"], name)
        frames.extend(trace.frames[int(len(trace) * record["fit_fraction"]) :: args.stride])
    frames = frames[: args.limit]
    if not frames:
        raise SystemExit("no held-out frames to score")

    tagger = load_bundle(args.bundle, device=args.device, half=args.half)
    device = getattr(tagger.scorer, "device", "unknown device")
    precision = "float16" if getattr(tagger.scorer, "half", False) else "float32"
    print(
        f"scoring {len(frames)} held-out frames from {len(names)} sequences "
        f"on {device} in {precision}"
    )

    rows: list[dict[str, object]] = []
    tiles: list[tuple[float, set[str]]] = []
    millis_by_grid: dict[int, list[float]] = defaultdict(list)
    for count, frame in enumerate(frames, start=1):
        image_bytes = frame.read()
        started = time.perf_counter()
        result = tagger.score(image_bytes)
        millis_by_grid[len(result.tiles)].append(1000 * (time.perf_counter() - started))
        tiles.extend(tile_labels(frame, result, record["margin"]))
        rows.append(
            {
                "image": frame.path.resolve(),
                "priority": f"{result.priority:.6f}",
                "sequence": frame.sequence,
                "frame_index": frame.index,
                "raw_score": f"{result.raw_score:.6f}",
                "boxes": len(frame.boxes),
            }
        )
        if count % 100 == 0:
            print(f"  {count}/{len(frames)} frames")

    priorities = [float(str(row["priority"])) for row in rows]
    print(f"\nframes: {len(rows)}")
    for grid, millis in sorted(millis_by_grid.items()):
        median = statistics.median(millis)
        print(
            f"  {grid:3d} tiles/frame: {median:7.1f} ms median ({median / grid:.1f} ms/tile) "
            f"over {len(millis)} frames"
        )
    print(
        f"  priority min={min(priorities):.3f} median={statistics.median(priorities):.3f} "
        f"max={max(priorities):.3f}"
    )
    report(tiles)

    if args.manifest is not None:
        write_manifest(args.manifest, rows)


if __name__ == "__main__":
    main()

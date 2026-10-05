"""
Set up a zero-shot CLIP tagger for a dataset and save it as a bundle.

Usage:
    uv run python scripts/calibrate_clip.py sparse-seadronessee --out data/models/clip-sds
    uv run python scripts/calibrate_clip.py DATASET --out DIR --targets "a person" "a boat"

CLIP needs no fitting, so this only calibrates: target-free tiles from the
first ``--fit-fraction`` of each sequence are scored, and their
``--percentile`` score becomes the raw score that maps to priority 0.5.
The split matches ``scripts/fit_patchcore.py``, and the same ``fit.json``
is written, so ``scripts/score_trace.py`` scores exactly the frames a
PatchCore bundle with the same split would. The calibration only places
0.5; frame rankings do not depend on it.

Weights download from the Hugging Face Hub on first use. Needs the
``model`` extra: ``uv sync --extra model``.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from tagurit.model.clip import ClipConfig, ClipScorer
from tagurit.sim import dataloader
from tagurit.sim.normal_tiles import sample_normal_tiles
from tagurit.sim.trace import TraceFrame
from tagurit.tagging.bundle import save_clip_bundle
from tagurit.tagging.priority import PriorityScale

FIT_RECORD = "fit.json"


def main(argv: list[str] | None = None) -> None:
    """
    Parse arguments, calibrate CLIP on normal tiles, and save the bundle.

    Parameters:
        - argv (list[str] | None): arguments without the program name; None means sys.argv

    Return: void
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset", choices=sorted(dataloader.DATASETS))
    parser.add_argument("--out", type=Path, required=True, help="bundle directory to write")
    parser.add_argument("--sequences", nargs="+", help="sequence names; default all on disk")
    parser.add_argument(
        "--fit-fraction", type=float, default=0.7, help="leading share per sequence"
    )
    parser.add_argument("--tile-size", type=int, default=512, help="tile side in frame pixels")
    parser.add_argument("--margin", type=int, default=32, help="pixels kept clear around boxes")
    parser.add_argument("--calibration-tiles", type=int, default=300)
    parser.add_argument("--tiles-per-frame", type=int, default=2)
    parser.add_argument("--percentile", type=float, default=99.0, help="calibration score -> 0.5")
    parser.add_argument("--model", default=ClipConfig.model, help="Hugging Face model id")
    parser.add_argument("--targets", nargs="+", help="target prompts; default ClipConfig's")
    parser.add_argument("--backgrounds", nargs="+", help="background prompts; default ClipConfig's")
    parser.add_argument("--device", help="torch device; default cuda, then mps, then cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="score in float16 (default); --no-half for float32",
    )
    args = parser.parse_args(argv)

    if not 0.0 < args.fit_fraction < 1.0:
        parser.error("--fit-fraction must be strictly between 0 and 1")
    names = args.sequences or dataloader.sequences(args.dataset)
    if not names:
        parser.error(f"no {args.dataset} sequences on disk")

    frames: list[TraceFrame] = []
    for name in names:
        trace = dataloader.load(args.dataset, name)
        frames.extend(trace.frames[: int(len(trace) * args.fit_fraction)])
    random.Random(args.seed).shuffle(frames)
    [calibration] = sample_normal_tiles(
        frames,
        [args.calibration_tiles],
        args.tile_size,
        args.margin,
        args.tiles_per_frame,
        args.seed,
    )

    defaults = ClipConfig()
    config = ClipConfig(
        model=args.model,
        targets=tuple(args.targets) if args.targets else defaults.targets,
        backgrounds=tuple(args.backgrounds) if args.backgrounds else defaults.backgrounds,
    )
    started = time.perf_counter()
    scorer = ClipScorer.load(config, device=args.device, half=args.half)
    print(f"loaded {config.model} on {scorer.device} in {time.perf_counter() - started:.1f}s")

    started = time.perf_counter()
    scores = scorer.score_tiles(calibration)
    elapsed = time.perf_counter() - started
    scale = PriorityScale.from_normal_scores(scores, args.percentile)
    print(
        f"calibrated on {len(calibration)} normal tiles in {elapsed:.1f}s "
        f"({1000 * elapsed / len(calibration):.1f} ms/tile): target probability "
        f"p50={np.percentile(scores, 50):.3f} p{args.percentile:g}={scale.reference:.3f} "
        f"max={max(scores):.3f}"
    )

    save_clip_bundle(args.out, scorer, scale, args.tile_size)
    record = {
        "dataset": args.dataset,
        "sequences": names,
        "fit_fraction": args.fit_fraction,
        "margin": args.margin,
        "fit_tiles": 0,
        "calibration_tiles": len(calibration),
        "percentile": args.percentile,
        "seed": args.seed,
    }
    (args.out / FIT_RECORD).write_text(json.dumps(record, indent=2) + "\n")
    print(f"saved bundle to {args.out}")


if __name__ == "__main__":
    main()

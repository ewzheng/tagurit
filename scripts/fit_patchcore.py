"""
Fit a PatchCore tagger on target-free tiles from a dataset and save it as a bundle.

Usage:
    uv run python scripts/fit_patchcore.py seadronessee --out data/models/seadronessee
    uv run python scripts/fit_patchcore.py seadronessee --out DIR --sequences NAME [NAME ...]

Each sequence is split in time: its first ``--fit-fraction`` of frames
supply tiles, and the rest are held out for ``scripts/score_trace.py``. A
tile is normal when no ground-truth box of any class comes within
``--margin`` pixels of it. A frame labelled as holding a target but with no
boxes to place it supplies nothing, since any of its tiles could hold it.
Fit frames are shuffled, and up to ``--tiles-per-frame`` normal tiles are
drawn from each, filling the fitting set first and then a calibration set
from later, disjoint frames. The
calibration set's ``--percentile`` score becomes the raw score that maps to
priority 0.5. Fitting always runs in float32; ``--half`` makes calibration
and later scoring run in float16.

The bundle directory gets ``tagger.json`` and ``patchcore.pt`` from
``tagurit.tagging.bundle``, plus ``fit.json``, which records the split so
the scoring script holds out exactly the frames that were not fitted on.
Fitting needs the ``model`` extra: ``uv sync --extra model``.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from tagurit.model.patchcore import PatchcoreConfig, PatchcoreScorer
from tagurit.sim import dataloader
from tagurit.sim.normal_tiles import sample_normal_tiles
from tagurit.sim.trace import TraceFrame
from tagurit.tagging.bundle import save_patchcore_bundle
from tagurit.tagging.priority import PriorityScale

FIT_RECORD = "fit.json"


def main(argv: list[str] | None = None) -> None:
    """
    Parse arguments, collect normal tiles, fit, calibrate, and save the bundle.

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
    parser.add_argument("--fit-tiles", type=int, default=300)
    parser.add_argument("--calibration-tiles", type=int, default=300)
    parser.add_argument("--tiles-per-frame", type=int, default=2)
    parser.add_argument("--percentile", type=float, default=99.0, help="calibration score -> 0.5")
    parser.add_argument("--backbone", default=PatchcoreConfig.backbone)
    parser.add_argument("--input-size", type=int, default=PatchcoreConfig.input_size)
    parser.add_argument("--coreset-ratio", type=float, default=PatchcoreConfig.coreset_ratio)
    parser.add_argument("--device", help="torch device; default cuda, then mps, then cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--half", action="store_true", help="score in float16 after fitting")
    args = parser.parse_args(argv)

    if not 0.0 < args.fit_fraction < 1.0:
        parser.error("--fit-fraction must be strictly between 0 and 1")
    if min(args.fit_tiles, args.calibration_tiles, args.tiles_per_frame) <= 0:
        parser.error("--fit-tiles, --calibration-tiles and --tiles-per-frame must be positive")
    names = args.sequences or dataloader.sequences(args.dataset)
    if not names:
        parser.error(f"no {args.dataset} sequences on disk; run scripts/fetch_data.py first")

    frames: list[TraceFrame] = []
    for name in names:
        trace = dataloader.load(args.dataset, name)
        frames.extend(trace.frames[: int(len(trace) * args.fit_fraction)])
    random.Random(args.seed).shuffle(frames)
    print(f"{len(names)} sequences, {len(frames)} fit frames")

    started = time.perf_counter()
    fit, calibration = sample_normal_tiles(
        frames,
        [args.fit_tiles, args.calibration_tiles],
        args.tile_size,
        args.margin,
        args.tiles_per_frame,
        args.seed,
    )
    print(
        f"collected {len(fit)} + {len(calibration)} normal tiles in "
        f"{time.perf_counter() - started:.1f}s"
    )

    config = PatchcoreConfig(
        backbone=args.backbone, input_size=args.input_size, coreset_ratio=args.coreset_ratio
    )
    started = time.perf_counter()
    scorer = PatchcoreScorer.fit(
        fit, config=config, device=args.device, seed=args.seed, half=args.half
    )
    print(
        f"fitted on {scorer.device} ({'float16' if scorer.half else 'float32'} scoring): "
        f"bank of {scorer.bank_size} embeddings in "
        f"{time.perf_counter() - started:.1f}s"
    )

    started = time.perf_counter()
    scores = scorer.score_tiles(calibration)
    elapsed = time.perf_counter() - started
    scale = PriorityScale.from_normal_scores(scores, args.percentile)
    print(
        f"calibrated in {elapsed:.1f}s ({1000 * elapsed / len(calibration):.1f} ms/tile): "
        f"normal scores p50={np.percentile(scores, 50):.3f} "
        f"p{args.percentile:g}={scale.reference:.3f} max={max(scores):.3f}"
    )

    save_patchcore_bundle(args.out, scorer, scale, args.tile_size)
    record = {
        "dataset": args.dataset,
        "sequences": names,
        "fit_fraction": args.fit_fraction,
        "margin": args.margin,
        "fit_tiles": len(fit),
        "calibration_tiles": len(calibration),
        "percentile": args.percentile,
        "seed": args.seed,
    }
    (args.out / FIT_RECORD).write_text(json.dumps(record, indent=2) + "\n")
    print(f"saved bundle to {args.out}")


if __name__ == "__main__":
    main()

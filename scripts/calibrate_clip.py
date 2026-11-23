"""
Set up a query-only CLIP tagger with window novelty for a dataset and save it as a bundle.

Usage:
    uv run python scripts/calibrate_clip.py sparse-seadronessee --out data/models/clip-sds
    uv run python scripts/calibrate_clip.py DATASET --out DIR --targets "a swimmer" "a boat"

CLIP needs no fitting, so this only calibrates (``tagurit.model.clip``):
target-free tiles from the first ``--fit-fraction`` of each sequence are
embedded and saved with the bundle, and the query score is put on a robust
z scale against them, both now and whenever the query changes. Their raw
scores, with novelty neutral, set the priority scale: the ``--percentile``
score maps to priority 0.5. The split matches ``scripts/fit_patchcore.py``,
and the same ``fit.json`` is written, so ``scripts/score_trace.py`` scores
exactly the frames a PatchCore bundle with the same split would. The
calibration only scales; frame rankings do not depend on the priority
reference.

The detection threshold for ``tagurit.tagging.order`` is calibrated too:
the leading ``--detection-frames`` frames of the fit split are scored in
capture order, novelty included, as the scout would score them, and the
``--detection-percentile`` of the priorities of those holding no target
becomes the threshold. ``--detection-frames 0`` skips it.

Weights download from the Hugging Face Hub on first use. Needs the
``model`` extra: ``uv sync --extra model``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from tagurit.model.clip import ClipConfig, ClipScorer
from tagurit.sim import dataloader
from tagurit.sim.normal_tiles import sample_normal_tiles
from tagurit.sim.trace import TraceFrame
from tagurit.tagging.bundle import save_clip_bundle
from tagurit.tagging.order import detection_threshold
from tagurit.tagging.priority import PriorityScale
from tagurit.tagging.tagger import Tagger

FIT_RECORD = "fit.json"
DONT_CARE = "ignored"


def normal_frame_priorities(
    tagger: Tagger, sequences: Sequence[Sequence[TraceFrame]], limit: int
) -> list[float]:
    """
    Priorities of target-free frames, scored in capture order as the scout would.

    Scores an equal share of ``limit`` from the start of every sequence,
    targets included so the novelty window sees what a flight shows, and
    keeps the frames labelled as holding no target. MUTATES the tagger's
    scorer, which is reset before each sequence.

    Parameters:
        - tagger (Tagger): the calibrated tagger
        - sequences (Sequence[Sequence[TraceFrame]]): each sequence's frames in capture order
        - limit (int): frames to score in total

    Return: one priority per target-free frame scored
    """
    share = math.ceil(limit / len(sequences))
    priorities: list[float] = []
    for frames in sequences:
        tagger.reset()
        for frame in frames[:share]:
            priority = tagger.score(frame.read(), frame.timestamp).priority
            if frame.has_target((DONT_CARE,)) is False:
                priorities.append(priority)
    return priorities


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
    defaults = ClipConfig()
    parser.add_argument("dataset", choices=sorted(dataloader.DATASETS))
    parser.add_argument("--out", type=Path, required=True, help="bundle directory to write")
    parser.add_argument("--sequences", nargs="+", help="sequence names; default all on disk")
    parser.add_argument(
        "--fit-fraction", type=float, default=0.7, help="leading share per sequence"
    )
    parser.add_argument("--tile-size", type=int, default=224, help="tile side in frame pixels")
    parser.add_argument("--margin", type=int, default=32, help="pixels kept clear around boxes")
    parser.add_argument("--calibration-tiles", type=int, default=300)
    parser.add_argument("--tiles-per-frame", type=int, default=2)
    parser.add_argument("--percentile", type=float, default=99.0, help="calibration score -> 0.5")
    parser.add_argument(
        "--detection-frames", type=int, default=2000, help="frames scored for the threshold"
    )
    parser.add_argument(
        "--detection-percentile",
        type=float,
        default=99.5,
        help="share of target-free frames below the detection threshold",
    )
    parser.add_argument("--model", default=defaults.model, help="Hugging Face model id")
    parser.add_argument("--targets", nargs="+", help="query phrases; default ClipConfig's")
    parser.add_argument(
        "--window", type=float, default=defaults.window, help="novelty window, seconds"
    )
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
    sequences: list[tuple[TraceFrame, ...]] = []
    for name in names:
        trace = dataloader.load(args.dataset, name)
        sequences.append(trace.split(args.fit_fraction)[0])
        frames.extend(sequences[-1])
    random.Random(args.seed).shuffle(frames)
    [calibration] = sample_normal_tiles(
        frames,
        [args.calibration_tiles],
        args.tile_size,
        args.margin,
        args.tiles_per_frame,
        args.seed,
    )

    config = ClipConfig(
        model=args.model,
        targets=tuple(args.targets) if args.targets else defaults.targets,
        window=args.window,
    )
    started = time.perf_counter()
    scorer = ClipScorer.load(config, device=args.device, half=args.half)
    print(f"loaded {config.model} on {scorer.device} in {time.perf_counter() - started:.1f}s")

    started = time.perf_counter()
    scorer.calibrate(calibration)
    scores = scorer.score_tiles(calibration)
    elapsed = time.perf_counter() - started
    scale = PriorityScale.from_normal_scores(scores, args.percentile)
    reference = scorer.query.reference
    print(
        f"calibrated on {len(calibration)} normal tiles in {elapsed:.1f}s "
        f"({1000 * elapsed / len(calibration):.1f} ms/tile per pass): query score "
        f"median {reference.median:.4f}, spread {reference.spread:.4f}; raw score "
        f"p50={np.percentile(scores, 50):.3f} p{args.percentile:g}={scale.reference:.3f} "
        f"max={max(scores):.3f}"
    )

    detection = None
    normal: list[float] = []
    if args.detection_frames > 0:
        started = time.perf_counter()
        tagger = Tagger(scorer, scale, args.tile_size)
        normal = normal_frame_priorities(tagger, sequences, args.detection_frames)
        detection = detection_threshold(normal, args.detection_percentile)
        tagger.reset()
        print(
            f"detection threshold {detection:.4f}: p{args.detection_percentile:g} of "
            f"{len(normal)} target-free frames' priorities "
            f"(median {np.median(normal):.4f}, max {max(normal):.4f}) "
            f"in {time.perf_counter() - started:.1f}s"
        )

    save_clip_bundle(args.out, scorer, scale, args.tile_size, detection)
    record = {
        "dataset": args.dataset,
        "sequences": names,
        "fit_fraction": args.fit_fraction,
        "margin": args.margin,
        "fit_tiles": 0,
        "calibration_tiles": len(calibration),
        "percentile": args.percentile,
        "detection_frames": len(normal),
        "detection_percentile": args.detection_percentile,
        "seed": args.seed,
    }
    (args.out / FIT_RECORD).write_text(json.dumps(record, indent=2) + "\n")
    print(f"saved bundle to {args.out}")


if __name__ == "__main__":
    main()

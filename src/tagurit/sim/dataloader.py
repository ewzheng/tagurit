"""
VisDrone-MOT on disk, loaded as ``trace`` types.

Expects the layout ``scripts/fetch_visdrone.py`` produces:

    <root>/
      annotations/<sequence>.txt
      sequences/<sequence>/0000001.jpg, 0000002.jpg, ...

One annotation line per box, ten comma-separated integers:

    frame_index,target_id,left,top,width,height,score,category,truncation,occlusion

The four box fields are absolute pixels from the image's top-left corner,
the same xywh convention as COCO, and are passed through to ``Box``
unchanged. Frames with no objects have no lines, so the frame list ALWAYS comes from
the directory, never from the annotation file. Score is 1 in ground truth
and is dropped. Every category is kept, including 0 (ignored regions).
Consumers filter.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from tagurit.sim.trace import Box, Trace, TraceFrame

CATEGORIES: dict[int, str] = {
    0: "ignored",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
    11: "others",
}

_DATA_DIR = Path("data") / "visdrone"
_FIELDS = 10


def default_root(split: str = "val") -> Path:
    """
    Where the fetch script puts a split, relative to the current directory.

    Parameters:
        - split (str): "val", "train", or "test-dev"

    Return: Path to ``data/visdrone/VisDrone2019-MOT-<split>``
    """
    return _DATA_DIR / f"VisDrone2019-MOT-{split}"


def sequences(root: Path) -> list[str]:
    """
    Names of the sequences present under ``root``.

    Parameters:
        - root (Path): a split folder, see ``default_root``

    Return: sorted directory names under ``root/sequences``
    """
    seq_dir = root / "sequences"
    if not seq_dir.is_dir():
        raise FileNotFoundError(f"no sequences directory at {seq_dir}")
    return sorted(p.name for p in seq_dir.iterdir() if p.is_dir())


def load(root: Path, name: str, fps: float = 30.0) -> Trace:
    """
    Load one sequence as a Trace.

    Lists the JPEGs under ``root/sequences/<name>``, parses
    ``root/annotations/<name>.txt`` once, and joins them by frame index.
    Nothing is read from the JPEGs here; frames read lazily.

    RAISES FileNotFoundError when the sequence directory or its annotation
    file is missing, and ValueError, naming the file and line, when a line
    is malformed or refers to a frame that has no file on disk. A bad
    dataset fails loudly here rather than as a silently short trace.

    Parameters:
        - root (Path): a split folder, see ``default_root``
        - name (str): sequence name, e.g. "uav0000086_00000_v"
        - fps (float): nominal frame rate used to derive timestamps

    Return: Trace with frames in ascending index order
    """
    seq_dir = root / "sequences" / name
    ann_file = root / "annotations" / f"{name}.txt"
    if not seq_dir.is_dir():
        raise FileNotFoundError(f"no sequence directory at {seq_dir}")
    if not ann_file.is_file():
        raise FileNotFoundError(f"no annotation file at {ann_file}")

    paths = {int(p.stem): p for p in seq_dir.glob("*.jpg")}
    boxes = _parse_annotations(ann_file)
    orphans = sorted(set(boxes) - set(paths))
    if orphans:
        raise ValueError(f"{ann_file}: annotations for frames with no file: {orphans[:5]}")
    frames = tuple(
        TraceFrame(
            sequence=name,
            index=index,
            timestamp=(index - 1) / fps,
            path=paths[index],
            boxes=tuple(boxes.get(index, ())),
        )
        for index in sorted(paths)
    )
    return Trace(name=name, fps=fps, frames=frames)


def _parse_annotations(ann_file: Path) -> dict[int, list[Box]]:
    """
    Parse a VisDrone-MOT annotation file into boxes keyed by frame index.

    Blank lines and a trailing comma are tolerated because the published
    files contain both. Anything else that is not ten integers with a
    known category RAISES ValueError naming the file and line.

    Parameters:
        - ann_file (Path): the ``annotations/<sequence>.txt`` file

    Return: dict mapping frame index to its boxes, in file order
    """
    by_frame: dict[int, list[Box]] = defaultdict(list)
    with ann_file.open() as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip().rstrip(",")
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != _FIELDS:
                raise ValueError(
                    f"{ann_file}:{lineno}: expected {_FIELDS} fields, got {len(parts)}"
                )
            try:
                values = [int(part) for part in parts]
            except ValueError as exc:
                raise ValueError(f"{ann_file}:{lineno}: non-integer field in {line!r}") from exc
            frame, track_id, left, top, width, height, _score, category, trunc, occ = values
            if category not in CATEGORIES:
                raise ValueError(f"{ann_file}:{lineno}: unknown category {category}")
            by_frame[frame].append(
                Box(
                    label=CATEGORIES[category],
                    left=left,
                    top=top,
                    width=width,
                    height=height,
                    track_id=track_id,
                    truncation=trunc,
                    occlusion=occ,
                )
            )
    return dict(by_frame)

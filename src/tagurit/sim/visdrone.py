"""
VisDrone-MOT on disk, parsed into ``trace`` types.

Expects what ``scripts/fetch_data.py visdrone`` produces under
``data/visdrone``, one folder per split:

    <root>/VisDrone2019-MOT-<split>/
      annotations/<sequence>.txt
      sequences/<sequence>/0000001.jpg, 0000002.jpg, ...

Sequence names are ``<split>/<sequence>``, e.g. ``val/uav0000086_00000_v``,
because each split holds different sequences.

One annotation line per box, ten comma-separated integers:

    frame_index,target_id,left,top,width,height,score,category,truncation,occlusion

The four box fields are absolute pixels from the image's top-left corner,
the same xywh convention as COCO, and are passed through to ``Box``
unchanged. Frames with no objects have no lines, so the frame list ALWAYS
comes from the directory, never from the annotation file. Score is 1 in
ground truth and is dropped. Every category is kept, including 0 (ignored
regions). Consumers filter.

Labels are COCO class names where one exists (``LABELS``): pedestrian and
people become person, van becomes car, and motor becomes motorcycle.
Tricycle, awning-tricycle, others, and ignored have no COCO equivalent and
keep their names. The VisDrone name is kept in ``Box.dataset_label``.
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

LABELS: dict[str, str] = {
    "ignored": "ignored",
    "pedestrian": "person",
    "people": "person",
    "bicycle": "bicycle",
    "car": "car",
    "van": "car",
    "truck": "truck",
    "tricycle": "tricycle",
    "awning-tricycle": "awning-tricycle",
    "bus": "bus",
    "motor": "motorcycle",
    "others": "others",
}

PREFIX = "VisDrone2019-MOT-"
_FIELDS = 10


def sequences(root: Path) -> list[str]:
    """
    Names of the sequences present under ``root``, across every split folder.

    Parameters:
        - root (Path): the dataset folder holding ``VisDrone2019-MOT-*`` splits

    Return: sorted ``<split>/<sequence>`` names
    """
    splits = sorted(p for p in root.glob(f"{PREFIX}*") if p.is_dir())
    if not splits:
        raise FileNotFoundError(f"no {PREFIX}* split folders under {root}")
    names: list[str] = []
    for split_dir in splits:
        seq_dir = split_dir / "sequences"
        if not seq_dir.is_dir():
            raise FileNotFoundError(f"no sequences directory at {seq_dir}")
        split = split_dir.name.removeprefix(PREFIX)
        names.extend(f"{split}/{p.name}" for p in sorted(seq_dir.iterdir()) if p.is_dir())
    return names


def load(root: Path, name: str, fps: float = 30.0) -> Trace:
    """
    Load one sequence as a Trace.

    Lists the JPEGs under the sequence directory, parses its annotation file
    once, and joins them by frame index. Nothing is read from the JPEGs
    here; frames read lazily.

    RAISES ValueError when ``name`` lacks the split prefix, FileNotFoundError
    when the sequence directory or its annotation file is missing, and
    ValueError, naming the file and line, when a line is malformed or refers
    to a frame that has no file on disk. A bad dataset fails loudly here
    rather than as a silently short trace.

    Parameters:
        - root (Path): the dataset folder holding ``VisDrone2019-MOT-*`` splits
        - name (str): ``<split>/<sequence>``, e.g. "val/uav0000086_00000_v"
        - fps (float): nominal frame rate used to derive timestamps

    Return: Trace with frames in ascending index order
    """
    split, _, seq = name.partition("/")
    if not split or not seq:
        raise ValueError(f"sequence name must be <split>/<sequence>, got {name!r}")
    split_dir = root / f"{PREFIX}{split}"
    seq_dir = split_dir / "sequences" / seq
    ann_file = split_dir / "annotations" / f"{seq}.txt"
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
            name = CATEGORIES[category]
            by_frame[frame].append(
                Box(
                    label=LABELS[name],
                    dataset_label=name,
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

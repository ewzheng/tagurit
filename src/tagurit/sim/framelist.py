"""
Frame-labelled image sequences on disk, parsed into ``trace`` types.

The format for datasets labelled per frame ("this frame holds a target")
rather than per box: the synthetic sparse streams ``scripts/make_sparse.py``
writes, and any frame-labelled set converted to it. One folder per sequence:

    <root>/<sequence>/frames.csv   file,target[,timestamp]  one row per frame, in order
    <root>/<sequence>/boxes.csv    optional: file,label,dataset_label,left,top,width,height
    <root>/<sequence>/source.json  optional: {"times": [...]} of the source recording
    <root>/<sequence>/<file>       the images

``target`` is 1 or 0 and becomes ``TraceFrame.target``. Without a
``timestamp`` column, frame i (one-based) is at (i - 1) / fps. Frame
indices are row numbers starting at one. Boxes are optional, follow the
``trace.Box`` xywh convention in absolute pixels, and leave track,
truncation, and occlusion as None. A malformed row, a missing column, or a
box naming a file that is not in ``frames.csv`` RAISES ValueError.
``source.json`` gives the frame times of the recording the frames were cut
from, which ``Trace.split`` uses to split them at the source's moments.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from tagurit.sim.trace import Box, Trace, TraceFrame

FRAMES_FILE = "frames.csv"
BOXES_FILE = "boxes.csv"
SOURCE_FILE = "source.json"
BOX_COLUMNS = ("file", "label", "dataset_label", "left", "top", "width", "height")


def sequences(root: Path) -> list[str]:
    """
    Names of the sequence folders under ``root`` that hold a ``frames.csv``.

    Parameters:
        - root (Path): the dataset folder

    Return: sorted sequence names
    """
    return sorted(path.parent.name for path in root.glob(f"*/{FRAMES_FILE}"))


def load(root: Path, name: str, fps: float = 30.0) -> Trace:
    """
    Load one sequence as a Trace.

    Parameters:
        - root (Path): the dataset folder
        - name (str): sequence folder name
        - fps (float): rate used for timestamps when ``frames.csv`` has none

    Return: Trace with frames in file order
    """
    folder = root / name
    boxes = _boxes(folder / BOXES_FILE)
    frames = []
    with (folder / FRAMES_FILE).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        columns = set(reader.fieldnames or [])
        if not {"file", "target"} <= columns:
            raise ValueError(f"{folder / FRAMES_FILE} needs 'file' and 'target' columns")
        for index, row in enumerate(reader, start=1):
            if row["target"] not in ("0", "1"):
                raise ValueError(f"{folder / FRAMES_FILE} row {index}: target must be 0 or 1")
            timestamp = float(row["timestamp"]) if "timestamp" in columns else (index - 1) / fps
            frames.append(
                TraceFrame(
                    sequence=name,
                    index=index,
                    timestamp=timestamp,
                    path=folder / row["file"],
                    boxes=tuple(boxes.pop(row["file"], ())),
                    target=row["target"] == "1",
                )
            )
    if boxes:
        raise ValueError(f"{folder / BOXES_FILE} names files not in {FRAMES_FILE}: {sorted(boxes)}")
    source = folder / SOURCE_FILE
    times = (
        tuple(float(t) for t in json.loads(source.read_text())["times"])
        if source.is_file()
        else None
    )
    return Trace(name=name, fps=fps, frames=tuple(frames), source_times=times)


def _boxes(path: Path) -> dict[str, list[Box]]:
    """
    Read the optional boxes file, grouped by image file name.

    Parameters:
        - path (Path): ``boxes.csv``, which may not exist

    Return: boxes per file name; empty when the file is absent
    """
    grouped: dict[str, list[Box]] = defaultdict(list)
    if not path.is_file():
        return grouped
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if not set(BOX_COLUMNS) <= set(reader.fieldnames or []):
            raise ValueError(f"{path} needs columns {BOX_COLUMNS}")
        for row in reader:
            grouped[row["file"]].append(
                Box(
                    label=row["label"],
                    dataset_label=row["dataset_label"],
                    left=int(row["left"]),
                    top=int(row["top"]),
                    width=int(row["width"]),
                    height=int(row["height"]),
                    track_id=None,
                    truncation=None,
                    occlusion=None,
                )
            )
    return grouped

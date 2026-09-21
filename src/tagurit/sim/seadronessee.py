"""
SeaDronesSee (object detection v2) on disk, parsed into ``trace`` types.

Expects the Hugging Face mirror layout that ``scripts/fetch_data.py
seadronessee`` produces under ``data/seadronessee``:

    <root>/unpacked/annotations/instances_train.json, instances_val.json
    <root>/unpacked/images/train/<id>.jpg
    <root>/unpacked/images/val/<id>.jpg

The detection set is drone video sampled every 15th frame. Each image entry
names its source video and frame number, so a sequence here is one source
video ordered by frame number, with frames drawn from BOTH splits because
the official split interleaves frames of the same videos. Images without a
source video (stills from a fixed-wing camera) are skipped.

Boxes are COCO xywh in absolute pixels, rounded to integers. Labels are the
dataset's names: swimmer, boat, jetski, life_saving_appliances, buoy, and
ignored. Track ids, truncation, and occlusion are not annotated and come
through as None.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from tagurit.sim.trace import Box, Trace, TraceFrame

SPLITS = ("train", "val")
SOURCE_FPS = 30.0


@dataclass(frozen=True)
class ImageRecord:
    """
    One video-derived image from the annotation files, before it becomes a frame.

    Parameters:
        - split (str): which images folder holds the file, "train" or "val"
        - file_name (str): the JPEG name, e.g. "3388.jpg"
        - frame_no (int): frame number in the source video
        - boxes (tuple[Box, ...]): ground truth on this image
    """

    split: str
    file_name: str
    frame_no: int
    boxes: tuple[Box, ...]

    def path(self, root: Path) -> Path:
        """
        Where this image lives on disk.

        Parameters:
            - root (Path): the dataset folder

        Return: Path to the JPEG
        """
        return root / "unpacked" / "images" / self.split / self.file_name


def index(root: Path) -> dict[str, list[ImageRecord]]:
    """
    Every video in the annotation files, with its images in frame order.

    Parses both split files once per process and caches the result, since
    the fetch script, ``sequences``, and ``load`` all need it and the
    training file is tens of megabytes. Images that share a video name but
    come from a different drone or folder RAISE, because the name is the
    sequence key and would silently merge two videos.

    Parameters:
        - root (Path): the dataset folder

    Return: dict from video name (the file stem) to its images sorted by frame number
    """
    return _index(root.resolve())


@cache
def _index(root: Path) -> dict[str, list[ImageRecord]]:
    videos: dict[str, list[ImageRecord]] = defaultdict(list)
    origin: dict[str, tuple[str, str]] = {}
    found = False
    for split in SPLITS:
        ann_file = root / "unpacked" / "annotations" / f"instances_{split}.json"
        if not ann_file.is_file():
            continue
        found = True
        data = json.loads(ann_file.read_text())
        names = {c["id"]: c["name"] for c in data["categories"]}
        boxes: dict[int, list[Box]] = defaultdict(list)
        for ann in data["annotations"]:
            left, top, width, height = (round(v) for v in ann["bbox"])
            boxes[ann["image_id"]].append(
                Box(
                    label=names[ann["category_id"]],
                    left=left,
                    top=top,
                    width=width,
                    height=height,
                    track_id=None,
                    truncation=None,
                    occlusion=None,
                )
            )
        for image in data["images"]:
            source = image.get("source", {})
            if "video" not in source:
                continue
            video = Path(source["video"]).stem
            where = (source.get("drone", ""), source.get("folder_name", ""))
            if origin.setdefault(video, where) != where:
                raise ValueError(
                    f"{ann_file}: video {video!r} appears under both {origin[video]} and {where}"
                )
            videos[video].append(
                ImageRecord(
                    split=split,
                    file_name=image["file_name"],
                    frame_no=int(source["frame_no"]),
                    boxes=tuple(boxes.get(image["id"], ())),
                )
            )
    if not found:
        raise FileNotFoundError(f"no instances_*.json under {root / 'unpacked' / 'annotations'}")
    return {v: sorted(ims, key=lambda im: im.frame_no) for v, ims in sorted(videos.items())}


def sequences(root: Path) -> list[str]:
    """
    Names of the videos whose every frame is on disk.

    A video that was never fetched, or only partly, is left out rather than
    offered as a short trace.

    Parameters:
        - root (Path): the dataset folder

    Return: sorted video names
    """
    return [v for v, ims in index(root).items() if all(im.path(root).is_file() for im in ims)]


def load(root: Path, name: str, fps: float = SOURCE_FPS) -> Trace:
    """
    Load one video as a Trace.

    Frames are the sampled images in frame-number order. The timestamp is
    the frame number over ``fps``, so at the source rate of 30 the default
    spacing is half a second. Nothing is read from the JPEGs here.

    RAISES FileNotFoundError when the video is not in the annotations or any
    of its frames is missing on disk.

    Parameters:
        - root (Path): the dataset folder
        - name (str): video name, e.g. "DJI_0032"
        - fps (float): source frame rate used to turn frame numbers into seconds

    Return: Trace with frames in ascending frame-number order
    """
    videos = index(root)
    if name not in videos:
        raise FileNotFoundError(f"no video {name!r} in the annotations under {root}")
    missing = [im.file_name for im in videos[name] if not im.path(root).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{name}: {len(missing)} frames not on disk, e.g. {missing[:3]}; fetch it first"
        )
    frames = tuple(
        TraceFrame(
            sequence=name,
            index=im.frame_no,
            timestamp=im.frame_no / fps,
            path=im.path(root),
            boxes=im.boxes,
        )
        for im in videos[name]
    )
    return Trace(name=name, fps=fps, frames=frames)

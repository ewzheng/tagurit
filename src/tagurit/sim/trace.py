"""
Format-agnostic types for a replayed image trace.

A trace is one video sequence on disk: ordered frames, each with a nominal
timestamp and the ground-truth boxes annotated on it. These types carry NO
image data. A frame knows where its JPEG lives and reads it on demand, so
building a Trace for a thousand-frame sequence costs a directory listing
and one text parse, not a gigabyte of memory.

Dataset parsers (see ``dataloader``) produce these types. Everything
downstream in ``sim`` consumes them.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Box:
    """
    One ground-truth bounding box on one frame.

    COCO-style xywh: ``left`` and ``top`` are the top-left corner and
    ``width`` and ``height`` extend right and down, all in absolute pixels
    with the origin at the image's top-left corner. NOT normalized, NOT a
    center point (YOLO), NOT two corners (Pascal VOC). The right edge is
    ``left + width`` and the bottom edge is ``top + height``.

    ``label`` is a dataset class name such as "pedestrian". Parsers keep
    every class the dataset defines, including don't-care regions, and
    leave filtering to the consumer.

    Parameters:
        - label (str): dataset class name
        - left (int): x of the left edge, pixels
        - top (int): y of the top edge, pixels
        - width (int): box width, pixels
        - height (int): box height, pixels
        - track_id (int): identity of the object across frames
        - truncation (int): 0 fully inside the frame, 1 partly outside
        - occlusion (int): 0 none, 1 partial, 2 heavy
    """

    label: str
    left: int
    top: int
    width: int
    height: int
    track_id: int
    truncation: int
    occlusion: int


@dataclass(frozen=True)
class TraceFrame:
    """
    One frame of a trace: where its JPEG is, when it nominally occurred,
    and what is annotated on it.

    Holds a path, NEVER bytes. ``read()`` hits the disk on every call, so a
    caller that needs the bytes twice should keep them.

    Parameters:
        - sequence (str): name of the sequence this frame belongs to
        - index (int): one-based frame number, as on disk
        - timestamp (float): seconds since the first frame of the sequence
        - path (Path): the JPEG file
        - boxes (tuple[Box, ...]): ground truth on this frame, possibly empty
    """

    sequence: str
    index: int
    timestamp: float
    path: Path
    boxes: tuple[Box, ...]

    def read(self) -> bytes:
        """
        Read the frame's JPEG from disk.

        Return: the file contents, unmodified
        """
        return self.path.read_bytes()


@dataclass(frozen=True)
class Trace:
    """
    One sequence: its frames in ascending index order.

    Iterating a Trace yields its frames in order and NEVER sleeps. Pacing
    playback is the caller's job. ``fps`` is recorded so a pacer has it
    and so the frames' timestamps are explained.

    Parameters:
        - name (str): sequence name
        - fps (float): nominal frame rate the timestamps were derived from
        - frames (tuple[TraceFrame, ...]): every frame, ascending by index
    """

    name: str
    fps: float
    frames: tuple[TraceFrame, ...]

    def __iter__(self) -> Iterator[TraceFrame]:
        return iter(self.frames)

    def __len__(self) -> int:
        return len(self.frames)

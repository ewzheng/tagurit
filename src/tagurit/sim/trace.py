"""
Format-agnostic types for a replayed image trace.

A trace is one video sequence on disk: ordered frames, each with a nominal
timestamp, the ground-truth boxes annotated on it, and, for datasets
labelled per frame rather than per box, whether it holds a target. These types carry NO
image data. A frame knows where its JPEG lives and reads it on demand, so
building a Trace for a thousand-frame sequence costs a directory listing
and one text parse, not a gigabyte of memory.

Dataset parsers (``visdrone``, ``seadronessee``, ``framelist``) produce these types
and ``dataloader`` hands them out. Everything downstream in ``sim``
consumes them without knowing which dataset they came from: frame
spacing comes from ``timestamp``, never from an assumed rate, and labels
are COCO class names wherever the dataset's class has one, so ground truth
compares directly against a COCO-trained detector.
"""

from __future__ import annotations

from collections.abc import Collection, Iterator
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

    ``label`` is the COCO class name when the dataset's class has a COCO
    equivalent, so VisDrone's "pedestrian" and SeaDronesSee's "swimmer" both
    arrive as "person". A class with no equivalent, such as "buoy" or the
    don't-care "ignored", keeps its dataset name. ``dataset_label`` is ALWAYS
    the dataset's own name, unchanged. Parsers keep every class the dataset
    defines, including don't-care regions, and leave filtering to the
    consumer. The last three fields are None when the dataset does not
    annotate them.

    Parameters:
        - label (str): COCO class name, or the dataset's name when COCO has none
        - dataset_label (str): the dataset's own class name
        - left (int): x of the left edge, pixels
        - top (int): y of the top edge, pixels
        - width (int): box width, pixels
        - height (int): box height, pixels
        - track_id (int | None): identity of the object across frames
        - truncation (int | None): 0 fully inside the frame, 1 partly outside
        - occlusion (int | None): 0 none, 1 partial, 2 heavy
    """

    label: str
    dataset_label: str
    left: int
    top: int
    width: int
    height: int
    track_id: int | None
    truncation: int | None
    occlusion: int | None


@dataclass(frozen=True)
class TraceFrame:
    """
    One frame of a trace: where its JPEG is, when it nominally occurred,
    and what is annotated on it.

    Holds a path, NEVER bytes. ``read()`` hits the disk on every call, so a
    caller that needs the bytes twice should keep them.

    Parameters:
        - sequence (str): name of the sequence this frame belongs to
        - index (int): frame number as the dataset counts it; VisDrone is
          one-based and dense, SeaDronesSee is the source video's frame
          number and steps by 15
        - timestamp (float): seconds since the first frame of the sequence
        - path (Path): the JPEG file
        - boxes (tuple[Box, ...]): ground truth on this frame, possibly empty
        - target (bool | None): frame-level ground truth, whether the frame
          holds a target; None when the dataset annotates boxes only
    """

    sequence: str
    index: int
    timestamp: float
    path: Path
    boxes: tuple[Box, ...]
    target: bool | None = None

    def has_target(self, dont_care: Collection[str] = ("ignored",)) -> bool | None:
        """
        Whether this frame holds something worth sending.

        An explicit frame-level ``target`` wins. Otherwise any box whose label
        is not a don't-care class makes it True and no boxes make it False.
        A frame whose only boxes are don't-care regions is ambiguous: None.

        Parameters:
            - dont_care (Collection[str]): labels that mark regions to ignore

        Return: True, False, or None when the frame cannot be called
        """
        if self.target is not None:
            return self.target
        if any(box.label not in dont_care for box in self.boxes):
            return True
        return None if self.boxes else False

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

    A trace drawn from a longer recording, such as a stream of crops cut
    from a video, keeps that recording's frame times in ``source_times`` so
    it splits at the same moment as the recording does (see ``split``).

    Parameters:
        - name (str): sequence name
        - fps (float): nominal frame rate the timestamps were derived from
        - frames (tuple[TraceFrame, ...]): every frame, ascending by index
        - source_times (tuple[float, ...] | None): timestamps of every frame of
          the source recording; None when the frames are the recording itself
    """

    name: str
    fps: float
    frames: tuple[TraceFrame, ...]
    source_times: tuple[float, ...] | None = None

    def __iter__(self) -> Iterator[TraceFrame]:
        return iter(self.frames)

    def __len__(self) -> int:
        return len(self.frames)

    def split(self, fraction: float) -> tuple[tuple[TraceFrame, ...], tuple[TraceFrame, ...]]:
        """
        Split frames in time: the leading ``fraction`` of the recording, and the rest.

        The cut is the timestamp of recording frame ``int(n * fraction)``, n
        being the number of frames in the recording (``source_times``, else
        this trace's own frames). Frames before it lead and the rest trail.
        On a trace that is its own recording this equals cutting the frame
        list at that index. On a crop stream it cuts at the same moment as
        the source does, and keeps every crop of one source frame on the
        same side, so a model fitted on one side of either never sees a
        frame scored from the other. A fraction outside [0, 1] RAISES
        ValueError.

        Parameters:
            - fraction (float): leading share of the recording, 0 to 1

        Return: (leading frames, trailing frames), each in trace order
        """
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"fraction must be between 0 and 1, got {fraction}")
        times = sorted(
            self.source_times if self.source_times is not None else (f.timestamp for f in self)
        )
        cut = int(len(times) * fraction)
        if cut >= len(times):
            return self.frames, ()
        cutoff = times[cut]
        lead = tuple(f for f in self.frames if f.timestamp < cutoff)
        trail = tuple(f for f in self.frames if f.timestamp >= cutoff)
        return lead, trail

"""
Draw the end-to-end demo: the latest frame, the bank in send order, and what was sent.

``Tracker`` follows the client's report events to know where each frame is:
in the live queue, in the bank, in flight, or acknowledged. It reads only
the event text and the scheduler's public counts, never its internals, so
a change to the client's event wording shows up here. ``FrameView`` holds
what the tagger saw for one frame, and ``render`` draws one dashboard image
from both. Drawing is plain OpenCV on numpy arrays; the caller decides
whether to show it in a window or write it to a video.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import cv2
import numpy as np

from tagurit.client.frame_scheduler import FrameScheduler

ARRIVE = re.compile(r"\bFrame (\d+) -> (live|bank)\b")
MIGRATE = re.compile(r"\bFrame (\d+) live -> bank\b")
FRAME_ID = re.compile(r"\bFrame (\d+)\b")

SIZE = (1280, 720)
PREVIEW = (840, 430)
THUMB = (192, 108)
STATE_COLOURS = {
    "CONNECTED": (60, 160, 60),
    "DISCONNECTED": (50, 50, 200),
    "REINTEGRATING": (0, 150, 230),
}
TARGET_COLOUR = (60, 200, 60)
EMPTY_COLOUR = (130, 130, 130)
TILE_COLOUR = (0, 215, 255)
INK = (235, 235, 235)
BACKGROUND = (30, 30, 30)


@dataclass(frozen=True)
class FrameView:
    """
    What the tagger saw for one frame, kept small enough to hold for a whole run.

    Parameters:
        - frame_id (int): the frame's ID
        - priority (float): the tagger's priority
        - target (bool | None): ground truth from the dataset, None when unknown
        - thumbnail (np.ndarray): BGR image fitted inside ``THUMB``
        - best_tile (tuple[float, float, float, float]): the highest-scoring tile as
          left, top, width, height in fractions of the frame
    """

    frame_id: int
    priority: float
    target: bool | None
    thumbnail: np.ndarray
    best_tile: tuple[float, float, float, float]


@dataclass
class Tracker:
    """
    Where each frame is, as the client's events describe it.

    Lanes are "live", "bank", "inflight" and "sent". Events that name no
    frame only update the state.

    Parameters:
        - lanes (dict[int, str]): frame ID to its current lane
        - sent (list[int]): acknowledged frame IDs in acknowledgement order
        - state (str): the scheduler's state at the latest event
    """

    lanes: dict[int, str] = field(default_factory=dict)
    sent: list[int] = field(default_factory=list)
    state: str = "STARTING"

    def update(self, event: str, detail: str, scheduler: FrameScheduler) -> None:
        """
        Apply one client report event.

        Parameters:
            - event (str): the event name, e.g. "ARRIVE"
            - detail (str): the event's text as the client wrote it
            - scheduler (FrameScheduler): read for its state only

        Return: void
        """
        self.state = scheduler.state.value
        if event == "ARRIVE" and (match := ARRIVE.search(detail)):
            self.lanes[int(match.group(1))] = match.group(2)
        elif event == "MIGRATE" and (match := MIGRATE.search(detail)):
            self.lanes[int(match.group(1))] = "bank"
        elif event in ("SUBMIT", "RESEND") and (match := FRAME_ID.search(detail)):
            self.lanes[int(match.group(1))] = "inflight"
        elif event == "ACK" and (match := FRAME_ID.search(detail)):
            frame_id = int(match.group(1))
            self.lanes[frame_id] = "sent"
            self.sent.append(frame_id)

    def in_lane(self, lane: str) -> list[int]:
        """
        Frame IDs currently in one lane, in ascending order.

        Parameters:
            - lane (str): "live", "bank", "inflight" or "sent"

        Return: frame IDs
        """
        return sorted(frame_id for frame_id, here in self.lanes.items() if here == lane)


def make_view(
    frame_id: int,
    priority: float,
    target: bool | None,
    image: np.ndarray,
    best_tile: tuple[int, int, int],
) -> FrameView:
    """
    Shrink a decoded frame and its best tile into a ``FrameView``.

    Parameters:
        - frame_id (int): the frame's ID
        - priority (float): the tagger's priority
        - target (bool | None): ground truth, None when unknown
        - image (np.ndarray): the decoded BGR frame
        - best_tile (tuple[int, int, int]): left, top and side of the best tile, pixels

    Return: the view
    """
    height, width = image.shape[:2]
    left, top, size = best_tile
    tile = (left / width, top / height, size / width, size / height)
    return FrameView(frame_id, priority, target, fit(image, THUMB), tile)


def fit(image: np.ndarray, box: tuple[int, int]) -> np.ndarray:
    """
    Resize an image to fit inside ``box`` keeping its aspect ratio.

    Parameters:
        - image (np.ndarray): BGR image
        - box (tuple[int, int]): width and height to fit inside

    Return: the resized image, at most ``box`` in each dimension
    """
    height, width = image.shape[:2]
    scale = min(box[0] / width, box[1] / height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def render(
    tracker: Tracker,
    views: dict[int, FrameView],
    latest: tuple[int, np.ndarray] | None,
    elapsed: float,
) -> np.ndarray:
    """
    Draw the dashboard.

    The latest frame fills the left with its best tile outlined. The right
    column shows the bank in send order, highest priority first, and the
    bottom strip the most recently acknowledged frames, newest on the
    right. Green borders mark frames that hold a target.

    Parameters:
        - tracker (Tracker): where each frame is
        - views (dict[int, FrameView]): what the tagger saw, by frame ID
        - latest (tuple[int, np.ndarray] | None): the newest frame's ID and BGR preview
        - elapsed (float): seconds since the demo started

    Return: BGR image of size ``SIZE``
    """
    canvas = np.full((SIZE[1], SIZE[0], 3), BACKGROUND, dtype=np.uint8)
    bank = sorted(
        (f for f in tracker.in_lane("bank") if f in views), key=lambda f: -views[f].priority
    )

    colour = STATE_COLOURS.get(tracker.state, EMPTY_COLOUR)
    cv2.rectangle(canvas, (0, 0), (SIZE[0], 44), colour, -1)
    header = (
        f"{tracker.state}   live {len(tracker.in_lane('live'))}   bank {len(bank)}   "
        f"in flight {len(tracker.in_lane('inflight'))}   sent {len(tracker.sent)}   "
        f"{elapsed:6.1f}s"
    )
    _text(canvas, header, (14, 30), 0.8, 2)

    if latest is not None and latest[0] in views:
        frame_id, preview = latest
        view = views[frame_id]
        shown = fit(preview, PREVIEW)
        x, y = _paste(canvas, shown, (10, 60))
        h, w = shown.shape[:2]
        left, top, tw, th = view.best_tile
        cv2.rectangle(
            canvas,
            (x + round(left * w), y + round(top * h)),
            (x + round((left + tw) * w), y + round((top + th) * h)),
            TILE_COLOUR,
            3,
        )
        lane = tracker.lanes.get(frame_id, "?")
        label = "TARGET" if view.target else "empty" if view.target is False else "unlabelled"
        _text(
            canvas,
            f"latest: frame {frame_id}   priority {view.priority:.3f}   {label}   -> {lane}",
            (14, 520),
            0.65,
            2,
        )

    _text(canvas, "bank, next to send", (870, 74), 0.6, 1)
    for i, frame_id in enumerate(bank[:8]):
        _thumbnail(canvas, views[frame_id], (870 + (i % 2) * 204, 86 + (i // 2) * 136))
    if len(bank) > 8:
        _text(canvas, f"+{len(bank) - 8} more", (870, 640), 0.55, 1)

    _text(canvas, "sent, newest on the right", (14, 560), 0.6, 1)
    for i, frame_id in enumerate(tracker.sent[-4:]):
        if frame_id in views:
            _thumbnail(canvas, views[frame_id], (14 + i * 204, 572))
    return canvas


def _thumbnail(canvas: np.ndarray, view: FrameView, at: tuple[int, int]) -> None:
    """
    Paste a thumbnail with a border coloured by ground truth and its priority beneath.

    Parameters:
        - canvas (np.ndarray): MUTATED, the dashboard
        - view (FrameView): the frame to draw
        - at (tuple[int, int]): top-left corner, pixels

    Return: void
    """
    x, y = _paste(canvas, view.thumbnail, at)
    h, w = view.thumbnail.shape[:2]
    colour = TARGET_COLOUR if view.target else EMPTY_COLOUR
    cv2.rectangle(canvas, (x - 2, y - 2), (x + w + 1, y + h + 1), colour, 3)
    _text(canvas, f"#{view.frame_id}  {view.priority:.3f}", (x, y + h + 18), 0.5, 1)


def _paste(canvas: np.ndarray, image: np.ndarray, at: tuple[int, int]) -> tuple[int, int]:
    """
    Copy an image onto the canvas, clipped to the canvas edges.

    Parameters:
        - canvas (np.ndarray): MUTATED, the dashboard
        - image (np.ndarray): BGR image to place
        - at (tuple[int, int]): top-left corner, pixels

    Return: the top-left corner used
    """
    x, y = at
    h = min(image.shape[0], canvas.shape[0] - y)
    w = min(image.shape[1], canvas.shape[1] - x)
    canvas[y : y + h, x : x + w] = image[:h, :w]
    return x, y


def _text(canvas: np.ndarray, text: str, at: tuple[int, int], scale: float, weight: int) -> None:
    """
    Write text on the canvas.

    Parameters:
        - canvas (np.ndarray): MUTATED, the dashboard
        - text (str): what to write
        - at (tuple[int, int]): baseline start, pixels
        - scale (float): font scale
        - weight (int): stroke thickness

    Return: void
    """
    cv2.putText(canvas, text, at, cv2.FONT_HERSHEY_SIMPLEX, scale, INK, weight, cv2.LINE_AA)

"""
Play a VisDrone-MOT sequence with its ground-truth boxes drawn, or write it
to an mp4.

Usage:
    uv run python scripts/view_trace.py [--split val] [--fps 30] [--width 1280]  # all, looping
    uv run python scripts/view_trace.py --list                     # list sequences and exit
    uv run python scripts/view_trace.py --sequence NAME            # loop one sequence
    uv run python scripts/view_trace.py --sequence NAME --out clip.mp4

Window keys: space pauses and resumes, "." and "," step forward and back
while paused, "n" and "p" jump to the next and previous sequence, "q" or
Esc quits. After the last frame of the last sequence, playback continues
from the first.

Frames and boxes come from ``tagurit.sim.dataloader``, so what you see is
exactly what the loader parsed. Boxes are COCO-style xywh in absolute
pixels; see ``tagurit.sim.trace.Box``.

The window shows frames 1:1 at ``--width`` pixels wide, shrunk by us with
area interpolation before the boxes are drawn. Letting Qt shrink a
full-size frame instead resamples the one-pixel outlines, and which pixels
survive changes every frame, which reads as flashing boxes.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if sys.platform.startswith("linux"):
    # The opencv-python wheel bundles a Qt without a Wayland plugin. Left alone it
    # warns loudly and falls back to X11 anyway; asking for X11 up front is silent.
    # setdefault keeps an explicit QT_QPA_PLATFORM from the user in charge.
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

if sys.platform.startswith("linux") and not os.path.isdir(os.environ.get("QT_QPA_FONTDIR", "")):
    # cv2 points its bundled Qt at a fonts directory the wheel does not ship, and
    # does so at import, so this has to come after it. Without real fonts Qt warns
    # on every hover. The system directory is where fontconfig would have looked.
    if os.path.isdir("/usr/share/fonts"):
        os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts"

from tagurit.sim import Trace, TraceFrame, dataloader

# BGR, one per VisDrone label. Ignored regions are gray so they read as "don't care".
COLORS: dict[str, tuple[int, int, int]] = {
    "ignored": (128, 128, 128),
    "pedestrian": (0, 200, 0),
    "people": (0, 255, 128),
    "bicycle": (255, 128, 0),
    "car": (0, 128, 255),
    "van": (0, 200, 255),
    "truck": (0, 64, 255),
    "tricycle": (255, 0, 128),
    "awning-tricycle": (255, 0, 200),
    "bus": (0, 0, 255),
    "motor": (255, 255, 0),
    "others": (200, 200, 200),
}
WINDOW = "tagurit trace"
FONT = cv2.FONT_HERSHEY_SIMPLEX


def render(frame: TraceFrame, total: int, max_width: int | None = None) -> np.ndarray:
    """
    Decode one frame and draw its boxes and a status overlay onto it.

    With ``max_width``, a wider frame is first shrunk to that width with
    area interpolation and the box coordinates are scaled to match, so the
    outlines are drawn crisp at the size they will be displayed. Each box
    is a one-pixel rectangle in its label's color with "label track_id"
    above it. The overlay in the top-left names the sequence, the frame
    index out of ``total``, the nominal timestamp, and the box count.

    Parameters:
        - frame (TraceFrame): the frame to draw
        - total (int): number of frames in the sequence, for the overlay
        - max_width (int | None): shrink to this width if wider; None keeps native size

    Return: BGR image array ready for imshow or VideoWriter
    """
    image = cv2.imdecode(np.frombuffer(frame.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        sys.exit(f"could not decode {frame.path}")
    scale = 1.0
    height, width = image.shape[:2]
    if max_width is not None and width > max_width:
        scale = max_width / width
        image = cv2.resize(image, (max_width, round(height * scale)), interpolation=cv2.INTER_AREA)
    for box in frame.boxes:
        color = COLORS.get(box.label, (255, 255, 255))
        left, top = round(box.left * scale), round(box.top * scale)
        right, bottom = round((box.left + box.width) * scale), round((box.top + box.height) * scale)
        cv2.rectangle(image, (left, top), (right, bottom), color, 1)
        label_at = (left, max(top - 3, 10))
        text = f"{box.label} {box.track_id}"
        cv2.putText(image, text, label_at, FONT, 0.4, color, 1, cv2.LINE_AA)
    status = (
        f"{frame.sequence}  {frame.index}/{total}  "
        f"t={frame.timestamp:.2f}s  boxes={len(frame.boxes)}"
    )
    # One text pass over a filled bar. Two passes at different thicknesses drift
    # apart because OpenCV's character advance depends on the stroke thickness.
    (text_w, text_h), baseline = cv2.getTextSize(status, FONT, 0.7, 1)
    cv2.rectangle(image, (4, 4), (16 + text_w, 12 + text_h + baseline), (0, 0, 0), cv2.FILLED)
    cv2.putText(image, status, (10, 8 + text_h), FONT, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def write(trace: Trace, out: Path, fps: float) -> None:
    """
    Render every frame of ``trace`` into an mp4 at ``fps``.

    The output size is the first frame's size. Any frame that differs is
    resized to it so the writer never drops frames silently.

    Parameters:
        - trace (Trace): the sequence to render
        - out (Path): mp4 file to create or overwrite
        - fps (float): playback rate written into the file

    Return: void
    """
    total = len(trace)
    first = render(trace.frames[0], total)
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        sys.exit(f"could not open {out} for writing")
    for frame in trace:
        image = render(frame, total)
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height))
        writer.write(image)
    writer.release()
    print(f"wrote {total} frames to {out}")


def window_closed() -> bool:
    """
    Whether the viewer window has gone away.

    Once the user closes the window, Qt has destroyed it and asking for its
    visibility RAISES instead of answering, so the exception is the answer.

    Return: True when the window is hidden or no longer exists
    """
    try:
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def show(root: Path, names: list[str], fps: float, width: int) -> None:
    """
    Play the sequences in ``names`` in a window, in order, looping forever.

    A sequence is loaded the first time it becomes current and kept, so the
    second pass through the playlist has no hitch at boundaries. Frames are
    rendered on demand at ``width`` pixels wide and shown 1:1 in an
    auto-sized window, so nothing is resampled on the way to the screen.
    The window is created without Qt's toolbar and status bar, which are
    what render text on hover. The loop ends on "q", Esc, or the window
    closing.

    Parameters:
        - root (Path): a split folder, see ``dataloader.default_root``
        - names (list[str]): the playlist, at least one sequence name
        - fps (float): playback rate; the wait between frames is 1000/fps ms
        - width (int): display width in pixels; wider frames are shrunk to it

    Return: void
    """
    delay_ms = max(1, round(1000 / fps))
    loaded: dict[str, Trace] = {}

    def trace_at(position: int) -> Trace:
        name = names[position % len(names)]
        if name not in loaded:
            loaded[name] = dataloader.load(root, name, fps=fps)
        return loaded[name]

    position = 0
    index = 0
    paused = False
    trace = trace_at(position)
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    while True:
        cv2.imshow(WINDOW, render(trace.frames[index], len(trace), max_width=width))
        key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
        if key in (ord("q"), 27) or window_closed():
            break
        if key == ord(" "):
            paused = not paused
        elif key == ord("."):
            index = min(index + 1, len(trace) - 1)
            paused = True
        elif key == ord(","):
            index = max(index - 1, 0)
            paused = True
        elif key in (ord("n"), ord("p")):
            position += 1 if key == ord("n") else -1
            trace = trace_at(position)
            index = 0
        elif not paused:
            index += 1
            if index == len(trace):
                position += 1
                trace = trace_at(position)
                index = 0
    cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> None:
    """
    Parse arguments, then list sequences, play a looping playlist, or write
    one sequence to mp4.

    Parameters:
        - argv (list[str] | None): arguments without the program name; None means sys.argv

    Return: void
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", default="val", help="val, train, or test-dev (default: val)")
    parser.add_argument("--list", action="store_true", help="list the split's sequences and exit")
    parser.add_argument("--sequence", help="play only this sequence (default: all, looping)")
    parser.add_argument("--fps", type=float, default=30.0, help="playback rate (default: 30)")
    parser.add_argument(
        "--out", type=Path, help="with --sequence: write an mp4 instead of a window"
    )
    parser.add_argument(
        "--width", type=int, default=1280, help="window width in px (default: 1280)"
    )
    args = parser.parse_args(argv)

    root = dataloader.default_root(args.split)
    try:
        names = dataloader.sequences(root)
        if args.list:
            for name in names:
                print(f"{name}  {len(dataloader.load(root, name))} frames")
            return
        if args.sequence is not None:
            if args.sequence not in names:
                sys.exit(
                    f"unknown sequence {args.sequence!r}; --list shows the {len(names)} available"
                )
            names = [args.sequence]
        if args.out is None:
            show(root, names, args.fps, args.width)
            return
        if len(names) != 1:
            sys.exit("--out needs --sequence to pick which one to write")
        write(dataloader.load(root, names[0], fps=args.fps), args.out, args.fps)
    except FileNotFoundError as exc:
        hint = (
            f"fetch the split first: uv run python scripts/fetch_visdrone.py --split {args.split}"
        )
        sys.exit(f"{exc}\n{hint}")


if __name__ == "__main__":
    main()

"""
Replay a dataset through tagging and the real client: the end-to-end demo.

Usage:
    uv run python3 -m tagurit.sim.run_pipeline data/models/sparse-seadronessee
    uv run python3 -m tagurit.sim.run_pipeline BUNDLE --empty-stride 5 --interval 1.0

The bundle's ``fit.json`` names the dataset and the frames it was not fitted
on; those are replayed in order, one every ``--interval`` seconds, as if a
camera captured them. ``--dataset``, ``--sequences`` and ``--all-frames``
replay something else. ``--empty-stride N`` keeps only every Nth frame
without a target, raising the share of targets so they come up often enough
to watch. ``tagurit.orchestrator`` tags each frame and runs the client, which
sends over Gabriel to the receiver at ``client.config.GABRIEL_ENDPOINT``.

The console shows the client's events and queue sizes as ``sim.run_client``
does, with each frame's priority and, from the dataset's labels, whether it
really holds a target. A window shows the same as pictures
(``sim.pipeline_view``): the latest frame with the tile that set its
priority, the bank in send order, and the frames sent most recently.
``--video PATH`` also writes that view to an MP4, and ``--no-view`` turns the
window off for headless runs. Press q or Esc in the window to stop. Cut the
network to watch tagged frames bank up; restore it to watch the bank drain
highest priority first. Needs the ``model`` extra and a running receiver.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tagurit import orchestrator
from tagurit.client.config import GABRIEL_ENDPOINT
from tagurit.client.frame_scheduler import FrameScheduler
from tagurit.protocol import ImageFrame
from tagurit.sim import dataloader
from tagurit.sim.config import FRAME_INTERVAL_SECONDS
from tagurit.sim.pipeline_view import (
    PREVIEW,
    SIZE,
    FrameView,
    Tracker,
    fit,
    make_view,
    render,
)
from tagurit.sim.run_client import queue_display
from tagurit.sim.trace import TraceFrame
from tagurit.tagging import Tagger, load_bundle

FIT_RECORD = "fit.json"
FRAME_ID = re.compile(r"\bFrame (\d+)\b")
WINDOW = "tagurit"
VIEW_FPS = 10.0


class Recorder:
    """
    Tag through a ``Tagger`` and keep what the console and the window show about each frame.

    Runs in the orchestrator's worker thread. Each frame's entries are
    written there before the frame reaches the client, and read on the
    event loop afterwards.

    Parameters:
        - tagger (Tagger): the real tagger
        - targets (dict[int, bool | None]): frame ID to ground truth from the dataset
    """

    def __init__(self, tagger: Tagger, targets: dict[int, bool | None]) -> None:
        self._tagger = tagger
        self._targets = targets
        self.priorities: dict[int, float] = {}
        self.views: dict[int, FrameView] = {}
        self.latest: tuple[int, np.ndarray] | None = None

    def tag(self, frame_id: int, timestamp: float, image_bytes: bytes) -> ImageFrame:
        """
        Tag one image and record its priority, thumbnail and best tile.

        Parameters:
            - frame_id (int): unique image number within the session
            - timestamp (float): capture time in seconds since the Unix epoch
            - image_bytes (bytes): encoded image

        Return: the tagged frame
        """
        result = self._tagger.score(image_bytes)
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        best = result.tiles[int(np.argmax(result.tile_scores))]
        target = self._targets.get(frame_id)
        self.views[frame_id] = make_view(
            frame_id, result.priority, target, image, (best.left, best.top, best.size)
        )
        self.latest = (frame_id, fit(image, PREVIEW))
        self.priorities[frame_id] = result.priority
        return ImageFrame(frame_id, timestamp, image_bytes, result.priority)


def select_frames(
    record: dict[str, Any] | None,
    dataset: str | None,
    sequences: Sequence[str] | None,
    all_frames: bool,
    empty_stride: int,
) -> list[TraceFrame]:
    """
    The frames to replay, in sequence order and time order within each sequence.

    Without overrides these are the frames the bundle was not fitted on.
    A dataset that cannot be determined RAISES ValueError.

    Parameters:
        - record (dict[str, Any] | None): the bundle's fit record, if it has one
        - dataset (str | None): replay this dataset instead of the fitted one
        - sequences (Sequence[str] | None): replay these sequences
        - all_frames (bool): replay whole sequences, not just the held-out part
        - empty_stride (int): keep every Nth frame without a target, all with one

    Return: frames to replay
    """
    dataset = dataset or (record["dataset"] if record else None)
    if dataset is None:
        raise ValueError("the bundle has no fit.json; pass --dataset")
    if sequences:
        names = list(sequences)
    elif record and dataset == record["dataset"]:
        names = list(record["sequences"])
    else:
        names = dataloader.sequences(dataset)
    fraction = 0.0 if all_frames or not record else float(record["fit_fraction"])

    frames: list[TraceFrame] = []
    empties = 0
    for name in names:
        trace = dataloader.load(dataset, name)
        for frame in trace.split(fraction)[1]:
            if frame.has_target():
                frames.append(frame)
            else:
                if empties % empty_stride == 0:
                    frames.append(frame)
                empties += 1
    return frames


async def replay(frames: Sequence[TraceFrame], interval: float) -> AsyncIterator[bytes]:
    """
    Release each frame's bytes, at most one every ``interval`` seconds.

    The interval runs from one release to the next, so time the consumer
    spends tagging counts towards it; a slower consumer slows the replay.

    Parameters:
        - frames (Sequence[TraceFrame]): frames in replay order
        - interval (float): seconds between releases

    Return: asynchronous iterator of encoded images
    """
    for frame in frames:
        released = time.monotonic()
        yield await asyncio.to_thread(frame.read)
        await asyncio.sleep(max(0.0, interval - (time.monotonic() - released)))


def console(
    priorities: dict[int, float],
    targets: dict[int, bool | None],
    tracker: Tracker,
    start: float,
) -> Callable[[str, str, FrameScheduler], None]:
    """
    A client report callback that updates the tracker and prints events, queues and priorities.

    Parameters:
        - priorities (dict[int, float]): frame ID to priority, filled while tagging
        - targets (dict[int, bool | None]): frame ID to whether the frame holds a target
        - tracker (Tracker): MUTATED, follows every event before it is printed
        - start (float): ``time.monotonic()`` at the start of the demo

    Return: the callback
    """

    def report(event: str, detail: str, scheduler: FrameScheduler) -> None:
        tracker.update(event, detail, scheduler)
        match = FRAME_ID.search(detail)
        if match:
            frame_id = int(match.group(1))
            if frame_id in priorities and "priority" not in detail:
                detail += f" | priority {priorities[frame_id]:.3f}"
            if targets.get(frame_id):
                detail += " | TARGET"
        print(
            f"\n{time.monotonic() - start:7.2f}s | {event:<8} | "
            f"{scheduler.state.value:<13} | {detail}"
        )
        print(
            f"          Live {queue_display(scheduler.live_count())}  "
            f"Bank {queue_display(scheduler.stored_count())}  "
            f"In flight {scheduler.inflight_count()}",
            flush=True,
        )

    return report


async def show(
    recorder: Recorder,
    tracker: Tracker,
    start: float,
    window: bool,
    video: cv2.VideoWriter | None,
    pipeline: asyncio.Task[None],
    quit_requested: asyncio.Event,
) -> None:
    """
    Redraw the dashboard at ``VIEW_FPS`` until cancelled.

    Runs on the event loop, which is the main thread, as macOS requires for
    windows. Redraws can fall behind ``VIEW_FPS``, so the video repeats a
    redraw as often as needed to stay in real time. Pressing q or Esc in the
    window cancels ``pipeline``.

    Parameters:
        - recorder (Recorder): what the tagger saw
        - tracker (Tracker): where each frame is
        - start (float): ``time.monotonic()`` at the start of the demo
        - window (bool): show a window
        - video (cv2.VideoWriter | None): also write each redraw here
        - pipeline (asyncio.Task[None]): the running pipeline, to stop on q
        - quit_requested (asyncio.Event): MUTATED, set before cancelling on q

    Return: void
    """
    written = 0
    while True:
        elapsed = time.monotonic() - start
        image = render(tracker, recorder.views, recorder.latest, elapsed)
        if video is not None:
            while written <= elapsed * VIEW_FPS:
                video.write(image)
                written += 1
        if window:
            cv2.imshow(WINDOW, image)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                quit_requested.set()
                pipeline.cancel()
                return
        await asyncio.sleep(1 / VIEW_FPS)


async def run_demo(
    frames: Sequence[TraceFrame],
    interval: float,
    recorder: Recorder,
    targets: dict[int, bool | None],
    window: bool,
    video: cv2.VideoWriter | None,
) -> np.ndarray:
    """
    Run the pipeline with the console, and the dashboard when asked for.

    A dashboard that fails, such as a window with no display to open on,
    stops the pipeline and RAISES its error at once rather than letting the
    run continue unseen.

    Parameters:
        - frames (Sequence[TraceFrame]): frames to replay
        - interval (float): seconds between releases
        - recorder (Recorder): the tagger wrapper that feeds the console and dashboard
        - targets (dict[int, bool | None]): frame ID to ground truth
        - window (bool): show a window
        - video (cv2.VideoWriter | None): write the dashboard here as well

    Return: the final dashboard image
    """
    start = time.monotonic()
    tracker = Tracker()
    quit_requested = asyncio.Event()
    report = console(recorder.priorities, targets, tracker, start)
    pipeline = asyncio.create_task(orchestrator.run(replay(frames, interval), recorder, report))
    drawer = None
    if window or video is not None:
        drawer = asyncio.create_task(
            show(recorder, tracker, start, window, video, pipeline, quit_requested)
        )
    try:
        if drawer is not None:
            await asyncio.wait({pipeline, drawer}, return_when=asyncio.FIRST_COMPLETED)
            failure = drawer.exception() if drawer.done() and not drawer.cancelled() else None
            if failure is not None:
                raise failure  # e.g. no display; the finally block stops the pipeline
        await pipeline
    except asyncio.CancelledError:
        # Ctrl+C cancels this task; only a q or Esc in the window is handled here.
        if not quit_requested.is_set():
            raise
        print("\nStopped from the window", flush=True)
    finally:
        tasks = [task for task in (pipeline, drawer) if task is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return render(tracker, recorder.views, recorder.latest, time.monotonic() - start)


def main(argv: list[str] | None = None) -> None:
    """
    Parse arguments, load the tagger, and replay frames through the pipeline.

    Parameters:
        - argv (list[str] | None): arguments without the program name; None means sys.argv

    Return: void
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("bundle", type=Path, help="tagger bundle directory")
    parser.add_argument("--dataset", choices=sorted(dataloader.DATASETS))
    parser.add_argument("--sequences", nargs="+")
    parser.add_argument("--all-frames", action="store_true", help="include the fitted frames")
    parser.add_argument("--empty-stride", type=int, default=1, help="keep every Nth empty frame")
    parser.add_argument("--interval", type=float, default=FRAME_INTERVAL_SECONDS)
    parser.add_argument("--limit", type=int, help="replay at most this many frames")
    parser.add_argument("--device", help="torch device; default cuda, then mps, then cpu")
    parser.add_argument(
        "--view", action=argparse.BooleanOptionalAction, default=True, help="show the window"
    )
    parser.add_argument("--video", type=Path, help="also write the dashboard to this MP4")
    args = parser.parse_args(argv)
    if args.empty_stride <= 0 or args.interval < 0:
        parser.error("--empty-stride must be positive and --interval non-negative")

    record_path = args.bundle / FIT_RECORD
    record = json.loads(record_path.read_text()) if record_path.is_file() else None
    frames = select_frames(
        record, args.dataset, args.sequences, args.all_frames, args.empty_stride
    )[: args.limit]
    if not frames:
        parser.error("no frames to replay")
    targets = {i: frame.has_target() for i, frame in enumerate(frames, start=1)}
    print(
        f"Replaying {len(frames)} frames, {sum(bool(t) for t in targets.values())} with a "
        f"target, one every {args.interval:g}s",
        flush=True,
    )

    started = time.perf_counter()
    tagger = load_bundle(args.bundle, device=args.device)
    print(f"Loaded tagger from {args.bundle} in {time.perf_counter() - started:.1f}s", flush=True)
    print(
        f"Sending to {GABRIEL_ENDPOINT}; start the receiver first: "
        "uv run python3 -m tagurit.cloudlet.image_receiver",
        flush=True,
    )
    print("Press Ctrl+C to stop | Queued frames are not drained on exit", flush=True)

    if args.view and sys.platform.startswith("linux"):
        # The opencv-python wheel's Qt has no Wayland plugin; ask for X11 quietly.
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    video = None
    if args.video is not None:
        args.video.parent.mkdir(parents=True, exist_ok=True)
        video = cv2.VideoWriter(str(args.video), cv2.VideoWriter_fourcc(*"mp4v"), VIEW_FPS, SIZE)
        if not video.isOpened():
            parser.error(f"could not open {args.video} for writing")

    recorder = Recorder(tagger, targets)
    try:
        final = asyncio.run(run_demo(frames, args.interval, recorder, targets, args.view, video))
    except KeyboardInterrupt:
        print("\nPipeline stopped")
        return
    finally:
        if video is not None:
            video.release()
            print(f"Wrote {args.video}", flush=True)
    if args.view:
        cv2.imshow(WINDOW, final)
        print("Press any key in the window to close it", flush=True)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

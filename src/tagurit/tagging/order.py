"""
Decide which banked frame to send next from capture times and priorities alone.

Sending the bank strictly by score spends the first sends on repeats: a
target that stayed in view fills the top of the bank with near-identical
frames while other targets wait. This order adds time without trading it
against the score. A frame at or above a detection threshold is a
detection. Detections whose capture times are at most ``gap`` seconds
apart form one encounter, one stretch of the scout looking at something.
Detections go first, in passes: every encounter gets one frame, its best,
before any gets a second, and within a pass the newest encounter goes
first. So a recent sighting is sent before an old one, and an old sighting
seen only once is still sent before any repeat and before every frame
below the threshold. Those follow by priority, as the bank orders them
today.

Encounters are derived from the frames each time, never stored on them, so
rescoring the bank for a new query regroups them by calling this again.
Nothing here reads a clock: only the order and spacing of capture times
matter, which must all come from one clock. Standard library only, so the
scheduler can use it without loading a model.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

ENCOUNTER_GAP = 10.0  # seconds between detections that still count as one encounter


def detection_threshold(normal_priorities: Sequence[float], percentile: float = 99.5) -> float:
    """
    The priority that the given percentile of target-free frames stays below.

    Calibrated on frames known to hold no target, scored as the scout would
    score them, so about ``100 - percentile`` percent of empty frames end up
    counted as detections. Read by linear interpolation between the sorted
    values. No priorities, or a percentile outside (0, 100], RAISES
    ValueError.

    Parameters:
        - normal_priorities (Sequence[float]): priorities of target-free frames
        - percentile (float): share of them that must fall below the threshold

    Return: the threshold, a priority
    """
    if len(normal_priorities) == 0:
        raise ValueError("a detection threshold needs at least one priority")
    if not 0.0 < percentile <= 100.0:
        raise ValueError(f"percentile must be in (0, 100], got {percentile}")
    values = sorted(normal_priorities)
    position = (len(values) - 1) * percentile / 100.0
    below = math.floor(position)
    above = min(below + 1, len(values) - 1)
    return values[below] + (values[above] - values[below]) * (position - below)


def encounters(
    timestamps: Sequence[float],
    priorities: Sequence[float],
    threshold: float,
    gap: float = ENCOUNTER_GAP,
) -> list[int | None]:
    """
    Group detections into encounters by capture time.

    A frame is a detection when its priority is at least ``threshold``.
    Walking the detections in capture order, one more than ``gap`` seconds
    after the previous detection starts a new encounter. Encounters are
    numbered from 0, oldest first. Frames may come in any order. Sequences
    of different lengths, or a negative gap, RAISE ValueError.

    Parameters:
        - timestamps (Sequence[float]): capture time of each frame, seconds on one clock
        - priorities (Sequence[float]): priority of each frame
        - threshold (float): least priority of a detection
        - gap (float): longest pause inside one encounter, seconds

    Return: per frame, its encounter number, or None below the threshold
    """
    _check(timestamps, priorities, gap)
    numbers: list[int | None] = [None] * len(timestamps)
    number, previous = -1, -math.inf
    for i in sorted(range(len(timestamps)), key=lambda i: (timestamps[i], i)):
        if priorities[i] < threshold:
            continue
        if timestamps[i] - previous > gap:
            number += 1
        previous = timestamps[i]
        numbers[i] = number
    return numbers


def next_frame(
    timestamps: Sequence[float],
    priorities: Sequence[float],
    sent: Sequence[bool],
    threshold: float,
    gap: float = ENCOUNTER_GAP,
) -> int | None:
    """
    Pick the frame to send next, given which frames were already sent.

    While an unsent detection remains: take the encounter with the fewest
    frames sent so far, the newest such encounter on a tie, and send its
    best unsent frame. After that, the unsent frame with the highest
    priority. Equal priorities go to the earlier capture. Sent frames must
    stay in the sequences, since they hold encounters together and count
    towards their passes. Sequences of different lengths, or a negative
    gap, RAISE ValueError.

    Parameters:
        - timestamps (Sequence[float]): capture time of each frame, seconds on one clock
        - priorities (Sequence[float]): priority of each frame
        - sent (Sequence[bool]): whether each frame was already sent
        - threshold (float): least priority of a detection
        - gap (float): longest pause inside one encounter, seconds

    Return: index of the frame to send, or None when every frame was sent
    """
    if len(sent) != len(timestamps):
        raise ValueError(f"{len(sent)} sent flags for {len(timestamps)} frames")
    numbers = encounters(timestamps, priorities, threshold, gap)
    delivered: dict[int, int] = {}
    waiting: dict[int, list[int]] = {}
    for i, number in enumerate(numbers):
        if number is None:
            continue
        if sent[i]:
            delivered[number] = delivered.get(number, 0) + 1
        else:
            waiting.setdefault(number, []).append(i)

    def best(frames: Sequence[int]) -> int:
        return min(frames, key=lambda i: (-priorities[i], timestamps[i], i))

    if waiting:
        encounter = min(waiting, key=lambda number: (delivered.get(number, 0), -number))
        return best(waiting[encounter])
    rest = [i for i, number in enumerate(numbers) if number is None and not sent[i]]
    return best(rest) if rest else None


def drain_order(
    timestamps: Sequence[float],
    priorities: Sequence[float],
    threshold: float,
    gap: float = ENCOUNTER_GAP,
) -> list[int]:
    """
    The order a bank that nothing was sent from, and nothing is added to, drains in.

    The same order as calling ``next_frame`` until it returns None, marking
    each pick as sent, computed in one sort.

    Parameters:
        - timestamps (Sequence[float]): capture time of each frame, seconds on one clock
        - priorities (Sequence[float]): priority of each frame
        - threshold (float): least priority of a detection
        - gap (float): longest pause inside one encounter, seconds

    Return: every frame index, in send order
    """
    numbers = encounters(timestamps, priorities, threshold, gap)

    def quality(i: int) -> tuple[float, float, int]:
        return (-priorities[i], timestamps[i], i)

    members: dict[int, list[int]] = {}
    for i, number in enumerate(numbers):
        if number is not None:
            members.setdefault(number, []).append(i)
    detections = [
        (turn, -number, i)
        for number, frames in members.items()
        for turn, i in enumerate(sorted(frames, key=quality))
    ]
    rest = sorted((i for i, number in enumerate(numbers) if number is None), key=quality)
    return [i for _, _, i in sorted(detections)] + rest


def _check(timestamps: Sequence[float], priorities: Sequence[float], gap: float) -> None:
    """
    Reject mismatched sequences and a negative gap.

    Parameters:
        - timestamps (Sequence[float]): capture time of each frame
        - priorities (Sequence[float]): priority of each frame
        - gap (float): longest pause inside one encounter, seconds

    Return: void
    """
    if len(timestamps) != len(priorities):
        raise ValueError(f"{len(timestamps)} timestamps for {len(priorities)} priorities")
    if not gap >= 0:
        raise ValueError(f"gap must be non-negative, got {gap}")

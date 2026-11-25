"""
Novelty against what the scout has just seen: a time window of recent tile embeddings.

A tile is novel when nothing in the last few seconds looked like it. Its
novelty is (1 - its highest cosine to a tile of an earlier frame in the
window) / 2, so it lies in [0, 1] for unit embeddings. Only frames strictly
earlier in time count: other tiles of the same frame, and frames stamped
with the same time, never do. A target that stays in view stops being novel
once its earlier sightings are in the window, while one that has just come
into view is.

How novel is unusually novel depends on frame rate and terrain, so
``NoveltyScale`` turns novelty into a percentile against a uniform sample of
the novelty of the flight's earlier tiles. This module is numpy only and
NEVER imports torch.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np


class NoveltyWindow:
    """
    Remembers the tile embeddings of the last ``seconds`` and scores new frames against them.

    Frames must arrive in time order. A timestamp earlier than the latest
    one starts a new window, as when a new flight's clock restarts. With
    no earlier frame in the window, novelty is NaN, which callers treat as
    neutral. Not thread-safe; drive one instance from one thread. A window
    that is not positive and finite RAISES ValueError.

    Parameters:
        - seconds (float): how far back a frame still counts as just seen
    """

    def __init__(self, seconds: float) -> None:
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError(f"window must be positive and finite, got {seconds}")
        self.seconds = seconds
        self._frames: deque[tuple[float, np.ndarray]] = deque()

    def reset(self) -> None:
        """
        Forget every remembered frame, as at the start of a new flight.

        Return: void
        """
        self._frames.clear()

    def observe(self, embeddings: np.ndarray, timestamp: float) -> np.ndarray:
        """
        Score one frame's tiles against the window, then remember them.

        A non-finite timestamp, embeddings that are not a 2-D array, or a
        width that differs from the remembered tiles' RAISES ValueError.

        Parameters:
            - embeddings (np.ndarray): (tiles, dim) unit tile embeddings of one frame
            - timestamp (float): the frame's capture time in seconds

        Return: (tiles,) novelty in [0, 1], NaN where no earlier frame is in the window
        """
        if not math.isfinite(timestamp):
            raise ValueError(f"timestamp must be finite, got {timestamp}")
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be (tiles, dim), got shape {embeddings.shape}")
        if self._frames and embeddings.shape[1] != self._frames[-1][1].shape[1]:
            raise ValueError(
                f"embeddings have width {embeddings.shape[1]}, "
                f"the window holds {self._frames[-1][1].shape[1]}"
            )
        if self._frames and timestamp < self._frames[-1][0]:
            self.reset()
        while self._frames and self._frames[0][0] < timestamp - self.seconds:
            self._frames.popleft()
        earlier = [tiles for seen, tiles in self._frames if seen < timestamp]
        if earlier and len(embeddings):
            nearest = (embeddings @ np.concatenate(earlier).T).max(axis=1)
            novelty = (1.0 - nearest) / 2.0
        else:
            novelty = np.full(len(embeddings), np.nan, dtype=np.float32)
        self._frames.append((timestamp, embeddings))
        return novelty


class NoveltyScale:
    """
    Novelty as a percentile among the novelty of earlier tiles of the flight.

    Keeps a uniform random sample (reservoir sampling, seeded, so runs
    repeat) of at most ``size`` earlier novelty values. A value's
    percentile is the share of the sample below it, counting ties as half,
    so it lies in [0, 1]. With fewer than ``minimum`` values seen, every
    percentile is 0.5: neutral. A size that is not positive, or a minimum
    outside [1, size], RAISES ValueError.

    Parameters:
        - size (int): most novelty values kept
        - minimum (int): values needed before percentiles mean anything
        - seed (int): seed of the sampling
    """

    def __init__(self, size: int, minimum: int, seed: int = 0) -> None:
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        if not 1 <= minimum <= size:
            raise ValueError(f"minimum must be in [1, {size}], got {minimum}")
        self.size = size
        self.minimum = minimum
        self._rng = np.random.default_rng(seed)
        self._sample: list[float] = []
        self._seen = 0

    def reset(self) -> None:
        """
        Forget every sampled value, as at the start of a new flight.

        Return: void
        """
        self._sample.clear()
        self._seen = 0

    def percentile(self, novelty: np.ndarray) -> np.ndarray:
        """
        Each value's percentile in the sample; 0.5 for NaN, or while the sample is too small.

        Parameters:
            - novelty (np.ndarray): (tiles,) novelty, NaN where nothing earlier was seen

        Return: (tiles,) percentiles in [0, 1], float64
        """
        novelty = np.asarray(novelty, dtype=np.float64)
        out = np.full(novelty.shape, 0.5)
        seen = np.isfinite(novelty)
        if len(self._sample) < self.minimum or not seen.any():
            return out
        sample = np.sort(np.asarray(self._sample))
        below = np.searchsorted(sample, novelty[seen], side="left")
        ties = np.searchsorted(sample, novelty[seen], side="right") - below
        out[seen] = (below + ties / 2) / len(sample)
        return out

    def add(self, novelty: np.ndarray) -> None:
        """
        Offer novelty values to the sample; NaN values are skipped, as they say nothing.

        Parameters:
            - novelty (np.ndarray): (tiles,) novelty of one frame

        Return: void
        """
        for value in np.asarray(novelty, dtype=np.float64):
            if not math.isfinite(value):
                continue
            self._seen += 1
            if len(self._sample) < self.size:
                self._sample.append(float(value))
            else:
                slot = int(self._rng.integers(self._seen))
                if slot < self.size:
                    self._sample[slot] = float(value)

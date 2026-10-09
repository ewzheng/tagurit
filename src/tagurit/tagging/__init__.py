"""
Turns raw tile scores into frame priorities for the scheduler.

``tagger`` scores a frame by its most unusual tile, ``priority`` maps raw
scores into [0, 1) without saturating, and ``bundle`` saves and loads a
fitted tagger as one directory. ``load_bundle`` and ``Tagger`` are the
entry points; torch loads only when a bundle does.
"""

from tagurit.tagging.bundle import load_bundle
from tagurit.tagging.tagger import FrameScore, Tagger

__all__ = ["FrameScore", "Tagger", "load_bundle"]

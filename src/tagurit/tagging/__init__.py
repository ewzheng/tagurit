"""
Turns raw tile scores into frame priorities for the scheduler.

``tagger`` scores a frame by its most unusual tile, ``priority`` maps raw
scores into [0, 1) without saturating, and ``bundle`` saves and loads a
fitted tagger as one directory.
"""

"""Tests for the demo dashboard: following client events and drawing the view."""

from types import SimpleNamespace

import numpy as np

from tagurit.sim.pipeline_view import SIZE, THUMB, Tracker, make_view, render


def scheduler(state: str) -> SimpleNamespace:
    return SimpleNamespace(state=SimpleNamespace(value=state))


def test_tracker_follows_frames_through_the_client() -> None:
    tracker = Tracker()
    tracker.update("ARRIVE", "Frame 1 -> live | priority 0.4", scheduler("CONNECTED"))
    tracker.update("ARRIVE", "Frame 2 -> bank", scheduler("DISCONNECTED"))
    tracker.update("ARRIVE", "Frame 3 -> live", scheduler("DISCONNECTED"))
    tracker.update("MIGRATE", "Frame 3 live -> bank | priority=0.70", scheduler("DISCONNECTED"))
    tracker.update(
        "SUBMIT", "Frame 2 from bank | h264 | JPEG=10B wire=5B", scheduler("REINTEGRATING")
    )
    assert tracker.in_lane("live") == [1]
    assert tracker.in_lane("bank") == [3]
    assert tracker.in_lane("inflight") == [2]
    tracker.update("ACK", "Frame 2 received and retained by server", scheduler("REINTEGRATING"))
    tracker.update("LINK", "Gabriel registered and ready", scheduler("CONNECTED"))
    assert tracker.sent == [2] and tracker.in_lane("sent") == [2]
    assert tracker.state == "CONNECTED"


def test_views_shrink_frames_and_normalise_the_best_tile() -> None:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    view = make_view(7, 0.6, True, image, (640, 360, 360))
    assert view.thumbnail.shape[1] <= THUMB[0] and view.thumbnail.shape[0] <= THUMB[1]
    assert view.best_tile == (0.5, 0.5, 360 / 1280, 0.5)


def test_render_draws_a_full_dashboard_with_or_without_frames() -> None:
    tracker = Tracker()
    assert render(tracker, {}, None, 0.0).shape == (SIZE[1], SIZE[0], 3)

    image = np.full((720, 1280, 3), 90, dtype=np.uint8)
    views = {i: make_view(i, i / 10, i == 2, image, (0, 0, 256)) for i in range(1, 12)}
    for i in range(1, 12):
        tracker.update("ARRIVE", f"Frame {i} -> bank", scheduler("DISCONNECTED"))
    tracker.update("ACK", "Frame 1 received", scheduler("REINTEGRATING"))
    dashboard = render(tracker, views, (11, image[:430, :840]), 12.3)
    assert dashboard.shape == (SIZE[1], SIZE[0], 3)
    assert dashboard.std() > 0  # something was drawn

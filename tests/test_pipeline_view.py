"""Tests for the demo dashboard: following client events and drawing the view."""

from types import SimpleNamespace

import numpy as np
import pytest

from tagurit.sim.pipeline_view import SENDING_COLOUR, SIZE, THUMB, Tracker, make_view, render


def scheduler(state: str, head: tuple[int, ...] = ()) -> SimpleNamespace:
    return SimpleNamespace(
        state=SimpleNamespace(value=state), stored_order=lambda limit: list(head)[:limit]
    )


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
    assert (tracker.sending, tracker.sending_from) == (2, "bank")
    tracker.update("ACK", "Frame 2 received and retained by server", scheduler("REINTEGRATING"))
    tracker.update("LINK", "Gabriel registered and ready", scheduler("CONNECTED"))
    assert tracker.sent == [2] and tracker.in_lane("sent") == [2]
    assert tracker.sending == 2  # kept after its receipt until the next submission
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


@pytest.mark.parametrize(("lane", "slot"), [("live", (14, 572)), ("bank", (870, 86))])
def test_the_latest_frame_sent_is_ringed_at_the_head_of_its_queue(
    lane: str, slot: tuple[int, int]
) -> None:
    image = np.full((720, 1280, 3), 90, dtype=np.uint8)
    views = {i: make_view(i, i / 10, False, image, (0, 0, 256)) for i in (1, 2)}
    tracker = Tracker()
    for i in (1, 2):
        tracker.update("ARRIVE", f"Frame {i} -> {lane}", scheduler("REINTEGRATING"))

    def ring_in_first_slot() -> bool:
        ys, xs = np.nonzero(np.all(render(tracker, views, None, 0.0) == SENDING_COLOUR, axis=-1))
        return (
            xs.size > 0
            and slot[0] - 8 <= xs.min() < slot[0]
            and xs.max() < slot[0] + THUMB[0] + 8
            and slot[1] - 8 <= ys.min() < slot[1]
            and ys.max() < slot[1] + THUMB[1] + 8
        )

    assert not ring_in_first_slot()
    tracker.update("SUBMIT", f"Frame 1 from {lane} | h264", scheduler("REINTEGRATING"))
    assert ring_in_first_slot()
    # A receipt that returns before the next redraw must not hide the send.
    tracker.update("ACK", "Frame 1 received and retained by server", scheduler("REINTEGRATING"))
    assert ring_in_first_slot()


def test_the_bank_column_follows_the_schedulers_order_not_the_priority() -> None:
    views = {
        i: make_view(i, i / 10, False, np.full((720, 1280, 3), 40 * i, dtype=np.uint8), (0, 0, 256))
        for i in (1, 2, 3)
    }

    def first_slot(head: tuple[int, ...]) -> int:
        tracker = Tracker()
        for i in (1, 2, 3):
            tracker.update("ARRIVE", f"Frame {i} -> bank", scheduler("DISCONNECTED", head))
        centre = render(tracker, views, None, 0.0)[86 + THUMB[1] // 2, 870 + THUMB[0] // 2]
        return int(centre[0]) // 40  # each frame's thumbnail is a flat grey of 40 times its ID

    assert first_slot(()) == 3  # no order from the scheduler: highest priority first
    assert first_slot((1, 3)) == 1
    assert Tracker().bank_head == []

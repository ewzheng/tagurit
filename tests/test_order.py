"""Tests for the bank's send order: detections first, one frame per encounter per pass."""

import random

import pytest

from tagurit.tagging.order import detection_threshold, drain_order, encounters, next_frame

THRESHOLD = 0.5


def drain(timestamps: list[float], priorities: list[float], gap: float = 10.0) -> list[int]:
    """Send everything by repeated ``next_frame`` calls, as a scheduler would."""
    sent = [False] * len(timestamps)
    order = []
    while (pick := next_frame(timestamps, priorities, sent, THRESHOLD, gap)) is not None:
        sent[pick] = True
        order.append(pick)
    return order


def test_threshold_is_the_requested_percentile() -> None:
    normal = [i / 100 for i in range(1, 101)]
    assert detection_threshold(normal, percentile=100.0) == 1.0
    assert detection_threshold(normal, percentile=50.0) == pytest.approx(0.505)
    assert detection_threshold([0.3], percentile=99.0) == 0.3


def test_threshold_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        detection_threshold([])
    with pytest.raises(ValueError):
        detection_threshold([0.1, 0.2], percentile=0.0)


def test_detections_within_the_gap_share_an_encounter() -> None:
    timestamps = [0.0, 4.0, 13.0, 30.0, 31.0]
    priorities = [0.9, 0.8, 0.7, 0.6, 0.2]
    assert encounters(timestamps, priorities, THRESHOLD, gap=10.0) == [0, 0, 0, 1, None]


def test_frames_below_the_threshold_do_not_bridge_a_gap() -> None:
    timestamps = [0.0, 8.0, 16.0]
    priorities = [0.9, 0.1, 0.9]
    assert encounters(timestamps, priorities, THRESHOLD, gap=10.0) == [0, None, 1]


def test_encounters_follow_capture_time_not_input_order() -> None:
    timestamps = [100.0, 0.0, 5.0]
    priorities = [0.9, 0.9, 0.9]
    assert encounters(timestamps, priorities, THRESHOLD, gap=10.0) == [1, 0, 0]


def test_newest_encounter_goes_first_and_repeats_wait_for_every_first_frame() -> None:
    # An old sighting seen once, a recent one seen twice, and an empty frame.
    timestamps = [300.0, 3000.0, 3008.0, 3540.0]
    priorities = [0.9, 0.7, 0.8, 0.3]
    assert drain_order(timestamps, priorities, THRESHOLD) == [2, 0, 1, 3]


def test_a_detection_seen_once_is_sent_before_any_frame_below_the_threshold() -> None:
    timestamps = [0.0, 5000.0, 5001.0]
    priorities = [0.51, 0.49, 0.45]
    assert drain_order(timestamps, priorities, THRESHOLD)[0] == 0


def test_frames_below_the_threshold_keep_score_order() -> None:
    timestamps = [0.0, 1.0, 2.0, 3.0]
    priorities = [0.1, 0.4, 0.4, 0.2]
    assert drain_order(timestamps, priorities, THRESHOLD) == [1, 2, 3, 0]


def test_sent_frames_still_count_towards_their_encounter() -> None:
    timestamps = [0.0, 2.0, 100.0]
    priorities = [0.9, 0.8, 0.6]
    # The older encounter already had a frame sent, so the newer one is owed its first.
    assert next_frame(timestamps, priorities, [True, False, False], THRESHOLD) == 2
    assert next_frame(timestamps, priorities, [True, False, True], THRESHOLD) == 1
    assert next_frame(timestamps, priorities, [True, True, True], THRESHOLD) is None


def test_a_sent_frame_keeps_its_encounter_together() -> None:
    timestamps = [0.0, 8.0, 16.0, 200.0]
    priorities = [0.6, 0.9, 0.7, 0.55]
    # With the middle frame sent, frames 0 and 2 are still one encounter with one frame delivered.
    assert next_frame(timestamps, priorities, [False, True, False, False], THRESHOLD) == 3


def test_drain_order_matches_repeated_next_frame() -> None:
    rng = random.Random(0)
    for _ in range(50):
        count = rng.randint(1, 40)
        timestamps = [round(rng.uniform(0, 120), 1) for _ in range(count)]
        priorities = [rng.choice([0.2, 0.4, 0.6, 0.8, rng.random()]) for _ in range(count)]
        expected = drain(timestamps, priorities)
        assert drain_order(timestamps, priorities, THRESHOLD) == expected
        assert sorted(expected) == list(range(count))


def test_order_does_not_depend_on_how_the_bank_lists_its_frames() -> None:
    rng = random.Random(1)
    timestamps = [float(rng.randint(0, 300)) + i / 1000 for i in range(30)]
    priorities = [rng.random() for _ in range(30)]
    shuffled = list(range(30))
    rng.shuffle(shuffled)
    order = drain_order(timestamps, priorities, THRESHOLD)
    again = drain_order(
        [timestamps[i] for i in shuffled], [priorities[i] for i in shuffled], THRESHOLD
    )
    assert [shuffled[i] for i in again] == order


def test_rejects_mismatched_lengths_and_negative_gaps() -> None:
    with pytest.raises(ValueError):
        encounters([0.0, 1.0], [0.5], THRESHOLD)
    with pytest.raises(ValueError):
        encounters([0.0], [0.5], THRESHOLD, gap=-1.0)
    with pytest.raises(ValueError):
        next_frame([0.0], [0.5], [False, True], THRESHOLD)

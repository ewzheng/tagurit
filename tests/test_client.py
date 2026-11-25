"""
Check priority ordering, arbitration, retention and disconnected migration
"""

from functools import partial

import pytest

from tagurit.client.frame_scheduler import FrameScheduler
from tagurit.client.scheduler_datatypes import SchedulerState
from tagurit.protocol import ImageFrame
from tagurit.tagging.order import next_frame

# ============================================================
# Shared test helpers
# ============================================================


# Create a small frame without reading an image file
def _make_frame(frame_id: int, priority: float = 0.50) -> ImageFrame:
    return ImageFrame(
        frame_id=frame_id, timestamp=0.0, image_bytes=b"test image bytes", priority=priority
    )


# Select and immediately complete a frame for scheduling-only checks
def _pop_id(scheduler: FrameScheduler) -> int:
    frame = scheduler.pop_next_frame()
    assert frame is not None
    return frame.frame_id


# ============================================================
# Stored priority bank tests
# ============================================================


# Check retention and ordering of banked frames
def test_priority_bank() -> None:
    """
    Check disconnected retention, priority ordering and score requirements

    Return:
        void
    """

    # Empty queues must have nothing to send
    def check_empty_bank() -> None:
        scheduler = FrameScheduler(
            window_size=1,
        )

        assert scheduler.pending_count() == 0
        assert scheduler.pop_next_frame() is None

    # A stored frame must remain until communication resumes
    def check_store_and_remove() -> None:
        scheduler = FrameScheduler(
            window_size=1,
        )
        frame = _make_frame(1, 0.50)

        scheduler.set_connected(False)
        scheduler.add_frame(frame)

        assert scheduler.stored_count() == 1
        assert scheduler.pop_next_frame() is None
        assert scheduler.stored_count() == 1

        scheduler.set_connected(True)

        assert scheduler.pop_next_frame() is frame
        assert scheduler.pending_count() == 0
        assert scheduler.state == SchedulerState.CONNECTED

    # Higher scores must leave the bank first
    def check_priority_order() -> None:
        scheduler = FrameScheduler(
            window_size=1,
        )
        scheduler.set_connected(False)

        for frame_id, priority in [(1, 0.20), (2, 1.00), (3, 0.50), (4, 0.00)]:
            scheduler.add_frame(_make_frame(frame_id, priority))

        scheduler.set_connected(True)

        assert [_pop_id(scheduler) for _ in range(4)] == [2, 3, 1, 4]
        assert scheduler.pending_count() == 0

    # Equal scores must use arrival order rather than frame ID
    def check_equal_priorities() -> None:
        scheduler = FrameScheduler(
            window_size=1,
        )
        scheduler.set_connected(False)

        for frame_id in [30, 10, 20]:
            scheduler.add_frame(_make_frame(frame_id, 0.90))

        scheduler.set_connected(True)

        assert [_pop_id(scheduler) for _ in range(3)] == [30, 10, 20]

    # A later outage can add work to a partly drained bank
    def check_add_after_removal() -> None:
        scheduler = FrameScheduler(
            window_size=1,
        )
        scheduler.set_connected(False)
        scheduler.add_frame(_make_frame(1, 0.20))
        scheduler.add_frame(_make_frame(2, 0.50))
        scheduler.set_connected(True)

        assert _pop_id(scheduler) == 2

        scheduler.set_connected(False)
        scheduler.add_frame(_make_frame(3, 0.90))
        scheduler.set_connected(True)

        assert _pop_id(scheduler) == 3
        assert _pop_id(scheduler) == 1

    # Every incoming frame must carry a valid score
    def check_unscored_frame() -> None:
        with pytest.raises(ValueError, match="priority score"):
            ImageFrame(frame_id=1, timestamp=0.0, image_bytes=b"test", priority=None)

    check_empty_bank()
    check_store_and_remove()
    check_priority_order()
    check_equal_priorities()
    check_add_after_removal()
    check_unscored_frame()


# ============================================================
# State and arbitration tests
# ============================================================


# Check live ordering and sharing between live and stored traffic
def test_scheduler_modes() -> None:
    """
    Check state transitions and configurable live versus stored arbitration

    Return:
        void
    """

    # Connected mode must serve live frames in arrival order
    def check_connected_order() -> None:
        scheduler = FrameScheduler(
            window_size=1,
        )
        scheduler.add_frame(_make_frame(1, 0.50))
        scheduler.add_frame(_make_frame(2, 0.90))

        assert scheduler.state == SchedulerState.CONNECTED
        assert _pop_id(scheduler) == 1
        assert _pop_id(scheduler) == 2

    # Reintegration must follow the configured two-to-one ratio
    def check_reintegration_order() -> None:
        scheduler = FrameScheduler(window_size=1, live_weight=2, stored_weight=1)
        scheduler.set_connected(False)
        scheduler.add_frame(_make_frame(1, 0.20))
        scheduler.add_frame(_make_frame(2, 0.90))

        assert scheduler.state == SchedulerState.DISCONNECTED
        assert scheduler.pop_next_frame() is None
        assert scheduler.stored_count() == 2

        scheduler.set_connected(True)

        for frame_id in [3, 4, 5, 6]:
            scheduler.add_frame(_make_frame(frame_id, 0.50))

        for expected_id in [3, 4, 2, 5, 6, 1]:
            assert scheduler.state == SchedulerState.REINTEGRATING
            assert _pop_id(scheduler) == expected_id

        assert scheduler.state == SchedulerState.CONNECTED
        assert scheduler.pending_count() == 0

    # Different weights must change the turn pattern
    def check_custom_ratio() -> None:
        scheduler = FrameScheduler(window_size=1, live_weight=1, stored_weight=2)
        scheduler.set_connected(False)
        scheduler.add_frame(_make_frame(1, 0.90))
        scheduler.add_frame(_make_frame(2, 0.80))
        scheduler.set_connected(True)
        scheduler.add_frame(_make_frame(3, 0.50))
        scheduler.add_frame(_make_frame(4, 0.50))

        assert [_pop_id(scheduler) for _ in range(4)] == [3, 1, 2, 4]
        assert scheduler.state == SchedulerState.CONNECTED

    # Waiting live frames must survive an outage before migration is due
    def check_live_queue_survives_outage() -> None:
        scheduler = FrameScheduler(
            window_size=1,
        )
        frame = _make_frame(1, 0.50)
        scheduler.add_frame(frame)
        scheduler.set_connected(False)

        assert scheduler.pop_next_frame() is None
        assert scheduler.live_count() == 1

        scheduler.set_connected(True)

        assert scheduler.state == SchedulerState.CONNECTED
        assert scheduler.pop_next_frame() is frame

    check_connected_order()
    check_reintegration_order()
    check_custom_ratio()
    check_live_queue_survives_outage()


# ============================================================
# Unresolved transmission tests
# ============================================================


# Check that selection never counts as successful delivery
def test_inflight_retention() -> None:
    """
    Check retention across reservation, invalid acknowledgment and reconnection

    These checks exercise scheduler ownership without network communication

    Return:
        void
    """

    # Reserving a frame must block another reservation until acknowledgment
    def check_reservation_and_acknowledgment() -> None:
        scheduler = FrameScheduler(
            window_size=1,
        )
        first = _make_frame(1, 0.50)
        second = _make_frame(2, 0.50)
        scheduler.add_frame(first)
        scheduler.add_frame(second)

        assert scheduler.reserve_next_frame() is first
        assert scheduler.live_count() == 1
        assert scheduler.inflight_count() == 1
        assert scheduler.pending_count() == 2
        assert scheduler.reserve_next_frame() is None

        scheduler.acknowledge_frame(1)

        assert scheduler.pending_count() == 1
        assert scheduler.reserve_next_frame() is second

        scheduler.acknowledge_frame(2)
        assert scheduler.pending_count() == 0

    # An incorrect acknowledgment must leave the frame unresolved
    def check_wrong_acknowledgment() -> None:
        scheduler = FrameScheduler(
            window_size=1,
        )
        scheduler.add_frame(_make_frame(1, 0.50))
        scheduler.reserve_next_frame()

        with pytest.raises(ValueError, match="in-flight frame"):
            scheduler.acknowledge_frame(999)

        assert scheduler.inflight_count() == 1
        assert scheduler.pending_count() == 1
        assert scheduler.reserve_next_frame() is None

        scheduler.acknowledge_frame(1)
        assert scheduler.pending_count() == 0

    # The final banked frame remains part of reintegration until acknowledged
    def check_outage_with_unresolved_bank_frame() -> None:
        scheduler = FrameScheduler(
            window_size=1,
        )
        frame = _make_frame(1, 0.90)

        scheduler.set_connected(False)
        scheduler.add_frame(frame)
        scheduler.set_connected(True)

        assert scheduler.reserve_next_frame() is frame
        assert scheduler.stored_count() == 0
        assert scheduler.inflight_count() == 1
        assert scheduler.inflight_lane() == "bank"
        assert scheduler.state == SchedulerState.REINTEGRATING

        scheduler.set_connected(False)

        assert scheduler.state == SchedulerState.DISCONNECTED
        assert scheduler.pending_count() == 1
        assert scheduler.reserve_next_frame() is None

        scheduler.set_connected(True)

        assert scheduler.state == SchedulerState.REINTEGRATING
        assert scheduler.inflight_count() == 1
        assert scheduler.reserve_next_frame() is None

        scheduler.acknowledge_frame(1)

        assert scheduler.state == SchedulerState.CONNECTED
        assert scheduler.pending_count() == 0

    check_reservation_and_acknowledgment()
    check_wrong_acknowledgment()
    check_outage_with_unresolved_bank_frame()


# ============================================================
# Frame score tests
# ============================================================


# Require valid scores in every operating state
def test_frame_scores() -> None:
    """
    Reject missing or invalid scores and accept the two score boundaries

    Return:
        void
    """

    # Missing and invalid scores must fail at the handoff
    def check_invalid_scores() -> None:
        for score in [None, True, "0.5", -0.01, 1.01, float("nan"), float("inf")]:
            with pytest.raises(ValueError):
                _make_frame(1, score)

        with pytest.raises(TypeError):
            ImageFrame(frame_id=1, timestamp=0.0, image_bytes=b"test")

    # Both ends of the allowed score range must remain valid
    def check_boundaries() -> None:
        assert _make_frame(1, 0.0).priority == 0.0
        assert _make_frame(2, 1.0).priority == 1.0

    check_invalid_scores()
    check_boundaries()


# ============================================================
# Disconnected migration tests
# ============================================================


# Exercise the timer with a clock controlled by each check
def test_live_migration() -> None:
    """
    Check timing, original arrival order and retention without real sleeps

    Return:
        void
    """

    # Move one oldest live frame after each full interval
    def check_interval_and_oldest_frame() -> None:
        now = 0.0
        scheduler = FrameScheduler(window_size=1, clock=lambda: now)
        first = _make_frame(30, 0.20)
        second = _make_frame(10, 0.90)
        scheduler.add_frame(first)
        scheduler.add_frame(second)
        scheduler.set_connected(False)

        now = 4.99
        assert scheduler.migrate_live_frame() is None
        assert scheduler.live_count() == 2

        now = 5.0
        assert scheduler.migrate_live_frame() is first
        assert scheduler.migrate_live_frame() is None
        assert scheduler.live_count() == 1
        assert scheduler.stored_count() == 1
        assert scheduler.pending_count() == 2

        now = 10.0
        assert scheduler.migrate_live_frame() is second
        assert scheduler.live_count() == 0
        assert scheduler.stored_count() == 2
        assert scheduler.reserve_next_frame() is None

        now = 15.0
        assert scheduler.migrate_live_frame() is None

        scheduler.set_connected(True)
        assert scheduler.pop_next_frame() is second
        assert scheduler.pop_next_frame() is first
        assert scheduler.state == SchedulerState.CONNECTED

    # Use original arrival order when migrated and banked scores tie
    def check_original_arrival_order() -> None:
        now = 0.0
        scheduler = FrameScheduler(window_size=1, clock=lambda: now)
        earlier = _make_frame(30, 0.90)
        later = _make_frame(10, 0.90)
        scheduler.add_frame(earlier)
        scheduler.set_connected(False)
        scheduler.add_frame(later)

        now = 5.0
        assert scheduler.migrate_live_frame() is earlier
        scheduler.set_connected(True)
        assert scheduler.pop_next_frame() is earlier
        assert scheduler.pop_next_frame() is later

    # Retry reports must not postpone the next move
    def check_repeated_down_reports() -> None:
        now = 0.0
        scheduler = FrameScheduler(window_size=1, clock=lambda: now)
        frame = _make_frame(1)
        scheduler.add_frame(frame)
        scheduler.set_connected(False)

        now = 4.0
        scheduler.set_connected(False)
        now = 5.0
        assert scheduler.migrate_live_frame() is frame

    # A new outage must wait its own full interval
    def check_timer_reset() -> None:
        now = 0.0
        scheduler = FrameScheduler(window_size=1, clock=lambda: now)
        frame = _make_frame(1)
        scheduler.add_frame(frame)
        scheduler.set_connected(False)

        now = 3.0
        scheduler.set_connected(True)
        now = 20.0
        assert scheduler.migrate_live_frame() is None
        assert scheduler.live_count() == 1
        scheduler.set_connected(False)

        now = 24.99
        assert scheduler.migrate_live_frame() is None
        now = 25.0
        assert scheduler.migrate_live_frame() is frame

    # A waiting frame can move while the unresolved frame remains retained
    def check_inflight_is_untouched() -> None:
        now = 0.0
        scheduler = FrameScheduler(window_size=1, clock=lambda: now)
        inflight = _make_frame(1)
        waiting = _make_frame(2)
        scheduler.add_frame(inflight)
        scheduler.add_frame(waiting)
        assert scheduler.reserve_next_frame() is inflight
        scheduler.set_connected(False)

        now = 5.0
        assert scheduler.migrate_live_frame() is waiting
        assert scheduler.inflight_count() == 1
        assert scheduler.inflight_lane() == "live"
        assert scheduler.pending_count() == 2

        scheduler.set_connected(True)
        assert scheduler.reserve_next_frame() is None
        scheduler.acknowledge_frame(1)
        assert scheduler.pop_next_frame() is waiting
        assert scheduler.pending_count() == 0

    # Late timer checks must not move a burst of frames
    def check_late_poll() -> None:
        now = 0.0
        scheduler = FrameScheduler(window_size=1, clock=lambda: now)
        first = _make_frame(1)
        second = _make_frame(2)
        scheduler.add_frame(first)
        scheduler.add_frame(second)
        scheduler.set_connected(False)

        now = 16.0
        assert scheduler.migrate_live_frame() is first
        assert scheduler.migrate_live_frame() is None
        now = 21.0
        assert scheduler.migrate_live_frame() is second

    # Reintegration must keep waiting live frames in their existing lane
    def check_no_migration_while_reintegrating() -> None:
        now = 0.0
        scheduler = FrameScheduler(window_size=1, clock=lambda: now)
        live = _make_frame(1)
        scheduler.add_frame(live)
        scheduler.set_connected(False)
        scheduler.add_frame(_make_frame(2))
        scheduler.set_connected(True)

        now = 100.0
        assert scheduler.state == SchedulerState.REINTEGRATING
        assert scheduler.migrate_live_frame() is None
        assert scheduler.pop_next_frame() is live

    # The migration interval must be configurable and positive
    def check_interval_setting() -> None:
        now = 0.0
        scheduler = FrameScheduler(window_size=1, migration_interval=2.0, clock=lambda: now)
        frame = _make_frame(1)
        scheduler.add_frame(frame)
        scheduler.set_connected(False)
        now = 2.0
        assert scheduler.migrate_live_frame() is frame

        for interval in [0, -1, float("inf"), float("nan"), True, None]:
            with pytest.raises(ValueError):
                FrameScheduler(window_size=1, migration_interval=interval)

    check_interval_and_oldest_frame()
    check_original_arrival_order()
    check_repeated_down_reports()
    check_timer_reset()
    check_inflight_is_untouched()
    check_late_poll()
    check_no_migration_while_reintegrating()
    check_interval_setting()


# ============================================================
# Supplied bank order tests
# ============================================================


# Create a frame with its own capture time
def _make_timed_frame(frame_id: int, timestamp: float, priority: float) -> ImageFrame:
    return ImageFrame(
        frame_id=frame_id, timestamp=timestamp, image_bytes=b"test image bytes", priority=priority
    )


# Bank frames during an outage and return the reconnected scheduler
def _banked(frames: list[ImageFrame], **options: object) -> FrameScheduler:
    scheduler = FrameScheduler(**options)
    scheduler.set_connected(False)

    for frame in frames:
        scheduler.add_frame(frame)

    scheduler.set_connected(True)
    return scheduler


# Check banked frames follow a supplied order instead of the score
def test_bank_order() -> None:
    """
    Check encounter ordering, delivered-frame history, previews and bad orders

    Return:
        void
    """
    by_encounter = partial(next_frame, threshold=0.50)

    # An old sighting, a recent one seen twice and an empty frame
    def sightings() -> list[ImageFrame]:
        return [
            _make_timed_frame(1, 300.0, 0.90),
            _make_timed_frame(2, 3000.0, 0.70),
            _make_timed_frame(3, 3008.0, 0.80),
            _make_timed_frame(4, 3540.0, 0.30),
        ]

    # Without an order the bank must keep sending the highest score first
    def check_default_is_score_order() -> None:
        scheduler = _banked(sightings())

        assert scheduler.stored_order(10) == [1, 3, 2, 4]
        assert [_pop_id(scheduler) for _ in range(4)] == [1, 3, 2, 4]

    # The newest encounter goes first and its repeat waits for the old sighting
    def check_encounter_order() -> None:
        scheduler = _banked(sightings(), bank_order=by_encounter)

        assert [_pop_id(scheduler) for _ in range(4)] == [3, 1, 2, 4]
        assert scheduler.state == SchedulerState.CONNECTED

    # A preview must show the same order without removing anything
    def check_preview_changes_nothing() -> None:
        scheduler = _banked(sightings(), bank_order=by_encounter)

        assert scheduler.stored_order(2) == [3, 1]
        assert scheduler.stored_order(10) == [3, 1, 2, 4]
        assert scheduler.stored_count() == 4
        assert [_pop_id(scheduler) for _ in range(4)] == [3, 1, 2, 4]

    # A frame already delivered live must count toward its encounter
    def check_live_delivery_counts() -> None:
        scheduler = FrameScheduler(bank_order=by_encounter)
        scheduler.add_frame(_make_timed_frame(1, 0.0, 0.90))
        assert _pop_id(scheduler) == 1

        scheduler.set_connected(False)
        scheduler.add_frame(_make_timed_frame(2, 5.0, 0.80))
        scheduler.add_frame(_make_timed_frame(3, 100.0, 0.60))
        scheduler.set_connected(True)

        # Frame 2 repeats the delivered encounter so the newer sighting goes first
        assert [_pop_id(scheduler) for _ in range(2)] == [3, 2]

    # Live turns must still alternate with the supplied order while reintegrating
    def check_reintegration_turns() -> None:
        scheduler = _banked(sightings(), live_weight=1, stored_weight=1, bank_order=by_encounter)
        scheduler.add_frame(_make_timed_frame(5, 4000.0, 0.10))
        scheduler.add_frame(_make_timed_frame(6, 4001.0, 0.10))

        assert [_pop_id(scheduler) for _ in range(6)] == [5, 3, 6, 1, 2, 4]

    # A reserved frame must count toward its encounter before its receipt arrives
    def check_reserved_frames_count() -> None:
        scheduler = _banked(sightings(), window_size=2, bank_order=by_encounter)

        first = scheduler.reserve_next_frame()
        second = scheduler.reserve_next_frame()
        assert (first.frame_id, second.frame_id) == (3, 1)
        assert scheduler.reserve_next_frame() is None

    # An order that names no waiting frame must be rejected
    def check_bad_orders() -> None:
        for answer in [None, 4, -1]:
            scheduler = _banked(
                sightings(), bank_order=lambda times, scores, sent, answer=answer: answer
            )

            with pytest.raises(ValueError):
                scheduler.pop_next_frame()

    check_default_is_score_order()
    check_encounter_order()
    check_preview_changes_nothing()
    check_live_delivery_counts()
    check_reintegration_turns()
    check_reserved_frames_count()
    check_bad_orders()


# ============================================================
# Rescoring tests
# ============================================================


# Check waiting frames can be rescored after the query changes
def test_rescoring() -> None:
    """
    Check waiting-frame listing, score replacement, history reset and bad scores

    Return:
        void
    """
    by_encounter = partial(next_frame, threshold=0.50)

    # Waiting frames must be listed from both queues without the in-flight frame
    def check_waiting_frames() -> None:
        scheduler = FrameScheduler()
        scheduler.add_frame(_make_frame(1, 0.20))
        scheduler.add_frame(_make_frame(2, 0.30))
        scheduler.set_connected(False)
        scheduler.add_frame(_make_frame(3, 0.90))
        scheduler.set_connected(True)

        assert scheduler.reserve_next_frame().frame_id == 1
        assert [frame.frame_id for frame in scheduler.waiting_frames()] == [2, 3]

    # New scores must reorder the bank and keep contents and arrival order
    def check_bank_reorders() -> None:
        scheduler = _banked([_make_frame(1, 0.90), _make_frame(2, 0.50), _make_frame(3, 0.10)])

        assert scheduler.reprioritize({1: 0.20, 3: 0.80, 99: 0.70}) == 2
        assert [frame.priority for frame in scheduler.waiting_frames()] == [0.20, 0.50, 0.80]
        assert scheduler.waiting_frames()[0].image_bytes == b"test image bytes"
        assert [_pop_id(scheduler) for _ in range(3)] == [3, 2, 1]

    # Live frames must carry their new score into the bank
    def check_live_frames_are_rescored() -> None:
        now = 0.0
        scheduler = FrameScheduler(clock=lambda: now)
        scheduler.add_frame(_make_frame(1, 0.10))
        scheduler.add_frame(_make_frame(2, 0.20))
        assert scheduler.reprioritize({1: 0.95}) == 1

        scheduler.set_connected(False)
        now = 5.0
        assert scheduler.migrate_live_frame().priority == 0.95
        assert scheduler.live_count() == 1

    # The in-flight frame must keep its score and still be acknowledged
    def check_inflight_is_untouched() -> None:
        scheduler = FrameScheduler()
        scheduler.add_frame(_make_frame(1, 0.10))
        frame = scheduler.reserve_next_frame()

        assert scheduler.reprioritize({1: 0.90}) == 0
        assert frame.priority == 0.10
        scheduler.acknowledge_frame(1)
        assert scheduler.pending_count() == 0

    # Delivered frames scored under the old query must stop counting toward encounters
    def check_history_is_cleared() -> None:
        scheduler = FrameScheduler(bank_order=by_encounter)
        scheduler.add_frame(_make_timed_frame(1, 0.0, 0.90))
        assert _pop_id(scheduler) == 1

        scheduler.set_connected(False)
        scheduler.add_frame(_make_timed_frame(2, 5.0, 0.80))
        scheduler.add_frame(_make_timed_frame(3, 100.0, 0.60))
        scheduler.set_connected(True)

        # Frame 2 repeats the delivered encounter so the newer sighting goes first
        assert scheduler.stored_order(2) == [3, 2]

        # The new query puts frame 3 below the threshold and forgets the old delivery
        scheduler.reprioritize({2: 0.80, 3: 0.40})
        assert scheduler.stored_order(2) == [2, 3]

    # A bad score must be rejected without changing either queue
    def check_bad_score_changes_nothing() -> None:
        scheduler = _banked([_make_frame(1, 0.90), _make_frame(2, 0.50)])

        with pytest.raises(ValueError):
            scheduler.reprioritize({1: 0.20, 2: 1.50})

        assert [frame.priority for frame in scheduler.waiting_frames()] == [0.90, 0.50]

    check_waiting_frames()
    check_bank_reorders()
    check_live_frames_are_rescored()
    check_inflight_is_untouched()
    check_history_is_cleared()
    check_bad_score_changes_nothing()

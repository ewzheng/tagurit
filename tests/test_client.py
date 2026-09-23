"""
Check priority ordering, arbitration and unresolved frame retention
"""

import pytest

from tagurit.client.frame_scheduler import FrameScheduler
from tagurit.client.scheduler_datatypes import SchedulerState
from tagurit.protocol import ImageFrame


# ============================================================
# Shared test helpers
# ============================================================

# Create a small frame without reading an image file
def _make_frame(frame_id: int, priority: float | None) -> ImageFrame:
    return ImageFrame(
        frame_id=frame_id,
        timestamp=0.0,
        image_bytes=b"test image bytes",
        priority=priority
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
        scheduler = FrameScheduler()

        assert scheduler.pending_count() == 0
        assert scheduler.pop_next_frame() is None

    # A stored frame must remain until communication resumes
    def check_store_and_remove() -> None:
        scheduler = FrameScheduler()
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
        scheduler = FrameScheduler()
        scheduler.set_connected(False)

        for frame_id, priority in [(1, 0.20), (2, 1.00), (3, 0.50), (4, 0.00)]:
            scheduler.add_frame(_make_frame(frame_id, priority))

        scheduler.set_connected(True)

        assert [_pop_id(scheduler) for _ in range(4)] == [2, 3, 1, 4]
        assert scheduler.pending_count() == 0

    # Equal scores must use arrival order rather than frame ID
    def check_equal_priorities() -> None:
        scheduler = FrameScheduler()
        scheduler.set_connected(False)

        for frame_id in [30, 10, 20]:
            scheduler.add_frame(_make_frame(frame_id, 0.90))

        scheduler.set_connected(True)

        assert [_pop_id(scheduler) for _ in range(3)] == [30, 10, 20]

    # A later outage can add work to a partly drained bank
    def check_add_after_removal() -> None:
        scheduler = FrameScheduler()
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

    # Unscored frames must not enter the bank
    def check_unscored_frame() -> None:
        scheduler = FrameScheduler()
        scheduler.set_connected(False)

        with pytest.raises(ValueError, match="priority score"):
            scheduler.add_frame(_make_frame(1, None))

        assert scheduler.pending_count() == 0

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
        scheduler = FrameScheduler()
        scheduler.add_frame(_make_frame(1, None))
        scheduler.add_frame(_make_frame(2, 0.90))

        assert scheduler.state == SchedulerState.CONNECTED
        assert _pop_id(scheduler) == 1
        assert _pop_id(scheduler) == 2

    # Reintegration must follow the configured two-to-one ratio
    def check_reintegration_order() -> None:
        scheduler = FrameScheduler(live_weight=2, stored_weight=1)
        scheduler.set_connected(False)
        scheduler.add_frame(_make_frame(1, 0.20))
        scheduler.add_frame(_make_frame(2, 0.90))

        assert scheduler.state == SchedulerState.DISCONNECTED
        assert scheduler.pop_next_frame() is None
        assert scheduler.stored_count() == 2

        scheduler.set_connected(True)

        for frame_id in [3, 4, 5, 6]:
            scheduler.add_frame(_make_frame(frame_id, None))

        for expected_id in [3, 4, 2, 5, 6, 1]:
            assert scheduler.state == SchedulerState.REINTEGRATING
            assert _pop_id(scheduler) == expected_id

        assert scheduler.state == SchedulerState.CONNECTED
        assert scheduler.pending_count() == 0

    # Different weights must change the turn pattern
    def check_custom_ratio() -> None:
        scheduler = FrameScheduler(live_weight=1, stored_weight=2)
        scheduler.set_connected(False)
        scheduler.add_frame(_make_frame(1, 0.90))
        scheduler.add_frame(_make_frame(2, 0.80))
        scheduler.set_connected(True)
        scheduler.add_frame(_make_frame(3, None))
        scheduler.add_frame(_make_frame(4, None))

        assert [_pop_id(scheduler) for _ in range(4)] == [3, 1, 2, 4]
        assert scheduler.state == SchedulerState.CONNECTED

    # Waiting live frames must survive an outage
    def check_live_queue_survives_outage() -> None:
        scheduler = FrameScheduler()
        frame = _make_frame(1, None)
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
        scheduler = FrameScheduler()
        first = _make_frame(1, None)
        second = _make_frame(2, None)
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
        scheduler = FrameScheduler()
        scheduler.add_frame(_make_frame(1, None))
        scheduler.reserve_next_frame()

        with pytest.raises(ValueError, match="in-flight frame"):
            scheduler.acknowledge_frame(999)

        assert scheduler.inflight_count() == 1
        assert scheduler.pending_count() == 1
        assert scheduler.reserve_next_frame() is None

        scheduler.acknowledge_frame(1)
        assert scheduler.pending_count() == 0

    # The final banked frame must remain part of reintegration until acknowledged
    def check_outage_with_unresolved_bank_frame() -> None:
        scheduler = FrameScheduler()
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
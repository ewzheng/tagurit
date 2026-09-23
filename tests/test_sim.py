"""
Check sample image loading and scheduled connectivity without real time or network
"""

import csv
from pathlib import Path

import pytest

from tagurit.client.frame_scheduler import FrameScheduler
from tagurit.client.scheduler_datatypes import SchedulerState
from tagurit.protocol import ImageFrame
from tagurit.sim.connectivity import ConnectivityEvent, ConnectivityMonitor
from tagurit.sim.image_feeder import feed_images


# ============================================================
# Shared test helpers
# ============================================================

# Create a frame for scheduler checks
def _make_frame(frame_id: int, priority: float | None) -> ImageFrame:
    return ImageFrame(
        frame_id=frame_id,
        timestamp=0.0,
        image_bytes=b"test image bytes",
        priority=priority
    )


# Select and complete a frame for scheduling-only checks
def _pop_id(scheduler: FrameScheduler) -> int:
    frame = scheduler.pop_next_frame()
    assert frame is not None
    return frame.frame_id


# ============================================================
# Image feeder tests
# ============================================================

# Build temporary input so tests do not depend on local demo data
def test_image_feeder(tmp_path: Path) -> None:
    """
    Check manifest order, frame IDs, scores and loaded image bytes

    The feeder copies encoded bytes without decoding images
    Temporary byte files are sufficient for this test

    Parameters:
        - tmp_path (Path): Temporary directory supplied by pytest

    Return:
        void
    """

    # Create two files and list them in a different order in the manifest
    image_directory = tmp_path / "images"
    image_directory.mkdir()
    (image_directory / "first.jpg").write_bytes(b"first image contents")
    (image_directory / "second.jpg").write_bytes(b"second image contents")

    manifest = tmp_path / "manifest.csv"

    with manifest.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(["image", "priority"])
        writer.writerow(["images/second.jpg", "0.90"])
        writer.writerow(["images/first.jpg", "0.20"])

    frames = list(feed_images(manifest))

    # Each manifest row must produce one frame
    def check_frame_count() -> None:
        assert len(frames) == 2

    # IDs must follow manifest order
    def check_frame_ids() -> None:
        assert [frame.frame_id for frame in frames] == [1, 2]

    # Scores must follow manifest order
    def check_priorities() -> None:
        assert [frame.priority for frame in frames] == [0.90, 0.20]

    # Paths must resolve relative to the temporary manifest
    def check_image_bytes() -> None:
        assert frames[0].image_bytes == b"second image contents"
        assert frames[1].image_bytes == b"first image contents"

    check_frame_count()
    check_frame_ids()
    check_priorities()
    check_image_bytes()


# ============================================================
# Simulated connectivity tests
# ============================================================

# Advance a fake clock to exercise connection changes
def test_connectivity() -> None:
    """
    Check schedule boundaries and their effects on scheduler state

    The clock is controlled directly so no real waiting or network is needed

    Return:
        void
    """
    schedule = (
        ConnectivityEvent(after_seconds=0.0, available=True),
        ConnectivityEvent(after_seconds=5.0, available=False),
        ConnectivityEvent(after_seconds=10.0, available=True)
    )

    # Boundary events must apply at their exact scheduled times
    def check_schedule_boundaries() -> None:
        now = 100.0
        monitor = ConnectivityMonitor(schedule=schedule, clock=lambda: now)
        cases = [
            (0.0, True),
            (4.9, True),
            (5.0, False),
            (9.9, False),
            (10.0, True),
            (30.0, True)
        ]

        for elapsed, expected in cases:
            now = 100.0 + elapsed
            assert monitor.is_available() is expected

    # Scheduled changes must produce a complete operating-state cycle
    def check_full_state_cycle() -> None:
        now = 0.0
        monitor = ConnectivityMonitor(schedule=schedule, clock=lambda: now)
        scheduler = FrameScheduler(live_weight=2, stored_weight=1)

        scheduler.set_connected(monitor.is_available())
        assert scheduler.state == SchedulerState.CONNECTED

        scheduler.add_frame(_make_frame(1, None))
        assert _pop_id(scheduler) == 1

        # Advance to the outage and retain scored frames
        now = 5.0
        scheduler.set_connected(monitor.is_available())
        assert scheduler.state == SchedulerState.DISCONNECTED

        scheduler.add_frame(_make_frame(2, 0.20))
        scheduler.add_frame(_make_frame(3, 0.90))

        assert scheduler.pop_next_frame() is None
        assert scheduler.stored_count() == 2

        # Restore availability and supply new live frames
        now = 10.0
        scheduler.set_connected(monitor.is_available())

        for frame_id in [4, 5, 6, 7]:
            scheduler.add_frame(_make_frame(frame_id, None))

        # Repeated availability updates must not reset arbitration
        for expected_id in [4, 5, 3, 6, 7, 2]:
            scheduler.set_connected(monitor.is_available())
            assert scheduler.state == SchedulerState.REINTEGRATING
            assert _pop_id(scheduler) == expected_id

        assert scheduler.state == SchedulerState.CONNECTED
        assert scheduler.pending_count() == 0
        assert monitor.is_available() is True

    # Reconnection without banked work must return directly to connected
    def check_reconnect_without_backlog() -> None:
        now = 0.0
        monitor = ConnectivityMonitor(schedule=schedule, clock=lambda: now)
        scheduler = FrameScheduler()

        now = 5.0
        scheduler.set_connected(monitor.is_available())
        assert scheduler.state == SchedulerState.DISCONNECTED

        now = 10.0
        scheduler.set_connected(monitor.is_available())
        assert scheduler.state == SchedulerState.CONNECTED

    # Out of order events must be rejected
    def check_invalid_schedule() -> None:
        invalid_schedule = (
            ConnectivityEvent(after_seconds=0.0, available=True),
            ConnectivityEvent(after_seconds=10.0, available=False),
            ConnectivityEvent(after_seconds=5.0, available=True)
        )

        with pytest.raises(ValueError, match="strictly increase"):
            ConnectivityMonitor(schedule=invalid_schedule)

    check_schedule_boundaries()
    check_full_state_cycle()
    check_reconnect_without_backlog()
    check_invalid_schedule()
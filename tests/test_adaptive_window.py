"""
Check adaptive client-window decisions with deterministic receipt samples.
"""

import pytest

from tagurit.client.adaptive_window import AdaptiveWindowController
from tagurit.client.rolling_metrics import RollingSnapshot


def sample(delay_ms: float, fps: float, receipts: int = 20) -> RollingSnapshot:
    """
    Build a full ten-second measurement sample for controller tests.

    Parameters:
        - delay_ms (float): Mean first-submit-to-receipt delay
        - fps (float): Acknowledged frames per second
        - receipts (int): Validated receipts in the sample

    Return: A rolling snapshot spanning ten seconds
    """
    return RollingSnapshot(10.0, receipts, delay_ms, fps)


def test_probe_reverts_when_delay_rises_then_recovers_after_cooldown():
    """Bad probes back off; recovered throughput can justify a later probe."""
    controller = AdaptiveWindowController(started_at=0.0, measurement_seconds=10.0)
    assert controller.window_size == 4

    probe = controller.evaluate(sample(500, 5.0), 10.0, waiting_frames=10, connected=True)
    assert probe.window_size == 8

    revert = controller.evaluate(sample(1500, 5.5), 20.0, waiting_frames=10, connected=True)
    assert revert.window_size == 4

    shrink = controller.evaluate(sample(1400, 4.0), 30.0, waiting_frames=10, connected=True)
    assert shrink.window_size == 2
    assert controller.evaluate(sample(700, 2.0), 40.0, waiting_frames=10, connected=True) is None

    retry_probe = controller.evaluate(sample(700, 2.0), 60.0, waiting_frames=10, connected=True)
    assert retry_probe.window_size == 4
    assert controller.evaluate(sample(900, 2.2), 70.0, waiting_frames=10, connected=True) is None
    assert controller.window_size == 4


def test_controller_waits_for_receipts_and_demand():
    """Startup, outages and idle clients do not cause a growth probe."""
    controller = AdaptiveWindowController(started_at=0.0, measurement_seconds=10.0)
    assert controller.evaluate(sample(300, 5.0), 9.0, waiting_frames=10, connected=True) is None
    assert (
        controller.evaluate(sample(300, 0.2, receipts=2), 10.0, waiting_frames=10, connected=True)
        is None
    )
    assert controller.evaluate(sample(300, 5.0), 12.0, waiting_frames=10, connected=False) is None
    assert controller.evaluate(sample(300, 5.0), 12.0, waiting_frames=0, connected=True) is None
    assert controller.window_size == 4


def test_connection_failure_shrinks_and_delays_another_probe():
    """A receipt timeout immediately reduces pressure before replay begins."""
    controller = AdaptiveWindowController(started_at=0.0, measurement_seconds=10.0)
    failure = controller.connection_failed(3.0)
    assert failure.window_size == 2
    assert controller.evaluate(sample(300, 3.0), 13.0, waiting_frames=10, connected=True) is None
    assert (
        controller.evaluate(sample(300, 3.0), 33.0, waiting_frames=10, connected=True).window_size
        == 4
    )


@pytest.mark.parametrize("invalid", (0, 3, 16, True))
def test_controller_rejects_unsupported_initial_window(invalid):
    """The first policy probes only tested window levels."""
    with pytest.raises(ValueError, match="Adaptive window"):
        AdaptiveWindowController(0.0, 10.0, invalid)

"""
Configure sample image input and simulated connectivity
"""

from pathlib import Path
from tagurit.sim.connectivity import ConnectivityEvent

# Locate the repository above src/tagurit/sim
REPOSITORY_DIRECTORY = Path(__file__).resolve().parents[3]

# Read sample images through the manifest in data/demo
MANIFEST_PATH = REPOSITORY_DIRECTORY / "data" / "demo" / "manifest.csv"

# Set connection availability for tests using the simulated monitor
CONNECTIVITY_SCHEDULE = (
    ConnectivityEvent(after_seconds=0.0, available=True),
    ConnectivityEvent(after_seconds=5.0, available=False),
    ConnectivityEvent(after_seconds=10.0, available=True)
)

# Set the timing for simulated image arrivals
FRAME_INTERVAL_SECONDS = 1.0
LOOP_INTERVAL_SECONDS = 0.02

# Limit the number of blocks printed for each queue
QUEUE_DISPLAY_LIMIT = 30

# Keep timing settings for demos that stop automatically
DEMO_DURATION_SECONDS = 60.0
SHUTDOWN_GRACE_SECONDS = 10.0
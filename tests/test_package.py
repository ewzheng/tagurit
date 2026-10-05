"""
Check that the migrated package modules import successfully
"""

import importlib
import subprocess
import sys

import pytest

# Include the current modules without the removed placeholder packages
MODULES = [
    "tagurit",
    "tagurit.protocol",
    "tagurit.orchestrator",
    "tagurit.model",
    "tagurit.model.clip",
    "tagurit.model.device",
    "tagurit.model.patchcore",
    "tagurit.model.preprocess",
    "tagurit.model.scorer",
    "tagurit.model.tiling",
    "tagurit.tagging",
    "tagurit.tagging.bundle",
    "tagurit.tagging.priority",
    "tagurit.tagging.tagger",
    "tagurit.client",
    "tagurit.client.config",
    "tagurit.client.scheduler_datatypes",
    "tagurit.client.frame_scheduler",
    "tagurit.client.gabriel_transport",
    "tagurit.client.main",
    "tagurit.cloudlet",
    "tagurit.cloudlet.config",
    "tagurit.cloudlet.receiver_datatypes",
    "tagurit.cloudlet.image_receiver",
    "tagurit.shared",
    "tagurit.shared.image_protocol",
    "tagurit.sim",
    "tagurit.sim.config",
    "tagurit.sim.connectivity",
    "tagurit.sim.image_feeder",
    "tagurit.sim.run_client",
    "tagurit.sim.trace",
    "tagurit.sim.dataloader",
    "tagurit.sim.visdrone",
    "tagurit.sim.seadronessee",
    "tagurit.sim.metrics",
    "tagurit.sim.framelist",
    "tagurit.sim.sparse",
    "tagurit.sim.normal_tiles",
]


# Import each module as a separate pytest case
@pytest.mark.parametrize("name", MODULES)
def test_imports(name: str) -> None:
    """
    Import a module without starting an application

    Parameters:
        - name (str): Full package module name

    Return:
        void
    """
    importlib.import_module(name)


def test_tagging_entry_points_import_without_torch() -> None:
    """
    Import the tagging entry points in a fresh interpreter and check torch stayed out

    Return:
        void
    """
    code = (
        "import sys; from tagurit.tagging import Tagger, load_bundle; print('torch' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"

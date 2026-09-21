"""
Check that the migrated package modules import successfully
"""

import importlib
import pytest


# Include the current modules without the removed placeholder packages
MODULES = [
    "tagurit",
    "tagurit.protocol",
    "tagurit.orchestrator",
    "tagurit.model",
    "tagurit.tagging",
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
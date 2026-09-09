"""Smoke test: every module imports cleanly."""

import importlib

import pytest

MODULES = [
    "tagurit",
    "tagurit.protocol",
    "tagurit.orchestrator",
    "tagurit.model",
    "tagurit.tagging",
    "tagurit.cache",
    "tagurit.link",
    "tagurit.transfer",
    "tagurit.cloudlet",
    "tagurit.sim",
]


@pytest.mark.parametrize("name", MODULES)
def test_imports(name: str) -> None:
    importlib.import_module(name)

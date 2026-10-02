"""
One door to every dataset on disk: which sequences exist, and load one as a Trace.

The sim never learns which dataset a Trace came from. Each parser module
exposes ``sequences(root)`` and ``load(root, name, fps)`` over its own
on-disk layout under ``data/<dataset>``; this module only names the
datasets and dispatches. Adding a dataset is a new parser module and one
entry in ``DATASETS``.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

from tagurit.sim import seadronessee, visdrone
from tagurit.sim.trace import Trace

DATASETS: dict[str, ModuleType] = {
    "visdrone": visdrone,
    "seadronessee": seadronessee,
}

DATA_DIR = Path("data")


def data_root(dataset: str) -> Path:
    """
    Where the fetch script puts a dataset, relative to the current directory.

    Parameters:
        - dataset (str): a key of ``DATASETS``

    Return: Path to ``data/<dataset>``
    """
    _parser(dataset)
    return DATA_DIR / dataset


def sequences(dataset: str, root: Path | None = None) -> list[str]:
    """
    Names of the sequences of ``dataset`` that are on disk.

    Parameters:
        - dataset (str): a key of ``DATASETS``
        - root (Path | None): dataset folder; None means ``data_root(dataset)``

    Return: sorted sequence names, in the parser's naming scheme
    """
    return _parser(dataset).sequences(root if root is not None else data_root(dataset))


def load(dataset: str, name: str, root: Path | None = None, fps: float = 30.0) -> Trace:
    """
    Load one sequence of ``dataset`` as a Trace.

    Parameters:
        - dataset (str): a key of ``DATASETS``
        - name (str): a name from ``sequences(dataset)``
        - root (Path | None): dataset folder; None means ``data_root(dataset)``
        - fps (float): source frame rate the parser turns frame numbers into seconds with

    Return: Trace with frames in ascending order
    """
    return _parser(dataset).load(root if root is not None else data_root(dataset), name, fps=fps)


def _parser(dataset: str) -> ModuleType:
    """
    The parser module for ``dataset``, or ValueError naming the known ones.

    Parameters:
        - dataset (str): a key of ``DATASETS``

    Return: the parser module
    """
    try:
        return DATASETS[dataset]
    except KeyError:
        raise ValueError(f"unknown dataset {dataset!r}; known: {sorted(DATASETS)}") from None

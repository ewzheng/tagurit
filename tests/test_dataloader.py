"""Tests for the dataset dispatcher: it names datasets and delegates, nothing more."""

from pathlib import Path

import pytest

from tagurit.sim import dataloader

SEQ = "uav0000001_00000_v"


@pytest.fixture
def visdrone_root(tmp_path: Path) -> Path:
    split = tmp_path / "VisDrone2019-MOT-val"
    (split / "sequences" / SEQ).mkdir(parents=True)
    (split / "sequences" / SEQ / "0000001.jpg").write_bytes(b"frame-1")
    (split / "annotations").mkdir()
    (split / "annotations" / f"{SEQ}.txt").write_text("1,1,10,20,30,40,1,1,0,0\n")
    return tmp_path


def test_datasets_are_named() -> None:
    assert sorted(dataloader.DATASETS) == ["seadronessee", "visdrone"]


def test_data_root_is_under_data_dir() -> None:
    assert dataloader.data_root("visdrone") == Path("data") / "visdrone"
    assert dataloader.data_root("seadronessee") == Path("data") / "seadronessee"


def test_unknown_dataset_raises() -> None:
    with pytest.raises(ValueError, match="nope.*seadronessee.*visdrone"):
        dataloader.sequences("nope")
    with pytest.raises(ValueError, match="nope"):
        dataloader.data_root("nope")


def test_sequences_and_load_delegate(visdrone_root: Path) -> None:
    assert dataloader.sequences("visdrone", root=visdrone_root) == [f"val/{SEQ}"]
    trace = dataloader.load("visdrone", f"val/{SEQ}", root=visdrone_root, fps=10.0)
    assert len(trace) == 1
    assert trace.fps == 10.0
    assert trace.frames[0].boxes[0].label == "pedestrian"

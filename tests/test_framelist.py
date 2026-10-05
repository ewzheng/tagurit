"""Tests for the frame-labelled sequence format: labels, boxes, timestamps, bad input."""

from pathlib import Path

import pytest

from tagurit.sim import framelist


def write_sequence(root: Path, name: str, frames: str, boxes: str | None = None) -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    (folder / framelist.FRAMES_FILE).write_text(frames)
    if boxes is not None:
        (folder / framelist.BOXES_FILE).write_text(boxes)
    return folder


def test_loads_labels_boxes_and_timestamps(tmp_path: Path) -> None:
    folder = write_sequence(
        tmp_path,
        "seq",
        "file,target,timestamp\na.jpg,0,0.5\nb.jpg,1,1.0\n",
        "file,label,dataset_label,left,top,width,height\nb.jpg,person,swimmer,1,2,3,4\n",
    )
    trace = framelist.load(tmp_path, "seq")
    a, b = trace.frames
    assert (a.index, a.timestamp, a.target, a.boxes) == (1, 0.5, False, ())
    assert (b.index, b.timestamp, b.target) == (2, 1.0, True)
    assert b.path == folder / "b.jpg"
    assert [(x.label, x.dataset_label, x.left, x.height) for x in b.boxes] == [
        ("person", "swimmer", 1, 4)
    ]
    assert b.boxes[0].track_id is None


def test_timestamps_fall_back_to_fps(tmp_path: Path) -> None:
    write_sequence(tmp_path, "seq", "file,target\na.jpg,0\nb.jpg,0\nc.jpg,1\n")
    trace = framelist.load(tmp_path, "seq", fps=10.0)
    assert [f.timestamp for f in trace] == [0.0, 0.1, 0.2]


def test_sequences_lists_folders_with_a_frames_file(tmp_path: Path) -> None:
    write_sequence(tmp_path, "b", "file,target\n")
    write_sequence(tmp_path, "a", "file,target\n")
    (tmp_path / "not-a-sequence").mkdir()
    assert framelist.sequences(tmp_path) == ["a", "b"]


@pytest.mark.parametrize(
    "frames",
    ["file\na.jpg\n", "file,target\na.jpg,yes\n", "file,target\na.jpg,2\n"],
)
def test_bad_frame_rows_are_rejected(tmp_path: Path, frames: str) -> None:
    write_sequence(tmp_path, "seq", frames)
    with pytest.raises(ValueError):
        framelist.load(tmp_path, "seq")


def test_boxes_for_unknown_files_are_rejected(tmp_path: Path) -> None:
    write_sequence(
        tmp_path,
        "seq",
        "file,target\na.jpg,0\n",
        "file,label,dataset_label,left,top,width,height\nzzz.jpg,person,person,1,2,3,4\n",
    )
    with pytest.raises(ValueError):
        framelist.load(tmp_path, "seq")

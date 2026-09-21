"""Tests for the VisDrone-MOT loader against a fake on-disk sequence."""

from pathlib import Path

import pytest

from tagurit.sim import dataloader

SEQ = "uav0000001_00000_v"

# frame, track, left, top, width, height, score, category, truncation, occlusion
ANNOTATIONS = """\
1,1,10,20,30,40,1,1,0,0
1,2,50,60,70,80,1,0,0,0
3,1,15,25,30,40,1,4,1,2
"""


@pytest.fixture
def root(tmp_path: Path) -> Path:
    seq_dir = tmp_path / "sequences" / SEQ
    seq_dir.mkdir(parents=True)
    for i in (1, 2, 3):
        (seq_dir / f"{i:07d}.jpg").write_bytes(f"frame-{i}".encode())
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()
    (ann_dir / f"{SEQ}.txt").write_text(ANNOTATIONS)
    return tmp_path


def test_categories_cover_all_twelve_ids() -> None:
    assert sorted(dataloader.CATEGORIES) == list(range(12))
    assert dataloader.CATEGORIES[0] == "ignored"
    assert dataloader.CATEGORIES[1] == "pedestrian"
    assert dataloader.CATEGORIES[4] == "car"
    assert dataloader.CATEGORIES[11] == "others"


def test_default_root_matches_fetch_script_layout() -> None:
    assert dataloader.default_root() == Path("data") / "visdrone" / "VisDrone2019-MOT-val"
    assert dataloader.default_root("train").name == "VisDrone2019-MOT-train"


def test_sequences_lists_directories_only(root: Path) -> None:
    (root / "sequences" / "stray.txt").write_text("not a sequence")
    assert dataloader.sequences(root) == [SEQ]


def test_sequences_are_sorted(root: Path) -> None:
    (root / "sequences" / "uav0000000_00000_v").mkdir()
    assert dataloader.sequences(root) == ["uav0000000_00000_v", SEQ]


def test_sequences_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        dataloader.sequences(tmp_path)


def test_load_orders_frames_and_derives_timestamps(root: Path) -> None:
    trace = dataloader.load(root, SEQ, fps=10.0)
    assert trace.name == SEQ
    assert trace.fps == 10.0
    assert len(trace) == 3
    assert [f.index for f in trace] == [1, 2, 3]
    assert [f.timestamp for f in trace] == pytest.approx([0.0, 0.1, 0.2])
    assert all(f.sequence == SEQ for f in trace)
    assert [f.path.name for f in trace] == ["0000001.jpg", "0000002.jpg", "0000003.jpg"]


def test_load_default_fps_is_30() -> None:
    import inspect

    assert inspect.signature(dataloader.load).parameters["fps"].default == 30.0


def test_load_joins_boxes_by_frame(root: Path) -> None:
    frames = dataloader.load(root, SEQ).frames
    assert [b.label for b in frames[0].boxes] == ["pedestrian", "ignored"]
    assert frames[1].boxes == ()
    (box,) = frames[2].boxes
    assert box.label == "car"
    assert (box.left, box.top, box.width, box.height) == (15, 25, 30, 40)
    assert box.track_id == 1
    assert box.truncation == 1
    assert box.occlusion == 2


def test_frames_read_their_own_bytes(root: Path) -> None:
    frames = dataloader.load(root, SEQ).frames
    assert frames[1].read() == b"frame-2"


def _rewrite_annotations(root: Path, text: str) -> Path:
    ann = root / "annotations" / f"{SEQ}.txt"
    ann.write_text(text)
    return ann


def test_unknown_sequence_raises(root: Path) -> None:
    with pytest.raises(FileNotFoundError, match="uav0000009_00000_v"):
        dataloader.load(root, "uav0000009_00000_v")


def test_missing_annotation_file_raises(root: Path) -> None:
    (root / "annotations" / f"{SEQ}.txt").unlink()
    with pytest.raises(FileNotFoundError, match=f"{SEQ}.txt"):
        dataloader.load(root, SEQ)


def test_wrong_field_count_raises_with_line_number(root: Path) -> None:
    ann = _rewrite_annotations(root, "1,1,10,20,30,40,1,1,0,0\n1,2,50,60\n")
    with pytest.raises(ValueError, match=f"{ann.name}:2"):
        dataloader.load(root, SEQ)


def test_non_integer_field_raises_with_line_number(root: Path) -> None:
    ann = _rewrite_annotations(root, "1,1,10,20,30,40,1,1,0,0\n2,1,x,20,30,40,1,1,0,0\n")
    with pytest.raises(ValueError, match=f"{ann.name}:2"):
        dataloader.load(root, SEQ)


def test_unknown_category_raises_with_line_number(root: Path) -> None:
    ann = _rewrite_annotations(root, "1,1,10,20,30,40,1,12,0,0\n")
    with pytest.raises(ValueError, match=f"{ann.name}:1.*12"):
        dataloader.load(root, SEQ)


def test_annotation_for_missing_frame_raises(root: Path) -> None:
    ann = _rewrite_annotations(root, "9,1,10,20,30,40,1,1,0,0\n")
    with pytest.raises(ValueError, match=f"{ann.name}.*9"):
        dataloader.load(root, SEQ)


def test_trailing_comma_and_blank_lines_are_tolerated(root: Path) -> None:
    _rewrite_annotations(root, "1,1,10,20,30,40,1,1,0,0,\n\n3,1,15,25,30,40,1,4,1,2\n")
    frames = dataloader.load(root, SEQ).frames
    assert len(frames[0].boxes) == 1
    assert frames[1].boxes == ()
    assert len(frames[2].boxes) == 1

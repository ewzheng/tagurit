"""Tests for the VisDrone-MOT parser against a fake on-disk split."""

from pathlib import Path

import pytest

from tagurit.sim import visdrone

SEQ = "uav0000001_00000_v"
NAME = f"val/{SEQ}"

# frame, track, left, top, width, height, score, category, truncation, occlusion
ANNOTATIONS = """\
1,1,10,20,30,40,1,1,0,0
1,2,50,60,70,80,1,0,0,0
3,1,15,25,30,40,1,4,1,2
"""


@pytest.fixture
def root(tmp_path: Path) -> Path:
    split = tmp_path / "VisDrone2019-MOT-val"
    seq_dir = split / "sequences" / SEQ
    seq_dir.mkdir(parents=True)
    for i in (1, 2, 3):
        (seq_dir / f"{i:07d}.jpg").write_bytes(f"frame-{i}".encode())
    ann_dir = split / "annotations"
    ann_dir.mkdir()
    (ann_dir / f"{SEQ}.txt").write_text(ANNOTATIONS)
    return tmp_path


def _rewrite_annotations(root: Path, text: str) -> Path:
    ann = root / "VisDrone2019-MOT-val" / "annotations" / f"{SEQ}.txt"
    ann.write_text(text)
    return ann


@pytest.mark.parametrize(
    ("category", "dataset_label", "label"),
    [
        (0, "ignored", "ignored"),
        (1, "pedestrian", "person"),
        (2, "people", "person"),
        (3, "bicycle", "bicycle"),
        (4, "car", "car"),
        (5, "van", "car"),
        (6, "truck", "truck"),
        (7, "tricycle", "tricycle"),
        (8, "awning-tricycle", "awning-tricycle"),
        (9, "bus", "bus"),
        (10, "motor", "motorcycle"),
        (11, "others", "others"),
    ],
)
def test_labels_are_coco_names_where_one_exists(
    root: Path, category: int, dataset_label: str, label: str
) -> None:
    _rewrite_annotations(root, f"1,1,10,20,30,40,1,{category},0,0\n")
    (box,) = visdrone.load(root, NAME).frames[0].boxes
    assert (box.dataset_label, box.label) == (dataset_label, label)


def test_sequences_are_split_prefixed_and_sorted(root: Path) -> None:
    train = root / "VisDrone2019-MOT-train" / "sequences"
    (train / "uav0000009_00000_v").mkdir(parents=True)
    (root / "VisDrone2019-MOT-val" / "sequences" / "stray.txt").write_text("not a sequence")
    assert visdrone.sequences(root) == ["train/uav0000009_00000_v", NAME]


def test_sequences_without_split_folders_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        visdrone.sequences(tmp_path)


def test_load_orders_frames_and_derives_timestamps(root: Path) -> None:
    trace = visdrone.load(root, NAME, fps=10.0)
    assert trace.name == NAME
    assert trace.fps == 10.0
    assert len(trace) == 3
    assert [f.index for f in trace] == [1, 2, 3]
    assert [f.timestamp for f in trace] == pytest.approx([0.0, 0.1, 0.2])
    assert all(f.sequence == NAME for f in trace)
    assert [f.path.name for f in trace] == ["0000001.jpg", "0000002.jpg", "0000003.jpg"]


def test_load_joins_boxes_by_frame(root: Path) -> None:
    frames = visdrone.load(root, NAME).frames
    assert [b.label for b in frames[0].boxes] == ["person", "ignored"]
    assert frames[1].boxes == ()
    (box,) = frames[2].boxes
    assert box.label == "car"
    assert (box.left, box.top, box.width, box.height) == (15, 25, 30, 40)
    assert box.track_id == 1
    assert box.truncation == 1
    assert box.occlusion == 2


def test_frames_read_their_own_bytes(root: Path) -> None:
    frames = visdrone.load(root, NAME).frames
    assert frames[1].read() == b"frame-2"


def test_name_without_split_raises(root: Path) -> None:
    with pytest.raises(ValueError, match="<split>/<sequence>"):
        visdrone.load(root, SEQ)


def test_unknown_sequence_raises(root: Path) -> None:
    with pytest.raises(FileNotFoundError, match="uav0000009_00000_v"):
        visdrone.load(root, "val/uav0000009_00000_v")


def test_missing_annotation_file_raises(root: Path) -> None:
    (root / "VisDrone2019-MOT-val" / "annotations" / f"{SEQ}.txt").unlink()
    with pytest.raises(FileNotFoundError, match=f"{SEQ}.txt"):
        visdrone.load(root, NAME)


def test_wrong_field_count_raises_with_line_number(root: Path) -> None:
    ann = _rewrite_annotations(root, "1,1,10,20,30,40,1,1,0,0\n1,2,50,60\n")
    with pytest.raises(ValueError, match=f"{ann.name}:2"):
        visdrone.load(root, NAME)


def test_non_integer_field_raises_with_line_number(root: Path) -> None:
    ann = _rewrite_annotations(root, "1,1,10,20,30,40,1,1,0,0\n2,1,x,20,30,40,1,1,0,0\n")
    with pytest.raises(ValueError, match=f"{ann.name}:2"):
        visdrone.load(root, NAME)


def test_unknown_category_raises_with_line_number(root: Path) -> None:
    ann = _rewrite_annotations(root, "1,1,10,20,30,40,1,12,0,0\n")
    with pytest.raises(ValueError, match=f"{ann.name}:1.*12"):
        visdrone.load(root, NAME)


def test_annotation_for_missing_frame_raises(root: Path) -> None:
    ann = _rewrite_annotations(root, "9,1,10,20,30,40,1,1,0,0\n")
    with pytest.raises(ValueError, match=f"{ann.name}.*9"):
        visdrone.load(root, NAME)


def test_trailing_comma_and_blank_lines_are_tolerated(root: Path) -> None:
    _rewrite_annotations(root, "1,1,10,20,30,40,1,1,0,0,\n\n3,1,15,25,30,40,1,4,1,2\n")
    frames = visdrone.load(root, NAME).frames
    assert len(frames[0].boxes) == 1
    assert frames[1].boxes == ()
    assert len(frames[2].boxes) == 1

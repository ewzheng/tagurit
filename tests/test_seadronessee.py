"""Tests for the SeaDronesSee parser against a fake on-disk mirror layout."""

import json
from pathlib import Path

import pytest

from tagurit.sim import seadronessee

CATEGORIES = [
    {"id": 0, "name": "ignored", "supercategory": "ignored"},
    {"id": 1, "name": "swimmer", "supercategory": "person"},
    {"id": 2, "name": "boat", "supercategory": "boat"},
    {"id": 3, "name": "jetski", "supercategory": "boat"},
    {"id": 4, "name": "life_saving_appliances", "supercategory": "object"},
    {"id": 5, "name": "buoy", "supercategory": "object"},
]


def _video(video: str, frame_no: int) -> dict:
    return {"drone": "mavic", "folder_name": video, "video": f"{video}.MP4", "frame_no": frame_no}


def _image(image_id: int, source: dict) -> dict:
    return {
        "id": image_id,
        "file_name": f"{image_id}.jpg",
        "width": 100,
        "height": 50,
        "source": source,
    }


def _annotation(ann_id: int, image_id: int, category: int, bbox: list[float]) -> dict:
    return {"id": ann_id, "image_id": image_id, "category_id": category, "bbox": bbox, "area": 1}


TRAIN = {
    "categories": CATEGORIES,
    "images": [
        _image(10, _video("DJI_0001", 30)),
        _image(11, _video("DJI_0001", 0)),
        _image(30, _video("DJI_0002", 0)),
        _image(40, {"drone": "trinity", "folder_name": "rgb", "image_name": "DSC1.JPG"}),
    ],
    "annotations": [
        _annotation(1, 11, 1, [10, 20, 30, 40]),
        _annotation(2, 11, 0, [50, 60, 70, 80]),
        _annotation(3, 10, 2, [15.4, 25.6, 30, 40]),
        _annotation(4, 40, 1, [1, 1, 1, 1]),
        *(_annotation(10 + c, 30, c, [c, c, 5, 5]) for c in range(6)),
    ],
}
VAL = {
    "categories": CATEGORIES,
    "images": [_image(20, _video("DJI_0001", 15))],
    "annotations": [],
}


@pytest.fixture
def root(tmp_path: Path) -> Path:
    ann_dir = tmp_path / "unpacked" / "annotations"
    ann_dir.mkdir(parents=True)
    (ann_dir / "instances_train.json").write_text(json.dumps(TRAIN))
    (ann_dir / "instances_val.json").write_text(json.dumps(VAL))
    for split, ids in (("train", (10, 11, 30)), ("val", (20,))):
        img_dir = tmp_path / "unpacked" / "images" / split
        img_dir.mkdir(parents=True)
        for i in ids:
            (img_dir / f"{i}.jpg").write_bytes(f"frame-{i}".encode())
    return tmp_path


def test_sequences_lists_videos_with_all_frames_on_disk(root: Path) -> None:
    assert seadronessee.sequences(root) == ["DJI_0001", "DJI_0002"]
    (root / "unpacked" / "images" / "train" / "30.jpg").unlink()
    assert seadronessee.sequences(root) == ["DJI_0001"]


def test_sequences_without_annotations_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="instances_"):
        seadronessee.sequences(tmp_path)


def test_load_merges_splits_and_orders_by_frame_number(root: Path) -> None:
    trace = seadronessee.load(root, "DJI_0001")
    assert trace.name == "DJI_0001"
    assert trace.fps == 30.0
    assert [f.index for f in trace] == [0, 15, 30]
    assert [f.timestamp for f in trace] == pytest.approx([0.0, 0.5, 1.0])
    assert [f.path.parts[-2:] for f in trace] == [
        ("train", "11.jpg"),
        ("val", "20.jpg"),
        ("train", "10.jpg"),
    ]
    assert trace.frames[1].read() == b"frame-20"


def test_load_joins_boxes_and_has_no_track_ids(root: Path) -> None:
    frames = seadronessee.load(root, "DJI_0001").frames
    assert [b.label for b in frames[0].boxes] == ["person", "ignored"]
    assert frames[1].boxes == ()
    (box,) = frames[2].boxes
    assert box.label == "boat"
    assert (box.left, box.top, box.width, box.height) == (15, 26, 30, 40)
    assert box.track_id is None
    assert box.truncation is None
    assert box.occlusion is None


def test_labels_are_coco_names_where_one_exists(root: Path) -> None:
    (frame,) = seadronessee.load(root, "DJI_0002").frames
    assert [(b.dataset_label, b.label) for b in frame.boxes] == [
        ("ignored", "ignored"),
        ("swimmer", "person"),
        ("boat", "boat"),
        ("jetski", "boat"),
        ("life_saving_appliances", "life_saving_appliances"),
        ("buoy", "buoy"),
    ]


def test_category_without_a_label_raises(root: Path) -> None:
    floater = {"id": 6, "name": "floater", "supercategory": "person"}
    ann = root / "unpacked" / "annotations" / "instances_train.json"
    ann.write_text(json.dumps({**TRAIN, "categories": [*CATEGORIES, floater]}))
    with pytest.raises(ValueError, match="instances_train.json.*floater"):
        seadronessee.load(root, "DJI_0001")


def test_stills_without_a_source_video_are_skipped(root: Path) -> None:
    assert "rgb" not in seadronessee.sequences(root)
    with pytest.raises(FileNotFoundError, match="rgb"):
        seadronessee.load(root, "rgb")


def test_unknown_video_raises(root: Path) -> None:
    with pytest.raises(FileNotFoundError, match="DJI_9999"):
        seadronessee.load(root, "DJI_9999")


def test_missing_frame_file_raises(root: Path) -> None:
    (root / "unpacked" / "images" / "val" / "20.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="20.jpg"):
        seadronessee.load(root, "DJI_0001")


def test_custom_fps_scales_timestamps(root: Path) -> None:
    trace = seadronessee.load(root, "DJI_0001", fps=15.0)
    assert [f.timestamp for f in trace] == pytest.approx([0.0, 1.0, 2.0])

"""Tests for the trace types: lazy reads, ordering, immutability."""

from pathlib import Path

import pytest

from tagurit.sim import Box, Trace, TraceFrame


def make_box(**overrides: object) -> Box:
    fields: dict[str, object] = {
        "label": "person",
        "dataset_label": "pedestrian",
        "left": 10,
        "top": 20,
        "width": 30,
        "height": 40,
        "track_id": 1,
        "truncation": 0,
        "occlusion": 0,
    }
    fields.update(overrides)
    return Box(**fields)  # type: ignore[arg-type]


def test_frame_read_returns_file_bytes(tmp_path: Path) -> None:
    jpeg = tmp_path / "0000001.jpg"
    jpeg.write_bytes(b"not really a jpeg")
    frame = TraceFrame(sequence="seq", index=1, timestamp=0.0, path=jpeg, boxes=())
    assert frame.read() == b"not really a jpeg"


def test_trace_iterates_frames_in_order(tmp_path: Path) -> None:
    frames = tuple(
        TraceFrame(
            sequence="seq",
            index=i,
            timestamp=(i - 1) / 30.0,
            path=tmp_path / f"{i:07d}.jpg",
            boxes=(),
        )
        for i in (1, 2, 3)
    )
    trace = Trace(name="seq", fps=30.0, frames=frames)
    assert len(trace) == 3
    assert [f.index for f in trace] == [1, 2, 3]
    assert list(trace) == list(frames)


def test_types_are_immutable(tmp_path: Path) -> None:
    box = make_box()
    with pytest.raises(AttributeError):
        box.label = "car"  # type: ignore[misc]
    frame = TraceFrame(
        sequence="seq", index=1, timestamp=0.0, path=tmp_path / "0000001.jpg", boxes=(box,)
    )
    with pytest.raises(AttributeError):
        frame.index = 2  # type: ignore[misc]
    assert frame.boxes[0] == make_box()


@pytest.mark.parametrize(
    ("target", "labels", "expected"),
    [
        (True, [], True),
        (False, ["person"], False),
        (None, ["ignored", "car"], True),
        (None, [], False),
        (None, ["ignored"], None),
    ],
)
def test_has_target_prefers_the_frame_label_then_boxes(
    tmp_path: Path, target: bool | None, labels: list[str], expected: bool | None
) -> None:
    boxes = tuple(make_box(label=label, dataset_label=label) for label in labels)
    frame = TraceFrame(
        sequence="seq", index=1, timestamp=0.0, path=tmp_path / "a.jpg", boxes=boxes, target=target
    )
    assert frame.has_target() is expected


def frames_at(tmp_path: Path, times: list[float]) -> tuple[TraceFrame, ...]:
    return tuple(
        TraceFrame(sequence="seq", index=i, timestamp=t, path=tmp_path / f"{i}.jpg", boxes=())
        for i, t in enumerate(times, start=1)
    )


def test_split_of_a_recording_cuts_at_the_frame_index(tmp_path: Path) -> None:
    frames = frames_at(tmp_path, [i / 30 for i in range(10)])
    lead, trail = Trace("seq", 30.0, frames).split(0.7)
    assert lead == frames[:7] and trail == frames[7:]
    assert Trace("seq", 30.0, frames).split(0.0) == ((), frames)
    assert Trace("seq", 30.0, frames).split(1.0) == (frames, ())


def test_split_of_crops_follows_the_source_and_keeps_frames_together(tmp_path: Path) -> None:
    source = tuple(float(t) for t in range(10))  # source frame 7 starts the held-out part
    # Crops of a few source frames, unevenly many per frame, as make_sparse writes them.
    crops = frames_at(tmp_path, [1.0, 1.0, 1.0, 1.0, 1.0, 6.0, 7.0, 7.0, 9.0])
    lead, trail = Trace("seq", 1.0, crops, source_times=source).split(0.7)
    assert [f.timestamp for f in lead] == [1.0] * 5 + [6.0]
    assert [f.timestamp for f in trail] == [7.0, 7.0, 9.0]
    # Without source times the cut still never divides one frame's crops: cutting the
    # list at row 4 would have split frame 1.0's five crops across both sides.
    assert Trace("seq", 1.0, crops).split(0.5) == ((), crops)


def test_split_rejects_fractions_outside_zero_to_one(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        Trace("seq", 30.0, frames_at(tmp_path, [0.0])).split(1.5)

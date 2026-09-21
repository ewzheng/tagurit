"""Tests for the trace types: lazy reads, ordering, immutability."""

from pathlib import Path

import pytest

from tagurit.sim import Box, Trace, TraceFrame


def make_box(**overrides: object) -> Box:
    fields: dict[str, object] = {
        "label": "pedestrian",
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

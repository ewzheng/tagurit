"""Check encoder startup selection and retained hardware retry bytes offline."""

import asyncio

import cv2
import numpy as np
import pytest

from tagurit.client import encoder_selection
from tagurit.client.encoder_selection import SelectedEncoder, select_encoder
from tagurit.client.frame_scheduler import FrameScheduler
from tagurit.client.gabriel_transport import GabrielTransport
from tagurit.protocol import ImageFrame


def test_auto_selects_hardware_after_encode_and_decode_probe(monkeypatch):
    """A usable hardware encode selects QP 29 for the full run."""
    calls = []

    class Hardware:
        def __init__(self, qp):
            self.qp = qp

        def encode(self, jpeg):
            calls.append((jpeg, self.qp))
            return b"\x00\x00\x00\x01probe", 320, 240

        def close(self):
            pass

    monkeypatch.setattr(encoder_selection, "JetsonH264Encoder", Hardware)
    monkeypatch.setattr(
        encoder_selection, "decode_video", lambda *_args: np.zeros((240, 320, 3), dtype=np.uint8)
    )
    selected = select_encoder("auto", "h264", 32)
    assert selected == SelectedEncoder("hardware", "h264", 29)
    assert len(calls) == 1
    assert calls[0][1] == 29
    assert cv2.imdecode(np.frombuffer(calls[0][0], dtype=np.uint8), cv2.IMREAD_COLOR).shape == (
        240,
        320,
        3,
    )


def test_auto_falls_back_and_forced_hardware_fails(monkeypatch):
    """Missing hardware is safe in auto and explicit in hardware mode."""

    class Unavailable:
        def __init__(self, _qp):
            pass

        def encode(self, _jpeg):
            raise OSError("no encoder device")

        def close(self):
            pass

    monkeypatch.setattr(encoder_selection, "JetsonH264Encoder", Unavailable)
    selected = select_encoder("auto", "h264", 32)
    assert selected.backend == "software"
    assert "no encoder device" in selected.description()
    with pytest.raises(RuntimeError, match="no encoder device"):
        select_encoder("hardware", "h264", 32)
    assert select_encoder("software", "h264", 32).backend == "software"


def test_hardware_retry_reuses_encoded_bytes(monkeypatch):
    """Reconnect replay sends the same wire payload without another encode."""
    calls = []

    def hardware(_jpeg, qp):
        calls.append(qp)
        return b"\x00\x00\x00\x01frame", 17, 15

    monkeypatch.setattr(encoder_selection, "encode_h264", hardware)
    scheduler = FrameScheduler()
    transport = GabrielTransport(
        scheduler,
        "00000000-0000-0000-0000-000000000001",
        lambda *_args: None,
        video_encoder=SelectedEncoder("hardware", "h264", 29),
        send_interval=0,
    )
    transport._ready = True
    transport._active_attempt = 1
    transport._update_scheduler_connection()
    scheduler.add_frame(ImageFrame(1, 0.0, b"jpeg", 0.8))

    async def transmit_twice():
        first = await transport._produce(1)
        transport._active_attempt = 2
        second = await transport._produce(2)
        return first.byte_payload, second.byte_payload

    first, second = asyncio.run(transmit_twice())
    assert first == second
    assert calls == [29]

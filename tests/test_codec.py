"""Check H.264 delivery, dimensions, and identical retry payloads without a server."""

import asyncio
from types import SimpleNamespace

import cv2
import numpy as np
from gabriel_protocol.v1 import gabriel_pb2

from tagurit.client.frame_scheduler import FrameScheduler
from tagurit.client.gabriel_transport import GabrielTransport
from tagurit.cloudlet.image_receiver import ImageReceiver
from tagurit.protocol import ImageFrame
from tagurit.shared.image_codec import decode_video, encode_video
from tagurit.shared.image_protocol import decode_image, decode_receipt


def _jpeg(width: int, height: int) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, : width // 2] = (15, 80, 180)
    image[:, width // 2 :] = (170, 120, 20)
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    return encoded.tobytes()


def test_h264_odd_dimensions_and_retry(monkeypatch) -> None:
    """A lost receipt triggers the identical H.264 payload and one acceptance."""
    jpeg = _jpeg(17, 15)
    encoded, width, height = encode_video(jpeg, "h264", 32)
    assert (width, height) == (17, 15)
    assert decode_video(encoded, "h264", width, height).shape == (15, 17, 3)

    scheduler = FrameScheduler()
    session = "00000000-0000-0000-0000-000000000001"
    transport = GabrielTransport(scheduler, session, lambda _event, _detail: None)
    transport._ready = True
    transport._active_attempt = 1
    transport._update_scheduler_connection()
    scheduler.add_frame(ImageFrame(1, 0.0, jpeg, 0.8))
    monkeypatch.setattr("tagurit.client.gabriel_transport.SEND_INTERVAL_SECONDS", 0)

    async def transmit_twice():
        first = await transport._produce(1)
        transport._submitted_at = None  # Simulate a lost receipt before reconnection.
        transport._active_attempt = 2
        second = await transport._produce(2)
        return first, second

    first, second = asyncio.run(transmit_twice())
    assert first.byte_payload == second.byte_payload
    message = decode_image(first.byte_payload)
    assert message.codec == "h264"
    assert (message.width, message.height) == (17, 15)
    assert message.image_bytes != jpeg

    receiver = ImageReceiver()
    first_result = receiver.handle(first, None)
    retry_result = receiver.handle(second, None)
    assert first_result.status.code == gabriel_pb2.StatusCode.SUCCESS
    assert retry_result.payload == first_result.payload
    assert receiver.accepted_count() == 1
    receipt = decode_receipt(first_result.payload)
    assert (receipt.width, receipt.height) == (17, 15)
    assert receipt.byte_count == len(message.image_bytes)

    transport._receive_result(SimpleNamespace(string_result=retry_result.payload), 2)
    assert transport.completed == 1
    assert scheduler.pending_count() == 0

"""
Check image acceptance and duplicate receipts without starting a server
"""

from unittest.mock import patch

import cv2
import numpy as np
from gabriel_protocol.v1 import gabriel_pb2

from tagurit.cloudlet.image_receiver import ImageReceiver
from tagurit.shared.image_protocol import encode_image, make_image_message


# ============================================================
# Shared test helpers
# ============================================================

# Create a JPEG entirely in memory
def _make_jpeg(value: int) -> bytes:
    image = np.full((16, 16, 3), value, dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)

    assert success, "Could not create test JPEG"
    return encoded.tobytes()


# Build the same Gabriel input message used by the client
def _make_input(session_id: str, frame_id: int, image_bytes: bytes) -> gabriel_pb2.InputFrame:
    message = make_image_message(session_id, frame_id, image_bytes)

    return gabriel_pb2.InputFrame(
        payload_type=gabriel_pb2.PayloadType.IMAGE,
        byte_payload=encode_image(message)
    )


# ============================================================
# Receiver duplicate tracking tests
# ============================================================

# Call the receiver directly to check acceptance and retry behavior
def test_duplicate_tracking() -> None:
    """
    Check duplicate receipts, conflicting identities and invalid images

    Each check uses a fresh receiver and fixed session identities
    No server or network connection is started

    Return:
        void
    """
    session_id = "00000000-0000-0000-0000-000000000001"
    other_session_id = "00000000-0000-0000-0000-000000000002"
    original_bytes = _make_jpeg(0)
    different_bytes = _make_jpeg(255)

    # Matching retries must return the original receipt without decoding again
    def check_identical_retry() -> None:
        receiver = ImageReceiver()
        incoming = _make_input(session_id, 1, original_bytes)
        first = receiver.handle(incoming, None)

        assert first.status.code == gabriel_pb2.StatusCode.SUCCESS
        assert receiver.accepted_count() == 1

        # Any attempt to decode the duplicate must fail this check
        decode_path = "tagurit.cloudlet.image_receiver.cv2.imdecode"

        with patch(decode_path, side_effect=AssertionError("Duplicate decoded again")):
            retry = receiver.handle(incoming, None)

        assert retry.status.code == gabriel_pb2.StatusCode.SUCCESS
        assert retry.payload == first.payload
        assert receiver.accepted_count() == 1

    # Reusing an identity with different contents must preserve the original record
    def check_identity_conflict() -> None:
        receiver = ImageReceiver()
        original = _make_input(session_id, 1, original_bytes)
        first = receiver.handle(original, None)

        conflict = receiver.handle(_make_input(session_id, 1, different_bytes), None)

        assert conflict.status.code == gabriel_pb2.StatusCode.ENGINE_ERROR
        assert receiver.accepted_count() == 1
        assert receiver.handle(original, None).payload == first.payload

    # Equal image bytes with distinct identities must count as separate images
    def check_distinct_identities() -> None:
        receiver = ImageReceiver()
        identities = [(session_id, 1), (session_id, 2), (other_session_id, 1)]

        for session, frame_id in identities:
            result = receiver.handle(_make_input(session, frame_id, original_bytes), None)
            assert result.status.code == gabriel_pb2.StatusCode.SUCCESS

        assert receiver.accepted_count() == 3

    # Invalid bytes must not reserve an identity
    def check_invalid_image() -> None:
        receiver = ImageReceiver()
        invalid = receiver.handle(_make_input(session_id, 1, b"not a JPEG"), None)

        assert invalid.status.code == gabriel_pb2.StatusCode.ENGINE_ERROR
        assert receiver.accepted_count() == 0

        valid = receiver.handle(_make_input(session_id, 1, original_bytes), None)

        assert valid.status.code == gabriel_pb2.StatusCode.SUCCESS
        assert receiver.accepted_count() == 1

    check_identical_retry()
    check_identity_conflict()
    check_distinct_identities()
    check_invalid_image()
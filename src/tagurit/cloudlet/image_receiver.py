"""
Receive images through Gabriel and acknowledge accepted identities

Accepted images and receipts remain in memory for the engine's lifetime
Matching retries receive the original receipt without accepting another copy
"""

import logging
from typing import Any

import cv2
import numpy as np
from gabriel_protocol.v1 import gabriel_pb2
from gabriel_server import cognitive_engine, local_engine

from tagurit.cloudlet.config import ENGINE_ID, INPUT_QUEUE_MAXSIZE, NUM_TOKENS, SERVER_PORT
from tagurit.cloudlet.receiver_datatypes import AcceptedImage
from tagurit.shared.image_protocol import ImageReceipt, decode_image, encode_receipt


# Return an error without confirming image acceptance
def error_result(message: str) -> cognitive_engine.Result:
    """
    Build a Gabriel error result without an acceptance receipt

    Parameters:
        - message (str): Reason the image could not be accepted

    Return:
        Gabriel result containing the error
    """
    return cognitive_engine.Result(
        status=gabriel_pb2.Status(
            code=gabriel_pb2.StatusCode.ENGINE_ERROR,
            message=message
        )
    )


# Return a successful receipt through Gabriel
def receipt_result(receipt: ImageReceipt) -> cognitive_engine.Result:
    """
    Wrap an image receipt in a successful Gabriel result

    Parameters:
        - receipt (ImageReceipt): Original confirmation for an accepted image

    Return:
        Gabriel result containing the encoded receipt
    """
    return cognitive_engine.Result(
        status=gabriel_pb2.Status(code=gabriel_pb2.StatusCode.SUCCESS),
        payload=encode_receipt(receipt)
    )


# Accept each image identity once and acknowledge valid retries
class ImageReceiver(cognitive_engine.Engine):
    """
    Validate incoming images and retain accepted images with their receipts

    Each session ID and frame ID pair identifies one accepted image
    Reusing an identity with different contents returns an error
    Acceptance is recorded BEFORE returning a receipt

    Records remain in memory and do not survive an engine restart
    Calls to handle MUST be serialized for this receiver instance
    """

    # Keep accepted images for the lifetime of this engine
    def __init__(self) -> None:
        self._accepted_images: dict[tuple[str, int], AcceptedImage] = {}

    # Return how many unique images have been accepted
    def accepted_count(self) -> int:
        """
        Count accepted identities without counting retries again

        Return:
            Number of unique accepted images
        """
        return len(self._accepted_images)

    # Validate new images or return the original receipt for a retry
    def handle(self, input_frame: gabriel_pb2.InputFrame, client_info: Any) -> cognitive_engine.Result:
        """
        Accept a valid new image or acknowledge a matching retry

        MUTATES the accepted-image records only after successful validation
        Known images are checked against their original contents
        Matching retries skip image decoding and return the original receipt

        Parameters:
            - input_frame (gabriel_pb2.InputFrame): Incoming Gabriel image message
            - client_info (Any): Gabriel client metadata unused by this receiver

        Return:
            Gabriel result containing an acceptance receipt or an error
        """
        if input_frame.payload_type != gabriel_pb2.PayloadType.IMAGE:
            return error_result("Expected an image frame")

        # Check the message format and hash before looking up its identity
        try:
            message = decode_image(input_frame.byte_payload)
        except (ValueError, TypeError) as error:
            return error_result(str(error))

        key = (message.session_id, message.frame_id)
        accepted = self._accepted_images.get(key)

        # Known identities must refer to exactly the same image contents
        if accepted is not None:
            same_hash = message.sha256 == accepted.message.sha256
            same_bytes = message.image_bytes == accepted.message.image_bytes

            if not same_hash or not same_bytes:
                return error_result("Image identity was reused with different contents")

            print(
                f"DUPLICATE | session={message.session_id} | "
                f"frame={message.frame_id} | returning original receipt",
                flush=True
            )

            return receipt_result(accepted.receipt)

        # Decode only images that have not already been accepted
        try:
            encoded_image = np.frombuffer(message.image_bytes, dtype=np.uint8)
            image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)
        except cv2.error as error:
            return error_result(str(error))

        if image is None:
            return error_result("Could not decode the image")

        # Build the receipt for this new image
        height, width = image.shape[:2]

        receipt = ImageReceipt(
            status="received",
            session_id=message.session_id,
            frame_id=message.frame_id,
            sha256=message.sha256,
            byte_count=len(message.image_bytes),
            width=width,
            height=height
        )

        # Retain the image and receipt before Gabriel attempts to return the ACK
        self._accepted_images[key] = AcceptedImage(message=message, receipt=receipt)

        print(
            f"ACCEPTED | session={message.session_id} | "
            f"frame={message.frame_id} | unique={self.accepted_count()}",
            flush=True
        )

        return receipt_result(receipt)


# Start the Gabriel server with one long-lived receiver engine
def main() -> None:
    """
    Start the image receiver using the configured Gabriel server settings

    Return:
        void
    """
    logging.basicConfig(level=logging.INFO)

    server = local_engine.LocalEngine(
        engine_factory=ImageReceiver,
        input_queue_maxsize=INPUT_QUEUE_MAXSIZE,
        port=SERVER_PORT,
        num_tokens=NUM_TOKENS,
        engine_id=ENGINE_ID
    )

    print(
        f"Starting Gabriel image receiver | "
        f"port={SERVER_PORT} | engine={ENGINE_ID}",
        flush=True
    )

    try:
        server.run()
    except KeyboardInterrupt:
        print("\nImage receiver stopped", flush=True)


# Run the server when this module is launched directly
if __name__ == "__main__":
    main()
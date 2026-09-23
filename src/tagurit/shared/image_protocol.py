import hashlib
import json
import struct
from dataclasses import asdict, dataclass
from uuid import UUID


# Identify our message format and limit the metadata header size
MESSAGE_MAGIC = b"DF01"
MAX_HEADER_BYTES = 4096


# Hold one image and its stable application identity
@dataclass(frozen=True)
class ImageMessage:
    session_id: str      # Client run that produced the image
    frame_id: int        # Image number within that run
    sha256: str          # Hash of the encoded image bytes
    image_bytes: bytes  # Encoded JPEG contents


# Hold the receiver's confirmation for one image
@dataclass(frozen=True)
class ImageReceipt:
    status: str       # Whether the image was received
    session_id: str   # Client run that produced the image
    frame_id: int     # Image number within that run
    sha256: str       # Hash checked by the receiver
    byte_count: int   # Number of encoded image bytes
    width: int        # Decoded image width
    height: int       # Decoded image height


# Require a session UUID and a positive integer frame ID
def validate_identity(session_id: str, frame_id: int) -> None:
    if not isinstance(session_id, str):
        raise ValueError("Session ID must be a UUID string")

    UUID(session_id)

    if type(frame_id) is not int or frame_id < 1:
        raise ValueError("Frame ID must be a positive integer")


# Hash the encoded image and attach its application identity
def make_image_message(session_id: str, frame_id: int, image_bytes: bytes) -> ImageMessage:
    validate_identity(session_id, frame_id)

    if not image_bytes:
        raise ValueError("Image bytes must not be empty")

    return ImageMessage(
        session_id=session_id,
        frame_id=frame_id,
        sha256=hashlib.sha256(image_bytes).hexdigest(),
        image_bytes=image_bytes
    )


# Pack a short JSON header followed by the original image bytes
def encode_image(message: ImageMessage) -> bytes:
    header = json.dumps({
        "session_id": message.session_id,
        "frame_id": message.frame_id,
        "sha256": message.sha256
    }).encode("utf-8")

    if len(header) > MAX_HEADER_BYTES:
        raise ValueError("Image metadata is too large")

    return MESSAGE_MAGIC + struct.pack("!I", len(header)) + header + message.image_bytes


# Read the metadata and verify the hash against the received image bytes
def decode_image(payload: bytes) -> ImageMessage:

    # Check the format marker and read the header length
    if len(payload) < 8 or payload[:4] != MESSAGE_MAGIC:
        raise ValueError("Unknown image message format")

    header_size = struct.unpack("!I", payload[4:8])[0]

    if not 0 < header_size <= MAX_HEADER_BYTES or len(payload) <= 8 + header_size:
        raise ValueError("Incomplete or invalid image message")

    # Separate the metadata from the encoded JPEG
    header = json.loads(payload[8:8 + header_size].decode("utf-8"))

    if not isinstance(header, dict) or set(header) != {"session_id", "frame_id", "sha256"}:
        raise ValueError("Invalid image metadata")

    image_bytes = payload[8 + header_size:]
    message = make_image_message(header["session_id"], header["frame_id"], image_bytes)

    # The received contents must match the sender's hash
    if header["sha256"] != message.sha256:
        raise ValueError("Image hash does not match its contents")

    return message


# Convert a receipt to the string returned through Gabriel
def encode_receipt(receipt: ImageReceipt) -> str:
    return json.dumps(asdict(receipt))


# Read a receipt and check that its fields have valid values
def decode_receipt(payload: str) -> ImageReceipt:
    fields = json.loads(payload)

    if not isinstance(fields, dict):
        raise ValueError("Receipt must be a JSON object")

    receipt = ImageReceipt(**fields)
    validate_identity(receipt.session_id, receipt.frame_id)

    # Successful receipts describe the received image contents
    if receipt.status != "received":
        raise ValueError("Server did not confirm receipt")

    if not isinstance(receipt.sha256, str) or len(receipt.sha256) != 64:
        raise ValueError("Invalid receipt hash")

    for value in (receipt.byte_count, receipt.width, receipt.height):
        if type(value) is not int or value <= 0:
            raise ValueError("Invalid receipt image size")

    return receipt
"""
Encode one JPEG as an independent video frame and decode it on the cloudlet.

Both operations use software FFmpeg. The client can later replace the encoder
with the Jetson hardware backend without changing the wire representation.
"""

import subprocess

import cv2
import imageio_ffmpeg
import numpy as np

VIDEO_CODECS = {"h264": ("libx264", "h264"), "h265": ("libx265", "hevc")}


def _ffmpeg(arguments: list[str], payload: bytes) -> bytes:
    """Run FFmpeg on one in-memory payload and raise on codec errors."""
    try:
        result = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", *arguments],
            input=payload,
            capture_output=True,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise ValueError("Image codec timed out") from error
    if result.returncode or not result.stdout:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Image codec failed: {detail}")
    return result.stdout


def encode_video(jpeg_bytes: bytes, codec: str, crf: int) -> tuple[bytes, int, int]:
    """
    Transcode JPEG pixels to one independently decodable video frame.

    Odd dimensions are padded by repeating the outer pixels for YUV420.
    Original dimensions are returned for cropping after decode.

    Parameters:
        - jpeg_bytes (bytes): source JPEG contents
        - codec (str): h264 or h265 software encoder
        - crf (int): software encoder quality setting from 0 to 51

    Return: encoded video bytes, original width, original height
    """
    if codec not in VIDEO_CODECS:
        raise ValueError("Unsupported video codec")
    if type(crf) is not int or not 0 <= crf <= 51:
        raise ValueError("Video CRF must be an integer from 0 to 51")
    image = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode source JPEG")
    height, width = image.shape[:2]
    padded = cv2.copyMakeBorder(image, 0, height % 2, 0, width % 2, cv2.BORDER_REPLICATE)
    padded_height, padded_width = padded.shape[:2]
    encoder, bitstream = VIDEO_CODECS[codec]
    encoded = _ffmpeg(
        [
            "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size",
            f"{padded_width}x{padded_height}", "-framerate", "1", "-i", "pipe:0",
            "-frames:v", "1", "-an", "-c:v", encoder, "-preset", "veryfast",
            "-crf", str(crf), "-pix_fmt", "yuv420p", "-g", "1", "-bf", "0",
            "-threads", "2", "-f", bitstream, "pipe:1",
        ],
        padded.tobytes(),
    )
    return encoded, width, height


def decode_video(payload: bytes, codec: str, width: int, height: int) -> np.ndarray:
    """
    Decode one independent video frame to its original BGR dimensions.

    Parameters:
        - payload (bytes): encoded video access unit
        - codec (str): h264 or h265 bitstream
        - width (int): original image width before even-dimension padding
        - height (int): original image height before even-dimension padding

    Return: decoded BGR image cropped to the original dimensions
    """
    if codec not in VIDEO_CODECS:
        raise ValueError("Unsupported video codec")
    if any(type(value) is not int or not 0 < value <= 16384 for value in (width, height)):
        raise ValueError("Invalid video image dimensions")
    padded_width = width + width % 2
    padded_height = height + height % 2
    raw = _ffmpeg(
        ["-f", VIDEO_CODECS[codec][1], "-i", "pipe:0", "-frames:v", "1", "-threads", "2",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
        payload,
    )
    if len(raw) != padded_width * padded_height * 3:
        raise ValueError("Decoded video dimensions do not match the message")
    padded = np.frombuffer(raw, dtype=np.uint8).reshape(padded_height, padded_width, 3)
    return padded[:height, :width]

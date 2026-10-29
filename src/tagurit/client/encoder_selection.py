"""Choose one H.264 encoder when the client starts and keep it for the run."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from tagurit.client.jetson_h264 import HARDWARE_QP, JetsonH264Encoder, encode_h264
from tagurit.shared.image_codec import decode_video, encode_video

ENCODER_MODES = ("auto", "software", "hardware")


@dataclass(frozen=True)
class SelectedEncoder:
    """
    Bind a tested backend and quality setting for all new frames in one run.

    Parameters:
        - backend (str): software or hardware
        - codec (str): selected wire video codec
        - quality (int): software CRF or hardware QP
        - reason (str | None): why automatic selection fell back
        - hardware_encoder (JetsonH264Encoder | None): persistent Jetson pipeline
    """

    backend: str
    codec: str
    quality: int
    reason: str | None = None
    hardware_encoder: JetsonH264Encoder | None = field(default=None, compare=False, repr=False)

    def encode(self, jpeg_bytes: bytes) -> tuple[bytes, int, int]:
        """
        Encode a new frame with the selected backend.

        Parameters:
            - jpeg_bytes (bytes): source JPEG contents

        Return: encoded bytes and original dimensions
        """
        if self.backend == "hardware":
            if self.hardware_encoder is not None:
                return self.hardware_encoder.encode(jpeg_bytes)
            return encode_h264(jpeg_bytes, self.quality)
        return encode_video(jpeg_bytes, self.codec, self.quality)

    def close(self) -> None:
        """
        Release a persistent hardware pipeline at the end of the client run.

        Return: void
        """
        if self.hardware_encoder is not None:
            self.hardware_encoder.close()

    def description(self) -> str:
        """
        Describe the backend and any automatic fallback for the client log.

        Return: human-readable selection
        """
        setting = f"QP {self.quality}" if self.backend == "hardware" else f"CRF {self.quality}"
        detail = f"{self.codec} {self.backend} {setting}"
        return f"{detail} | auto fallback: {self.reason}" if self.reason else detail


def select_encoder(mode: str, codec: str, software_crf: int) -> SelectedEncoder:
    """
    Probe Jetson hardware once or select the portable software encoder.

    A real patterned JPEG is hardware-encoded and decoded in isolation so an
    installed plugin without a usable device does not pass the probe. Auto
    falls back only during selection; encoding errors later in the run surface.

    Parameters:
        - mode (str): auto, software, or hardware
        - codec (str): wire video codec
        - software_crf (int): software quality setting

    Return: selected encoder for the lifetime of the client run
    """
    if mode not in ENCODER_MODES:
        raise ValueError(f"Encoder must be one of {', '.join(ENCODER_MODES)}")
    if codec not in ("h264", "h265"):
        if mode == "hardware":
            raise ValueError("Hardware encoding is available only for H.264")
        return SelectedEncoder("software", codec, software_crf)
    if mode == "software":
        return SelectedEncoder("software", codec, software_crf)
    if codec != "h264":
        if mode == "hardware":
            raise ValueError("Hardware encoding is available only for H.264")
        return SelectedEncoder("software", codec, software_crf)

    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:, :160] = (20, 90, 180)
    image[:, 160:] = (160, 120, 30)
    success, jpeg = cv2.imencode(".jpg", image)
    if not success:
        raise RuntimeError("Could not create encoder probe image")
    hardware = None
    try:
        hardware = JetsonH264Encoder(HARDWARE_QP)
        encoded, width, height = hardware.encode(jpeg.tobytes())
        if decode_video(encoded, "h264", width, height).shape != image.shape:
            raise ValueError("Jetson encoder probe decoded to incorrect dimensions")
    except (OSError, RuntimeError, ValueError, ImportError) as error:
        if hardware is not None:
            hardware.close()
        if mode == "hardware":
            raise RuntimeError(f"Jetson hardware H.264 encoder unavailable: {error}") from error
        return SelectedEncoder("software", codec, software_crf, str(error))
    return SelectedEncoder("hardware", codec, HARDWARE_QP, hardware_encoder=hardware)

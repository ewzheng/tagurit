"""
Keep a Jetson V4L2 H.264 encoder alive across independently decodable images.

The GStreamer pipeline is rebuilt only when padded image dimensions change.
All calls into one pipeline are serialized, including shutdown. Each output
is an IDR access unit with its own SPS/PPS headers for cloudlet decoding.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

HARDWARE_QP = 29
SYSTEM_GI_PATH = Path("/usr/lib/python3/dist-packages")


def _load_gstreamer() -> Any:
    """
    Load JetPack's system GStreamer bindings from the isolated Python venv.

    The Jetson has PyGObject in the Ubuntu system packages, while the uv venv
    omits system site packages. Other hosts simply fail the startup probe.

    Return: initialized Gst module
    """
    try:
        import gi
    except ImportError:
        if not sys.platform.startswith("linux") or not SYSTEM_GI_PATH.is_dir():
            raise RuntimeError("Jetson GStreamer Python bindings are unavailable") from None
        if str(SYSTEM_GI_PATH) not in sys.path:
            sys.path.append(str(SYSTEM_GI_PATH))
        try:
            import gi
        except ImportError as error:
            raise RuntimeError("Jetson GStreamer Python bindings are unavailable") from error
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    return Gst


class JetsonH264Encoder:
    """
    Serialize independent H.264 frames through one hardware pipeline.

    The caller MUST call close after its worker tasks finish. A resolution
    change closes the old pipeline before starting one with new caps.

    Parameters:
        - qp (int): constant I-frame quantization parameter
    """

    def __init__(self, qp: int = HARDWARE_QP) -> None:
        if type(qp) is not int or not 0 <= qp <= 51:
            raise ValueError("Hardware H.264 QP must be an integer from 0 to 51")
        self._gst = _load_gstreamer()
        self._qp = qp
        self._lock = threading.Lock()
        self._pipeline: Any = None
        self._source: Any = None
        self._sink: Any = None
        self._size: tuple[int, int] | None = None
        self._frame_index = 0

    def _stop_pipeline(self) -> None:
        if self._pipeline is not None:
            self._pipeline.set_state(self._gst.State.NULL)
        self._pipeline = None
        self._source = None
        self._sink = None
        self._size = None
        self._frame_index = 0

    def _start_pipeline(self, width: int, height: int) -> None:
        description = (
            "appsrc name=source is-live=false format=time block=true "
            "! videoconvert ! video/x-raw,format=I420 "
            "! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 "
            "! nvv4l2h264enc preset-level=0 ratecontrol-enable=false "
            f"quant-i-frames={self._qp} iframeinterval=1 idrinterval=1 "
            "insert-sps-pps=true "
            "! video/x-h264,stream-format=byte-stream,alignment=au "
            "! appsink name=sink sync=false max-buffers=1"
        )
        pipeline = self._gst.parse_launch(description)
        source = pipeline.get_by_name("source")
        sink = pipeline.get_by_name("sink")
        source.set_property(
            "caps",
            self._gst.Caps.from_string(
                f"video/x-raw,format=BGR,width={width},height={height},framerate=1/1"
            ),
        )
        result = pipeline.set_state(self._gst.State.PLAYING)
        if result == self._gst.StateChangeReturn.FAILURE:
            pipeline.set_state(self._gst.State.NULL)
            raise RuntimeError("Jetson H.264 pipeline could not start")
        self._pipeline = pipeline
        self._source = source
        self._sink = sink
        self._size = (width, height)
        self._frame_index = 0

    def encode(self, jpeg_bytes: bytes) -> tuple[bytes, int, int]:
        """
        Encode one JPEG as a standalone Annex B H.264 access unit.

        Odd dimensions are padded by repeating the outside pixels. Original
        dimensions are returned so the receiver crops after decoding.

        Parameters:
            - jpeg_bytes (bytes): source JPEG contents

        Return: encoded H.264 bytes, original width, and original height
        """
        image = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode source JPEG")
        height, width = image.shape[:2]
        padded = cv2.copyMakeBorder(image, 0, height % 2, 0, width % 2, cv2.BORDER_REPLICATE)
        padded_height, padded_width = padded.shape[:2]

        with self._lock:
            try:
                if self._size != (padded_width, padded_height):
                    self._stop_pipeline()
                    self._start_pipeline(padded_width, padded_height)
                buffer = self._gst.Buffer.new_allocate(None, padded.nbytes, None)
                buffer.fill(0, padded.tobytes())
                buffer.pts = self._frame_index * self._gst.SECOND
                buffer.duration = self._gst.SECOND
                if self._source.emit("push-buffer", buffer) != self._gst.FlowReturn.OK:
                    raise RuntimeError("Jetson H.264 input was rejected")
                sample = self._sink.emit("try-pull-sample", 10 * self._gst.SECOND)
                if sample is None:
                    message = self._pipeline.get_bus().timed_pop_filtered(
                        0, self._gst.MessageType.ERROR
                    )
                    detail = message.parse_error()[0] if message is not None else "timed out"
                    raise RuntimeError(f"Jetson H.264 produced no frame: {detail}")
                output = sample.get_buffer()
                mapped_ok, mapped = output.map(self._gst.MapFlags.READ)
                if not mapped_ok:
                    raise RuntimeError("Could not read Jetson H.264 output")
                try:
                    encoded = bytes(mapped.data)
                finally:
                    output.unmap(mapped)
                if not encoded.startswith(bytes.fromhex("00000001")):
                    raise ValueError("Jetson encoder did not produce an Annex B H.264 frame")
                self._frame_index += 1
                return encoded, width, height
            except Exception:
                self._stop_pipeline()
                raise

    def close(self) -> None:
        """
        Release the hardware pipeline after client workers have stopped.

        Return: void
        """
        with self._lock:
            self._stop_pipeline()


def encode_h264(jpeg_bytes: bytes, qp: int = HARDWARE_QP) -> tuple[bytes, int, int]:
    """
    Encode one image with a temporary Jetson encoder instance.

    Production runs keep a JetsonH264Encoder instance across many images.

    Parameters:
        - jpeg_bytes (bytes): source JPEG contents
        - qp (int): constant I-frame quantization parameter

    Return: encoded H.264 bytes, original width, and original height
    """
    encoder = JetsonH264Encoder(qp)
    try:
        return encoder.encode(jpeg_bytes)
    finally:
        encoder.close()

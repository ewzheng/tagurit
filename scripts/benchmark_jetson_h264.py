"""
Compare Jetson hardware H.264 QP settings with Tagurit's software CRF 32.

Each source JPEG is encoded as a standalone IDR frame. All size and quality
measurements use the same decoded source JPEG as the reference. This benchmark
uses a fresh GStreamer process and temporary raw file per image, so its timing
includes process startup and file I/O and is not a steady-state hardware rate.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from tagurit.shared.image_codec import decode_video, encode_video

FIELDS = (
    "image",
    "encoder",
    "quality_setting",
    "width",
    "height",
    "jpeg_bytes",
    "h264_bytes",
    "ratio_to_jpeg",
    "ssim_luma",
    "psnr_db",
    "encode_ms",
)


def ssim_luma(reference: np.ndarray, decoded: np.ndarray) -> float:
    """
    Calculate the same grayscale SSIM used by compare_lossy_codecs.py.

    Parameters:
        - reference (np.ndarray): JPEG-decoded BGR pixels
        - decoded (np.ndarray): H.264-decoded BGR pixels

    Return: mean grayscale SSIM
    """
    left = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    right = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY).astype(np.float32)

    def blur(image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(image, (11, 11), 1.5)

    mean_left = blur(left)
    mean_right = blur(right)
    variance_left = blur(left * left) - mean_left * mean_left
    variance_right = blur(right * right) - mean_right * mean_right
    covariance = blur(left * right) - mean_left * mean_right
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    return float(
        np.mean(
            ((2 * mean_left * mean_right + c1) * (2 * covariance + c2))
            / (
                (mean_left * mean_left + mean_right * mean_right + c1)
                * (variance_left + variance_right + c2)
            )
        )
    )


def sample_images(directory: Path, count: int) -> list[Path]:
    """
    Select evenly spaced JPEGs in filename order.

    Parameters:
        - directory (Path): source image directory
        - count (int): maximum images to select

    Return: selected JPEG paths
    """
    images = sorted((*directory.glob("*.jpg"), *directory.glob("*.jpeg")))
    if not images:
        raise ValueError(f"No JPEGs in {directory}")
    indexes = np.linspace(0, len(images) - 1, min(count, len(images)), dtype=int)
    return [images[int(index)] for index in indexes]


def hardware_encode(jpeg: bytes, qp: int) -> tuple[bytes, int, int]:
    """
    Encode a standalone H.264 IDR frame with the Jetson V4L2 encoder.

    Parameters:
        - jpeg (bytes): source image
        - qp (int): constant I-frame quantization parameter

    Return: H.264 bytes and original dimensions
    """
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode source JPEG")
    height, width = image.shape[:2]
    padded = cv2.copyMakeBorder(image, 0, height % 2, 0, width % 2, cv2.BORDER_REPLICATE)
    padded_height, padded_width = padded.shape[:2]

    with tempfile.TemporaryDirectory(prefix="tagurit-h264-sweep-") as directory:
        raw_path = Path(directory) / "frame.bgr"
        encoded_path = Path(directory) / "frame.h264"
        raw_path.write_bytes(padded.tobytes())
        command = [
            "gst-launch-1.0",
            "-q",
            "filesrc",
            f"location={raw_path}",
            "!",
            "rawvideoparse",
            "format=bgr",
            f"width={padded_width}",
            f"height={padded_height}",
            "framerate=1/1",
            "!",
            "videoconvert",
            "!",
            "video/x-raw,format=I420",
            "!",
            "nvvidconv",
            "!",
            "video/x-raw(memory:NVMM),format=NV12",
            "!",
            "nvv4l2h264enc",
            "preset-level=0",
            "ratecontrol-enable=false",
            f"quant-i-frames={qp}",
            "iframeinterval=1",
            "idrinterval=1",
            "insert-sps-pps=true",
            "!",
            "filesink",
            f"location={encoded_path}",
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise RuntimeError(f"Jetson encoder failed (QP {qp}): {result.stderr.strip()}")
        encoded = encoded_path.read_bytes()
    if not encoded.startswith(bytes.fromhex("00000001")):
        raise ValueError("Hardware output is not an Annex B H.264 byte stream")
    return encoded, width, height


def measure(image_path: Path, encoder: str, setting: int) -> dict[str, str | int | float]:
    """
    Encode, independently decode, and measure one JPEG.

    Parameters:
        - image_path (Path): source JPEG
        - encoder (str): software or hardware
        - setting (int): software CRF or hardware QP

    Return: one CSV row
    """
    jpeg = image_path.read_bytes()
    reference = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if reference is None:
        raise ValueError(f"Could not decode {image_path}")
    started = time.perf_counter()
    if encoder == "software":
        encoded, width, height = encode_video(jpeg, "h264", setting)
    else:
        encoded, width, height = hardware_encode(jpeg, setting)
    encode_ms = (time.perf_counter() - started) * 1000
    decoded = decode_video(encoded, "h264", width, height)
    if decoded.shape != reference.shape:
        raise ValueError(f"Decoded dimensions differ for {image_path}")
    return {
        "image": str(image_path),
        "encoder": encoder,
        "quality_setting": setting,
        "width": width,
        "height": height,
        "jpeg_bytes": len(jpeg),
        "h264_bytes": len(encoded),
        "ratio_to_jpeg": round(len(encoded) / len(jpeg), 6),
        "ssim_luma": round(ssim_luma(reference, decoded), 5),
        "psnr_db": round(cv2.PSNR(reference, decoded), 3),
        "encode_ms": round(encode_ms, 3),
    }


def main() -> None:
    """
    Sweep quality settings and write per-image and summary CSV files.

    Return: void
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images", type=Path, default=Path("data/seadronessee/unpacked/images/val")
    )
    parser.add_argument("--count", type=int, default=28)
    parser.add_argument("--qps", type=int, nargs="+", default=[20, 24, 28, 32, 36, 40])
    parser.add_argument(
        "--output", type=Path, default=Path("data/compression/jetson-h264-qp-sweep")
    )
    args = parser.parse_args()
    if args.count < 1 or not args.qps or any(qp < 0 or qp > 51 for qp in args.qps):
        parser.error("count must be positive and QP values must be in 0..51")
    if args.output.exists():
        parser.error(f"Output directory already exists: {args.output}")
    images = sample_images(args.images, args.count)
    args.output.mkdir(parents=True)
    rows = []
    with (args.output / "per-image.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, FIELDS)
        writer.writeheader()
        for index, image_path in enumerate(images, 1):
            for encoder, setting in [("software", 32), *[("hardware", qp) for qp in args.qps]]:
                row = measure(image_path, encoder, setting)
                writer.writerow(row)
                destination.flush()
                rows.append(row)
            print(f"{index}/{len(images)} images complete: {image_path.name}", flush=True)

    summaries = []
    for encoder, setting in [("software", 32), *[("hardware", qp) for qp in args.qps]]:
        group = [
            row for row in rows if row["encoder"] == encoder and row["quality_setting"] == setting
        ]
        summaries.append(
            {
                "encoder": encoder,
                "quality_setting": setting,
                "images": len(group),
                "jpeg_bytes_total": sum(int(row["jpeg_bytes"]) for row in group),
                "h264_bytes_total": sum(int(row["h264_bytes"]) for row in group),
                "ratio_total": round(
                    sum(int(row["h264_bytes"]) for row in group)
                    / sum(int(row["jpeg_bytes"]) for row in group),
                    6,
                ),
                "median_ssim": round(
                    statistics.median(float(row["ssim_luma"]) for row in group), 5
                ),
                "min_ssim": round(min(float(row["ssim_luma"]) for row in group), 5),
                "median_psnr_db": round(
                    statistics.median(float(row["psnr_db"]) for row in group), 3
                ),
                "median_encode_ms": round(
                    statistics.median(float(row["encode_ms"]) for row in group), 3
                ),
            }
        )
    with (args.output / "summary.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(destination, summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    for summary in summaries:
        print(
            f"{summary['encoder']} {summary['quality_setting']}: "
            f"{summary['ratio_total']:.3f}x JPEG; "
            f"median SSIM {summary['median_ssim']:.3f}; "
            f"min SSIM {summary['min_ssim']:.3f}; "
            f"median encode {summary['median_encode_ms']:.1f} ms",
            flush=True,
        )
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()

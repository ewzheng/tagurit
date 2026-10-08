"""
Measure independent lossless H.264 frames against source JPEGs.

Each JPEG is decoded with OpenCV, encoded as one H.264 IDR frame by
libx264rgb, then decoded in isolation. The CSV is flushed after every frame
so a long run can resume. Encoded images are not kept on disk.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from tagurit.sim import dataloader
from tagurit.sim.trace import TraceFrame

FIELDS = (
    "sequence",
    "frame_index",
    "path",
    "width",
    "height",
    "jpeg_bytes",
    "h264_bytes",
    "ratio_h264_to_jpeg",
    "encode_ms",
    "decode_ms",
    "mismatched_values",
    "max_abs_difference",
)


def run_ffmpeg(ffmpeg: str, arguments: list[str], payload: bytes) -> bytes:
    """
    Run FFmpeg with bytes on stdin and return its output bytes.

    Parameters:
        - ffmpeg (str): path to the FFmpeg executable
        - arguments (list[str]): FFmpeg options following the executable
        - payload (bytes): input image or H.264 stream

    Return: bytes emitted to stdout
    """
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", *arguments],
        input=payload,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
    return result.stdout


def benchmark_frame(ffmpeg: str, frame: TraceFrame) -> dict[str, str | int | float]:
    """
    Encode and independently decode one JPEG, checking every pixel value.

    The reference is OpenCV's decoded BGR image. RGB channel reordering for
    libx264rgb is reversible, so an exact result has zero differing values.

    Parameters:
        - ffmpeg (str): path to FFmpeg with libx264rgb
        - frame (TraceFrame): dataset frame whose JPEG is read once

    Return: CSV row with byte sizes, timings, and fidelity results
    """
    jpeg = frame.read()
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode JPEG: {frame.path}")
    height, width = image.shape[:2]
    raw = image.tobytes()

    started = time.perf_counter()
    encoded = run_ffmpeg(
        ffmpeg,
        [
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgr24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            "1",
            "-i",
            "pipe:0",
            "-frames:v",
            "1",
            "-an",
            "-c:v",
            "libx264rgb",
            "-preset",
            "veryfast",
            "-qp",
            "0",
            "-g",
            "1",
            "-bf",
            "0",
            "-threads",
            "2",
            "-f",
            "h264",
            "pipe:1",
        ],
        raw,
    )
    encode_ms = (time.perf_counter() - started) * 1000
    if not encoded:
        raise ValueError(f"Encoder produced no bytes for {frame.path}")

    started = time.perf_counter()
    decoded = run_ffmpeg(
        ffmpeg,
        [
            "-f",
            "h264",
            "-i",
            "pipe:0",
            "-frames:v",
            "1",
            "-threads",
            "2",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ],
        encoded,
    )
    decode_ms = (time.perf_counter() - started) * 1000
    if len(decoded) != len(raw):
        raise ValueError(f"Decoded byte count differs for {frame.path}")
    reconstructed = np.frombuffer(decoded, dtype=np.uint8).reshape(image.shape)
    differences = cv2.absdiff(image, reconstructed)

    return {
        "sequence": frame.sequence,
        "frame_index": frame.index,
        "path": str(frame.path),
        "width": width,
        "height": height,
        "jpeg_bytes": len(jpeg),
        "h264_bytes": len(encoded),
        "ratio_h264_to_jpeg": round(len(encoded) / len(jpeg), 6),
        "encode_ms": round(encode_ms, 3),
        "decode_ms": round(decode_ms, 3),
        "mismatched_values": int(np.count_nonzero(differences)),
        "max_abs_difference": int(differences.max()),
    }


def print_summary(rows: list[dict[str, str]]) -> None:
    """
    Print aggregate sizes, fidelity, and median timings from CSV rows.

    Parameters:
        - rows (list[dict[str, str]]): completed benchmark records

    Return: void
    """
    if not rows:
        print("No frames measured")
        return
    jpeg_bytes = sum(int(row["jpeg_bytes"]) for row in rows)
    h264_bytes = sum(int(row["h264_bytes"]) for row in rows)
    smaller = sum(int(row["h264_bytes"]) < int(row["jpeg_bytes"]) for row in rows)
    mismatched = sum(int(row["mismatched_values"]) != 0 for row in rows)
    encode_ms = statistics.median(float(row["encode_ms"]) for row in rows)
    decode_ms = statistics.median(float(row["decode_ms"]) for row in rows)
    print(
        f"{len(rows)} frames | JPEG {jpeg_bytes / 1e9:.3f} GB | "
        f"H.264 {h264_bytes / 1e9:.3f} GB | ratio {h264_bytes / jpeg_bytes:.2f}x | "
        f"H.264 smaller: {smaller} | pixel mismatches: {mismatched} | "
        f"median encode {encode_ms:.0f} ms, decode {decode_ms:.0f} ms",
        flush=True,
    )


def main() -> None:
    """
    Benchmark the VisDrone validation set, resuming an existing CSV.

    Return: void
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/compression/visdrone-val.csv"))
    parser.add_argument("--limit", type=int, help="process at most this many frames")
    parser.add_argument("--workers", type=int, default=4, help="parallel FFmpeg jobs (default: 4)")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    names = dataloader.sequences("visdrone")
    frames = (frame for name in names for frame in dataloader.load("visdrone", name))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    if args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as existing:
            reader = csv.DictReader(existing)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ValueError(f"Unexpected CSV columns in {args.output}")
            rows = [row for row in reader if int(row["mismatched_values"]) == 0]
    completed = {row["path"] for row in rows}
    selected = []
    for total, frame in enumerate(frames, start=1):
        if args.limit is not None and total > args.limit:
            break
        if str(frame.path) not in completed:
            selected.append(frame)
    with args.output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        destination.flush()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for frame, result in zip(
                selected, pool.map(partial(benchmark_frame, ffmpeg), selected), strict=True
            ):
                if result["mismatched_values"]:
                    raise ValueError(f"Lossless round trip changed pixels: {frame.path}")
                writer.writerow(result)
                destination.flush()
                rows.append({key: str(value) for key, value in result.items()})
                if len(rows) % 100 == 0:
                    print_summary(rows)
    print_summary(rows)
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()

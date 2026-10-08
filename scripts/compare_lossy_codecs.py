"""
Compare independent lossy H.264 and H.265 frames with their source JPEGs.

The reference is the image decoded from the JPEG. Results describe additional
loss from transcoding, not the loss already present in the JPEG. The program
uses a small, spread-out sample by default and writes a CSV and previews.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from benchmark_h264 import run_ffmpeg

from tagurit.sim import dataloader
from tagurit.sim.trace import TraceFrame

FIELDS = (
    "sequence",
    "frame_index",
    "codec",
    "crf",
    "width",
    "height",
    "jpeg_bytes",
    "video_bytes",
    "ratio_video_to_jpeg",
    "psnr_db",
    "ssim_luma",
    "mean_abs_difference",
    "encode_ms",
    "decode_ms",
)
CODECS = {"h264": ("libx264", "h264"), "h265": ("libx265", "hevc")}


def sample_frames(per_sequence: int, limit: int | None) -> list[TraceFrame]:
    """
    Choose evenly spaced frames from every available VisDrone sequence.

    Parameters:
        - per_sequence (int): maximum frames from each sequence
        - limit (int | None): optional total image limit for a quick run

    Return: selected TraceFrame records
    """
    chosen = []
    for name in dataloader.sequences("visdrone"):
        trace = dataloader.load("visdrone", name)
        count = min(per_sequence, len(trace))
        indexes = np.linspace(0, len(trace) - 1, num=count, dtype=int)
        chosen.extend(trace.frames[int(i)] for i in indexes)
    return chosen[:limit]


def ssim_luma(reference: np.ndarray, decoded: np.ndarray) -> float:
    """
    Compute structural similarity on the full grayscale images.

    Parameters:
        - reference (np.ndarray): original JPEG-decoded BGR pixels
        - decoded (np.ndarray): decoded video BGR pixels of the same shape

    Return: average luminance SSIM, where 1 is an identical image
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
    numerator = (2 * mean_left * mean_right + c1) * (2 * covariance + c2)
    denominator = (mean_left * mean_left + mean_right * mean_right + c1) * (
        variance_left + variance_right + c2
    )
    return float(np.mean(numerator / denominator))


def preview(reference: np.ndarray, decoded: np.ndarray, path: Path, label: str) -> None:
    """
    Save a resized original/decoded comparison for visual inspection.

    Parameters:
        - reference (np.ndarray): original JPEG-decoded BGR pixels
        - decoded (np.ndarray): decoded video BGR pixels
        - path (Path): output PNG location
        - label (str): codec setting shown on the comparison image

    Return: void
    """
    scale = min(1.0, 960 / reference.shape[1])
    size = (round(reference.shape[1] * scale), round(reference.shape[0] * scale))
    left = cv2.resize(reference, size, interpolation=cv2.INTER_AREA)
    right = cv2.resize(decoded, size, interpolation=cv2.INTER_AREA)
    canvas = np.concatenate((left, right), axis=1)
    cv2.putText(canvas, "Original JPEG", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(
        canvas, "Original JPEG", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )
    cv2.putText(canvas, label, (size[0] + 12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(
        canvas, label, (size[0] + 12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"Could not write preview: {path}")
    crop_size = min(512, reference.shape[0], reference.shape[1])
    top = (reference.shape[0] - crop_size) // 2
    left_edge = (reference.shape[1] - crop_size) // 2
    crop = np.concatenate(
        (
            reference[top : top + crop_size, left_edge : left_edge + crop_size],
            decoded[top : top + crop_size, left_edge : left_edge + crop_size],
        ),
        axis=1,
    )
    crop_path = path.with_name(f"{path.stem}-crop.png")
    if not cv2.imwrite(str(crop_path), crop):
        raise OSError(f"Could not write preview crop: {crop_path}")


def compare_frame(
    ffmpeg: str, frame: TraceFrame, codec: str, crf: int, preview_path: Path | None
) -> dict[str, str | int | float]:
    """
    Encode and independently decode one frame, then measure size and quality.

    Parameters:
        - ffmpeg (str): path to the software FFmpeg binary
        - frame (TraceFrame): source JPEG frame
        - codec (str): h264 or h265
        - crf (int): codec quality setting, lower means higher quality
        - preview_path (Path | None): optional comparison PNG destination

    Return: CSV row of sizes, quality measures, and timings
    """
    jpeg = frame.read()
    original = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if original is None:
        raise ValueError(f"Could not decode JPEG: {frame.path}")
    height, width = original.shape[:2]
    # YUV420 encoders require even dimensions. Repeat the outer pixels for padding.
    padded = cv2.copyMakeBorder(original, 0, height % 2, 0, width % 2, cv2.BORDER_REPLICATE)
    encoded_height, encoded_width = padded.shape[:2]
    encoder, bitstream = CODECS[codec]

    started = time.perf_counter()
    encoded = run_ffmpeg(
        ffmpeg,
        [
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgr24",
            "-video_size",
            f"{encoded_width}x{encoded_height}",
            "-framerate",
            "1",
            "-i",
            "pipe:0",
            "-frames:v",
            "1",
            "-an",
            "-c:v",
            encoder,
            "-preset",
            "veryfast",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-g",
            "1",
            "-bf",
            "0",
            "-threads",
            "2",
            "-f",
            bitstream,
            "pipe:1",
        ],
        padded.tobytes(),
    )
    encode_ms = (time.perf_counter() - started) * 1000
    if not encoded:
        raise ValueError(f"Encoder produced no bytes for {frame.path}")

    started = time.perf_counter()
    raw = run_ffmpeg(
        ffmpeg,
        [
            "-f",
            bitstream,
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
    if len(raw) != padded.size:
        raise ValueError(f"Decoded byte count differs for {frame.path}")
    reconstructed = np.frombuffer(raw, dtype=np.uint8).reshape(padded.shape)[:height, :width]

    if preview_path is not None:
        preview(original, reconstructed, preview_path, f"{codec.upper()} CRF {crf}")

    return {
        "sequence": frame.sequence,
        "frame_index": frame.index,
        "codec": codec,
        "crf": crf,
        "width": width,
        "height": height,
        "jpeg_bytes": len(jpeg),
        "video_bytes": len(encoded),
        "ratio_video_to_jpeg": round(len(encoded) / len(jpeg), 6),
        "psnr_db": round(cv2.PSNR(original, reconstructed), 3),
        "ssim_luma": round(ssim_luma(original, reconstructed), 5),
        "mean_abs_difference": round(float(cv2.absdiff(original, reconstructed).mean()), 3),
        "encode_ms": round(encode_ms, 3),
        "decode_ms": round(decode_ms, 3),
    }


def main() -> None:
    """
    Compare selected images and save results as CSV and preview PNGs.

    Return: void
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-sequence", type=int, default=2)
    parser.add_argument("--limit", type=int, help="maximum source images")
    parser.add_argument("--codecs", choices=("h264", "h265", "both"), default="both")
    parser.add_argument("--crf", type=int, nargs="+", default=[18, 23, 28])
    parser.add_argument("--output", type=Path, default=Path("data/compression/lossy-sample.csv"))
    args = parser.parse_args()
    if args.per_sequence < 1 or (args.limit is not None and args.limit < 1):
        parser.error("frame counts must be positive")
    if any(not 0 <= crf <= 51 for crf in args.crf):
        parser.error("CRF must be between 0 and 51")

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    frames = sample_frames(args.per_sequence, args.limit)
    codecs = tuple(CODECS) if args.codecs == "both" else (args.codecs,)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    preview_root = args.output.parent / "previews"
    rows = []
    with args.output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=FIELDS)
        writer.writeheader()
        for frame in frames:
            for codec in codecs:
                for crf in args.crf:
                    preview_path = None
                    if frame is frames[0]:
                        preview_path = preview_root / f"{codec}-crf{crf}.png"
                    row = compare_frame(ffmpeg, frame, codec, crf, preview_path)
                    writer.writerow(row)
                    destination.flush()
                    rows.append(row)
                    print(
                        f"{frame.sequence}/{frame.index} {codec} CRF {crf}: "
                        f"{row['ratio_video_to_jpeg']:.2f}x JPEG, "
                        f"PSNR {row['psnr_db']:.1f} dB, SSIM {row['ssim_luma']:.3f}",
                        flush=True,
                    )
    for codec in codecs:
        for crf in args.crf:
            group = [row for row in rows if row["codec"] == codec and row["crf"] == crf]
            ratio = sum(row["video_bytes"] for row in group) / sum(
                row["jpeg_bytes"] for row in group
            )
            median_psnr = statistics.median(row["psnr_db"] for row in group)
            median_ssim = statistics.median(row["ssim_luma"] for row in group)
            print(
                f"{codec} CRF {crf}: {ratio:.2f}x JPEG bytes, "
                f"median PSNR {median_psnr:.1f} dB, SSIM {median_ssim:.3f}"
            )
    print(f"Results: {args.output} | Previews: {preview_root}")


if __name__ == "__main__":
    main()

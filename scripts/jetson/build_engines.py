"""
Build TensorRT engines for the CLIP image encoder on the Jetson: float16, and int8.

Usage (on the Jetson, in the clipnn venv):
    python build_engines.py switch/stage --out models

Reads ``clip-b16-vision.onnx``, ``calibration.npy`` and ``manifest.json``
from the stage directory that ``scripts/experiment_clipnn_switch.py prepare``
writes. Both engines take float32 pixel batches of 1 to 32 tiles, tuned for
24 (one 1280x720 frame), and return unit embeddings. The int8 engine is
calibrated with TensorRT's entropy calibrator on the staged tiles, which come
from empty frames outside the mission; layers that TensorRT finds faster in
float16 stay float16. A calibration cache is kept next to the engine.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

LOGGER = trt.Logger(trt.Logger.WARNING)
INPUT = "pixel_values"
SHAPES = {"min": 1, "opt": 24, "max": 32}


class Calibrator(trt.IInt8EntropyCalibrator2):
    """
    Feed normalised tiles to TensorRT's int8 entropy calibration.

    Parameters:
        - tiles (np.ndarray): (N, size, size, 3) uint8 RGB tiles
        - mean (list[float]): per-channel mean
        - std (list[float]): per-channel standard deviation
        - batch (int): tiles per calibration batch
        - cache (Path): calibration cache file, read if present
    """

    def __init__(
        self, tiles: np.ndarray, mean: list[float], std: list[float], batch: int, cache: Path
    ) -> None:
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.tiles, self.batch, self.cache = tiles, batch, cache
        self.mean = torch.tensor(mean, device="cuda").view(1, 3, 1, 1)
        self.std = torch.tensor(std, device="cuda").view(1, 3, 1, 1)
        self.index = 0
        self.current: torch.Tensor | None = None

    def get_batch_size(self) -> int:
        return self.batch

    def get_batch(self, names: list[str]) -> list[int] | None:
        if self.index + self.batch > len(self.tiles):
            return None
        chunk = torch.from_numpy(self.tiles[self.index : self.index + self.batch]).cuda()
        self.index += self.batch
        pixels = chunk.permute(0, 3, 1, 2).float().div_(255)
        self.current = ((pixels - self.mean) / self.std).contiguous()
        torch.cuda.synchronize()
        return [int(self.current.data_ptr())]

    def read_calibration_cache(self) -> bytes | None:
        return self.cache.read_bytes() if self.cache.exists() else None

    def write_calibration_cache(self, cache: object) -> None:
        self.cache.write_bytes(bytes(cache))


def build(onnx: Path, out: Path, size: int, calibrator: Calibrator | None) -> None:
    """
    Parse the ONNX encoder and write a serialised engine.

    Parameters:
        - onnx (Path): the exported encoder
        - out (Path): engine file to write
        - size (int): tile side the encoder takes
        - calibrator (Calibrator | None): int8 calibration; None builds float16 only

    Return: void
    """
    builder = trt.Builder(LOGGER)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, LOGGER)
    if not parser.parse_from_file(str(onnx)):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(f"cannot parse {onnx}: {errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    profile.set_shape(INPUT, *((n, 3, size, size) for n in SHAPES.values()))
    config.add_optimization_profile(profile)
    if calibrator is not None:
        # calibration runs at one fixed shape, so it needs its own static profile
        fixed = builder.create_optimization_profile()
        shape = (calibrator.get_batch_size(), 3, size, size)
        fixed.set_shape(INPUT, shape, shape, shape)
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator
        config.set_calibration_profile(fixed)
    start = time.perf_counter()
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError(f"TensorRT could not build {out}")
    out.write_bytes(bytes(engine))
    print(f"built {out} in {time.perf_counter() - start:.0f}s, {out.stat().st_size / 1e6:.0f} MB")


def main() -> None:
    """
    Build the float16 and int8 engines.

    Return: void
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("--out", type=Path, default=Path("models"))
    parser.add_argument("--batch", type=int, default=16, help="calibration batch")
    args = parser.parse_args()

    manifest = json.loads((args.stage / "manifest.json").read_text())
    onnx = args.stage / "clip-b16-vision.onnx"
    size = manifest["tile"]
    args.out.mkdir(parents=True, exist_ok=True)
    build(onnx, args.out / "clip-b16-fp16.engine", size, None)
    tiles = np.load(args.stage / "calibration.npy")
    cache = args.out / "clip-b16-int8.calib"
    calibrator = Calibrator(tiles, manifest["mean"], manifest["std"], args.batch, cache)
    build(onnx, args.out / "clip-b16-int8.engine", size, calibrator)


if __name__ == "__main__":
    main()

"""
Replay the clipnn switch mission on the Jetson: encode, store, switch, re-rank.

Usage (on the Jetson, in the clipnn venv):
    python clipnn_scout.py switch/stage --engines models --out switch/results

For each engine (float16, int8), first a steady-state benchmark: batches of
24 real tiles for a few seconds, with board power from tegrastats. Then the
mission for each dataset in the stage: every frame is decoded, cut into its
staged tiles, encoded, and its tile embeddings kept as float16, timing the
decode and the encode separately. At the switch the backlog (the first half)
is re-ranked for query B against the vocabulary reject set, with nothing but
the stored embeddings, timed on the GPU and on the CPU, from float16 storage
and from int8 storage (one scale per vector). Embeddings are written per
engine and dataset for scoring on the cloudlet, and every timing goes to
``timings.json``.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch
from PIL import Image

LOGGER = trt.Logger(trt.Logger.WARNING)
VDD_IN = re.compile(r"VDD_IN (\d+)mW")
ENGINES = ("fp16", "int8")
QUERY_B = {"sparse-seadronessee": "swimmers", "sparse-visdrone": "people"}


class Power:
    """Average total board power (VDD_IN) over a block, sampled by tegrastats."""

    def __enter__(self) -> Power:
        self.log = Path(tempfile.mkstemp(suffix=".log")[1])
        self.proc = subprocess.Popen(
            ["tegrastats", "--interval", "100", "--logfile", str(self.log)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(0.3)
        return self

    def __exit__(self, *exc: object) -> None:
        self.proc.terminate()
        self.proc.wait()
        readings = [int(m.group(1)) for m in VDD_IN.finditer(self.log.read_text())]
        self.log.unlink()
        self.watts = statistics.mean(readings) / 1000 if readings else float("nan")


class Encoder:
    """
    A TensorRT CLIP image encoder taking uint8 tiles.

    Parameters:
        - path (Path): serialised engine
        - mean (list[float]): per-channel mean
        - std (list[float]): per-channel standard deviation
    """

    def __init__(self, path: Path, mean: list[float], std: list[float]) -> None:
        self.engine = trt.Runtime(LOGGER).deserialize_cuda_engine(path.read_bytes())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor(mean, device="cuda").view(1, 3, 1, 1)
        self.std = torch.tensor(std, device="cuda").view(1, 3, 1, 1)

    def __call__(self, tiles: np.ndarray) -> np.ndarray:
        """
        Encode tiles.

        Parameters:
            - tiles (np.ndarray): (N, size, size, 3) uint8 RGB tiles

        Return: (N, 512) float16 unit embeddings
        """
        with torch.cuda.stream(self.stream):
            pixels = torch.from_numpy(tiles).cuda().permute(0, 3, 1, 2).float().div_(255)
            pixels = ((pixels - self.mean) / self.std).contiguous()
            out = torch.empty((len(tiles), 512), dtype=torch.float32, device="cuda")
            self.context.set_input_shape("pixel_values", tuple(pixels.shape))
            self.context.set_tensor_address("pixel_values", pixels.data_ptr())
            self.context.set_tensor_address("embeds", out.data_ptr())
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("TensorRT execution failed")
            result = out.half().cpu()
        self.stream.synchronize()
        return result.numpy()


def benchmark(encoder: Encoder, tiles: np.ndarray, seconds: float, idle: float) -> dict:
    """
    Steady-state encoding speed and energy on one batch of real tiles.

    Parameters:
        - encoder (Encoder): the engine
        - tiles (np.ndarray): one batch of uint8 tiles
        - seconds (float): timed run length
        - idle (float): idle board power, watts

    Return: ms per tile, board watts, and millijoules per tile in total and above idle
    """
    for _ in range(5):
        encoder(tiles)
    runs = 0
    with Power() as power:
        start = time.perf_counter()
        while time.perf_counter() - start < seconds:
            encoder(tiles)
            runs += 1
        elapsed = time.perf_counter() - start
    ms = 1000 * elapsed / (runs * len(tiles))
    return {
        "ms_per_tile": round(ms, 3),
        "watts": round(power.watts, 2),
        "mj_per_tile": round(power.watts * ms, 1),
        "mj_per_tile_above_idle": round((power.watts - idle) * ms, 1),
    }


def fly(encoder: Encoder, stage: Path, frames: list[dict], size: int) -> tuple[np.ndarray, dict]:
    """
    Run the mission: decode each frame, cut its tiles, encode them, keep the embeddings.

    Parameters:
        - encoder (Encoder): the engine
        - stage (Path): stage directory the frame paths are relative to
        - frames (list[dict]): manifest frames, in mission order
        - size (int): tile side

    Return: (tiles, 512) float16 embeddings in mission order, and per-frame timings
    """
    store, decode, encode = [], 0.0, 0.0
    with Power() as power:
        start = time.perf_counter()
        for entry in frames:
            t0 = time.perf_counter()
            image = np.asarray(Image.open(stage / entry["path"]).convert("RGB"))
            tiles = np.stack([image[t : t + size, s : s + size] for s, t in entry["tiles"]])
            t1 = time.perf_counter()
            store.append(encoder(tiles))
            encode += time.perf_counter() - t1
            decode += t1 - t0
        elapsed = time.perf_counter() - start
    n = len(frames)
    return np.concatenate(store), {
        "frames": n,
        "tiles": sum(len(f["tiles"]) for f in frames),
        "decode_ms_per_frame": round(1000 * decode / n, 2),
        "encode_ms_per_frame": round(1000 * encode / n, 2),
        "watts": round(power.watts, 2),
        "joules_per_frame": round(power.watts * elapsed / n, 3),
    }


def rerank(
    bank: np.ndarray, owner: np.ndarray, phrases: np.ndarray, targets: int, scale: float
) -> dict:
    """
    Time re-ranking a stored backlog for a new query, on the GPU and the CPU.

    Each run scores every stored tile as the softmax share of the query
    phrases against the reject phrases, takes each frame's best tile, and
    sorts the frames.

    Parameters:
        - bank (np.ndarray): (N, 512) float16 stored tile embeddings
        - owner (np.ndarray): (N,) frame index of each tile
        - phrases (np.ndarray): (P + Q, 512) query then reject phrase embeddings
        - targets (int): P, the number of query phrases
        - scale (float): CLIP's logit scale

    Return: median milliseconds per re-rank for each storage and device
    """
    x = bank.astype(np.float32)
    scales = np.abs(x).max(axis=1, keepdims=True) / 127
    q8 = np.round(x / scales).astype(np.int8)
    frames = int(owner.max()) + 1

    def rank_cpu(emb: np.ndarray) -> np.ndarray:
        logits = scale * emb @ phrases.astype(np.float32).T
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        tile = probs[:, :targets].sum(axis=1) / probs.sum(axis=1)
        best = np.full(frames, -np.inf, dtype=np.float32)
        np.maximum.at(best, owner, tile)
        return np.argsort(-best)

    gpu_bank = torch.from_numpy(bank).cuda()
    gpu_q8 = torch.from_numpy(q8).cuda()
    gpu_scales = torch.from_numpy(scales).cuda().half()
    gpu_phrases = torch.from_numpy(phrases).cuda().half()
    gpu_owner = torch.from_numpy(owner).cuda()

    def rank_gpu(emb: torch.Tensor) -> np.ndarray:
        probs = (scale * (emb @ gpu_phrases.T)).float().softmax(dim=1)
        tile = probs[:, :targets].sum(dim=1)
        best = torch.full((frames,), -1.0, device="cuda").scatter_reduce(
            0, gpu_owner, tile, reduce="amax"
        )
        return best.argsort(descending=True).cpu().numpy()

    def timed(fn: object) -> float:
        times = []
        for _ in range(7):
            torch.cuda.synchronize()
            start = time.perf_counter()
            fn()  # type: ignore[operator]
            torch.cuda.synchronize()
            times.append(1000 * (time.perf_counter() - start))
        return round(statistics.median(times[2:]), 2)

    return {
        "tiles": len(bank),
        "frames": frames,
        "gpu_float16_ms": timed(lambda: rank_gpu(gpu_bank)),
        "gpu_int8_ms": timed(lambda: rank_gpu(gpu_q8.half() * gpu_scales)),
        "cpu_float16_ms": timed(lambda: rank_cpu(x)),
        "cpu_int8_ms": timed(lambda: rank_cpu(q8.astype(np.float32) * scales)),
        "store_bytes_float16": int(bank.nbytes),
        "store_bytes_int8": int(q8.nbytes + 2 * len(q8)),
    }


def main() -> None:
    """
    Benchmark both engines, fly the mission with each, and time the switch.

    Return: void
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("--engines", type=Path, default=Path("models"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--seconds", type=float, default=6.0)
    args = parser.parse_args()

    manifest = json.loads((args.stage / "manifest.json").read_text())
    messages = np.load(args.stage / "messages.npz")
    calibration = np.load(args.stage / "calibration.npy")
    datasets = [k for k in manifest if k.startswith("sparse-")]
    size = manifest["tile"]
    args.out.mkdir(parents=True, exist_ok=True)

    with Power() as idle:
        time.sleep(3)
    timings: dict = {"idle_watts": round(idle.watts, 2)}
    print(f"idle board power {idle.watts:.1f} W", flush=True)
    for name in ENGINES:
        encoder = Encoder(
            args.engines / f"clip-b16-{name}.engine", manifest["mean"], manifest["std"]
        )
        bench = benchmark(encoder, calibration[:24], args.seconds, idle.watts)
        timings[f"{name}/benchmark_batch24"] = bench
        print(f"{name}: {bench}", flush=True)
        for dataset in datasets:
            frames = manifest[dataset]["frames"]
            embeddings, flight = fly(encoder, args.stage, frames, size)
            np.save(args.out / f"{name}-{dataset}.npy", embeddings)
            timings[f"{name}/{dataset}/mission"] = flight
            print(f"{name} {dataset}: {flight}", flush=True)
        del encoder
        torch.cuda.empty_cache()

    for dataset in datasets:
        frames = manifest[dataset]["frames"]
        switch = manifest[dataset]["switch"]
        owner = np.concatenate([np.full(len(f["tiles"]), i) for i, f in enumerate(frames)])
        backlog = owner < switch
        bank = np.load(args.out / f"int8-{dataset}.npy")[backlog]
        query = messages[f"{dataset}/{QUERY_B[dataset]}"]
        phrases = np.concatenate([query, messages["vocabulary"]])
        result = rerank(bank, owner[backlog], phrases, len(query), float(messages["scale"]))
        result["message_bytes"] = int(query.nbytes)
        timings[f"switch/{dataset}"] = result
        print(f"switch {dataset}: {result}", flush=True)
    (args.out / "timings.json").write_text(json.dumps(timings, indent=2))


if __name__ == "__main__":
    main()

"""
Run the same finite image benchmark across several fixed client windows.

The cloudlet and any SSH tunnel must already be running. Each window gets its
own raw events, frame rows, summary, and console log. A combined CSV and a
terminal table compare the completed runs without losing individual evidence.
"""

import argparse
import csv
import hashlib
import json
import math
import socket
import statistics
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

from tagurit.client.config import GABRIEL_ENDPOINT

BENCHMARK = Path(__file__).with_name("benchmark_flow_control.py")
REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_IMAGES = REPOSITORY / "data/visdrone/VisDrone2019-MOT-val/sequences"
WINDOWS = (1, 2, 4, 8, 16)
FIELDS = (
    "condition",
    "trial",
    "client_window",
    "frames_offered",
    "frames_acknowledged",
    "mean_submit_to_receipt_ms",
    "acknowledged_frames_per_s",
    "retried_frames",
    "status",
    "raw_events",
    "run_directory",
)


def read_result(run_directory: Path, condition: str, trial: int, window: int) -> dict:
    """
    Read one benchmark result and calculate delay from first submission.

    Parameters:
        - run_directory (Path): Benchmark output with summary and frame CSV
        - condition (str): Network profile label supplied by the operator
        - trial (int): Repetition number supplied by the operator
        - window (int): Client's fixed in-flight limit

    Return: One flat row for the sweep CSV
    """
    with (run_directory / "summary.json").open() as stream:
        summary = json.load(stream)
    with (run_directory / "frames.csv").open(newline="") as stream:
        frames = list(csv.DictReader(stream))

    delays = [
        float(frame["first_submit_to_receipt_ms"])
        for frame in frames
        if frame["status"] == "acknowledged" and frame["first_submit_to_receipt_ms"]
    ]
    acknowledged = summary["acknowledged"]
    if len(delays) != acknowledged or len(frames) != summary["arrived"]:
        raise ValueError(f"Frame rows do not match the summary in {run_directory}")
    status = "complete" if summary["unfinished"] == 0 else "incomplete"
    if summary["retried_frames"]:
        status += "; retries"

    return {
        "condition": condition,
        "trial": trial,
        "client_window": window,
        "frames_offered": summary["arrived"],
        "frames_acknowledged": acknowledged,
        "mean_submit_to_receipt_ms": statistics.mean(delays) if delays else "",
        "acknowledged_frames_per_s": summary["acknowledged_frames_per_s"],
        "retried_frames": summary["retried_frames"],
        "status": status,
        "raw_events": str(run_directory / "client-events.csv"),
        "run_directory": str(run_directory),
    }


def save_table(path: Path, rows: list[dict]) -> None:
    """
    Save completed sweep rows as a spreadsheet-friendly CSV.

    Parameters:
        - path (Path): Combined CSV destination
        - rows (list[dict]): Results in test order

    Return: void
    """
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: list[dict]) -> None:
    """
    Print the two chosen measurements and completion status for each run.

    Parameters:
        - rows (list[dict]): Results in test order

    Return: void
    """
    print("\n| Window | Frames acked | Mean submit-to-receipt | Acked frames/s | Status |")
    print("|---:|---:|---:|---:|---|")
    for row in rows:
        delay = row["mean_submit_to_receipt_ms"]
        delay_text = f"{delay:.1f} ms" if delay != "" else "n/a"
        print(
            f"| {row['client_window']} | "
            f"{row['frames_acknowledged']}/{row['frames_offered']} | "
            f"{delay_text} | {row['acknowledged_frames_per_s']:.2f} | "
            f"{row['status']} |"
        )


def check_endpoint(endpoint: str) -> None:
    """
    Fail quickly when the selected local or remote port is not listening.

    Parameters:
        - endpoint (str): Gabriel WebSocket URL

    Return: void
    """
    parsed = urlsplit(endpoint)
    if parsed.scheme not in ("ws", "wss") or not parsed.hostname:
        raise ValueError("Endpoint must be a ws:// or wss:// URL with a host")
    port = parsed.port or (443 if parsed.scheme == "wss" else 80)
    with socket.create_connection((parsed.hostname, port), timeout=3):
        pass


def main() -> None:
    """
    Execute each fixed-window benchmark and preserve a combined comparison.

    Return: void; an unsuccessful run exits nonzero after saving available data
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--condition", default="normal (conditioner off)")
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--windows", nargs="+", type=int, default=WINDOWS)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--arrival-interval", type=float, default=0.1)
    parser.add_argument("--send-interval", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--endpoint", default=GABRIEL_ENDPOINT)
    args = parser.parse_args()

    if not args.condition.strip():
        parser.error("condition must have a name")
    if args.trial < 1 or args.count < 1:
        parser.error("trial and count must be positive")
    if not args.windows or any(window < 1 for window in args.windows):
        parser.error("windows must contain positive integers")
    if len(set(args.windows)) != len(args.windows):
        parser.error("windows must not contain duplicates")
    for name in ("arrival_interval", "send_interval", "timeout"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name == "timeout" and value == 0):
            parser.error(f"{name.replace('_', '-')} must be finite and nonnegative")
    if args.output.exists():
        parser.error(f"Output directory already exists: {args.output}")
    if len(list(args.images.rglob("*.jpg"))) < args.count:
        parser.error(f"Need at least {args.count} JPEGs under {args.images}")
    try:
        check_endpoint(args.endpoint)
    except (OSError, ValueError) as error:
        parser.error(f"Cannot reach {args.endpoint}: {error}")

    output = args.output.resolve()
    output.mkdir(parents=True)
    settings = {
        **vars(args),
        "output": str(output),
        "images": str(args.images.resolve()),
        "server_tokens_required": max(args.windows),
    }
    (output / "sweep-config.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"Condition: {args.condition}")
    print(f"Server must already have at least {max(args.windows)} tokens.")

    rows: list[dict] = []
    input_hash: str | None = None
    for window in args.windows:
        run_directory = output / f"window-{window}"
        command = [
            sys.executable,
            str(BENCHMARK),
            "--images",
            str(args.images.resolve()),
            "--window",
            str(window),
            "--count",
            str(args.count),
            "--arrival-interval",
            str(args.arrival_interval),
            "--send-interval",
            str(args.send_interval),
            "--endpoint",
            args.endpoint,
            "--timeout",
            str(args.timeout),
            "--output",
            str(run_directory),
        ]
        print(f"Running window {window}...", flush=True)
        log_path = output / f"window-{window}-console.log"
        with log_path.open("w") as log:
            result = subprocess.run(
                command, cwd=REPOSITORY, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        if not (run_directory / "summary.json").exists():
            raise SystemExit(f"Window {window} failed; inspect {log_path}")
        current_hash = hashlib.sha256((run_directory / "inputs.csv").read_bytes()).hexdigest()
        if input_hash is not None and current_hash != input_hash:
            raise SystemExit(
                f"Window {window} used different input images; inspect {run_directory}"
            )
        input_hash = current_hash
        row = read_result(run_directory, args.condition, args.trial, window)
        if row["frames_offered"] != args.count or row["frames_acknowledged"] != args.count:
            row["status"] = "incomplete"
        if result.returncode != 0:
            row["status"] += "; process error"
        rows.append(row)
        save_table(output / "window-results.csv", rows)
        print_table(rows)
        if result.returncode != 0 or row["status"].startswith("incomplete"):
            raise SystemExit(f"Window {window} did not complete cleanly; inspect {log_path}")

    print(f"\nSpreadsheet CSV: {output / 'window-results.csv'}")


if __name__ == "__main__":
    main()

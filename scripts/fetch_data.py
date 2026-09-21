"""
Fetch a dataset into data/<dataset> so the sim and the viewer can load it.

Usage:
    uv run python scripts/fetch_data.py visdrone [--split val|train|test-dev]
    uv run python scripts/fetch_data.py seadronessee [--videos 5 | --all]

VisDrone-MOT comes as one zip per split from the Hugging Face mirror
vanthanh/VisDrone2019-MOT, checked against the SHA-256 pinned below and
extracted. SeaDronesSee comes as individual frames from the mirror
lalalostcode/SeaDroneSee_ObjectDetect_Compresed: the two annotation files
first, then every frame of the N longest videos (default 5, about 2.7 GB),
or of all 39 videos with --all (about 9 GB). Both are idempotent: what is
already on disk is not downloaded again.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from huggingface_hub import hf_hub_download  # noqa: E402

from tagurit.sim import seadronessee  # noqa: E402

DATA_DIR = Path("data")

VISDRONE_REPO = "vanthanh/VisDrone2019-MOT"
# The LFS object ids Hugging Face publishes for each zip, read 2026-09-17.
VISDRONE_SHA256 = {
    "val": "e53571990dfc79229e0a8ae10264bc4fa604a027c44b06e3a097417e4fa55705",
    "train": "566d08fb53fff4e539f386f5a408ccf17854fd53814dc756bdede2de1dbb4014",
    "test-dev": "758abe40bf20246e7e778ac61eaa557cf004b034b6e68e85006e5add68e17eb5",
}

SEADRONESSEE_REPO = "lalalostcode/SeaDroneSee_ObjectDetect_Compresed"
SEADRONESSEE_WORKERS = 8


def sha256_of(path: Path) -> str:
    """
    Hash a file in chunks so a multi-gigabyte zip never sits in memory.

    Parameters:
        - path (Path): file to hash

    Return: lowercase hex digest
    """
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_visdrone(split: str, dest: Path) -> Path:
    """
    Download, verify, and extract one VisDrone-MOT split.

    An extracted folder short-circuits the run. A zip that verifies is not
    downloaded again and is kept so a failed extraction does not re-download.

    Parameters:
        - split (str): "val", "train", or "test-dev"
        - dest (Path): the dataset folder, ``data/visdrone``

    Return: the extracted split folder, ``dest/VisDrone2019-MOT-<split>``
    """
    dest.mkdir(parents=True, exist_ok=True)
    zip_name = f"VisDrone2019-MOT-{split}.zip"
    zip_path = dest / zip_name
    extracted = dest / f"VisDrone2019-MOT-{split}"
    if extracted.is_dir():
        print(f"already extracted: {extracted}")
        return extracted

    if zip_path.is_file() and sha256_of(zip_path) == VISDRONE_SHA256[split]:
        print(f"already downloaded and verified: {zip_path}")
    else:
        print(f"downloading {zip_name} from {VISDRONE_REPO} ...")
        hf_hub_download(
            repo_id=VISDRONE_REPO, repo_type="dataset", filename=zip_name, local_dir=dest
        )
        actual = sha256_of(zip_path)
        if actual != VISDRONE_SHA256[split]:
            zip_path.unlink()
            sys.exit(f"checksum mismatch for {zip_name}: got {actual}")
        print(f"verified {zip_name}")

    print(f"extracting into {dest} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    if not extracted.is_dir():
        sys.exit(f"extraction finished but {extracted} does not exist")
    print(f"done: {extracted}")
    return extracted


def fetch_seadronessee(videos: int | None, dest: Path) -> list[str]:
    """
    Download the annotation files, then every frame of the chosen videos.

    Videos are ranked by frame count and the longest ``videos`` are taken;
    None takes all of them. Frames already on disk are skipped, so a
    partial run resumes. Downloads run on a small thread pool because the
    mirror stores frames as individual files.

    Parameters:
        - videos (int | None): how many of the longest videos to fetch, None for all
        - dest (Path): the dataset folder, ``data/seadronessee``

    Return: names of the videos that are now complete on disk
    """
    dest.mkdir(parents=True, exist_ok=True)
    for split in seadronessee.SPLITS:
        hf_hub_download(
            repo_id=SEADRONESSEE_REPO,
            repo_type="dataset",
            filename=f"unpacked/annotations/instances_{split}.json",
            local_dir=dest,
        )
    by_video = seadronessee.index(dest)
    ranked = sorted(by_video, key=lambda v: -len(by_video[v]))
    chosen = ranked if videos is None else ranked[:videos]
    wanted = [im for v in chosen for im in by_video[v]]
    todo = [im for im in wanted if not im.path(dest).is_file()]
    print(f"{len(chosen)} videos, {len(wanted)} frames, {len(todo)} to download")
    for v in chosen:
        print(f"  {v}: {len(by_video[v])} frames")

    def fetch_one(im: seadronessee.ImageRecord) -> None:
        hf_hub_download(
            repo_id=SEADRONESSEE_REPO,
            repo_type="dataset",
            filename=f"unpacked/images/{im.split}/{im.file_name}",
            local_dir=dest,
        )

    with ThreadPoolExecutor(SEADRONESSEE_WORKERS) as pool:
        for n, _ in enumerate(pool.map(fetch_one, todo), start=1):
            if n % 200 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} frames")
    ready = seadronessee.sequences(dest)
    print(f"done: {len(ready)} videos complete under {dest}")
    return ready


def main(argv: list[str] | None = None) -> None:
    """
    Parse arguments and fetch one dataset.

    Parameters:
        - argv (list[str] | None): arguments without the program name; None means sys.argv

    Return: void
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="dataset", required=True)
    visdrone = sub.add_parser("visdrone", help="VisDrone-MOT, one zip per split")
    visdrone.add_argument("--split", choices=sorted(VISDRONE_SHA256), default="val")
    sds = sub.add_parser("seadronessee", help="SeaDronesSee detection set, per video")
    group = sds.add_mutually_exclusive_group()
    group.add_argument("--videos", type=int, default=5, help="fetch the N longest (default 5)")
    group.add_argument("--all", action="store_true", help="fetch every video")
    args = parser.parse_args(argv)

    if args.dataset == "visdrone":
        fetch_visdrone(args.split, DATA_DIR / "visdrone")
    else:
        fetch_seadronessee(None if args.all else args.videos, DATA_DIR / "seadronessee")


if __name__ == "__main__":
    main()

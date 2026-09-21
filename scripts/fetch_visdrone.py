"""
Fetch a VisDrone-MOT split into data/visdrone and unpack it.

Usage:
    uv run python scripts/fetch_visdrone.py [--split val|train|test-dev] [--dest data/visdrone]

Downloads the official zip from the Hugging Face mirror
(vanthanh/VisDrone2019-MOT), checks it against the SHA-256 pinned below,
and extracts it. Idempotent: an extracted folder short-circuits the run,
and a zip that verifies is not downloaded again. The zip is kept so a
re-run after a failed extraction does not re-download.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "vanthanh/VisDrone2019-MOT"

# The LFS object ids Hugging Face publishes for each zip, read 2026-09-17.
SHA256 = {
    "val": "e53571990dfc79229e0a8ae10264bc4fa604a027c44b06e3a097417e4fa55705",
    "train": "566d08fb53fff4e539f386f5a408ccf17854fd53814dc756bdede2de1dbb4014",
    "test-dev": "758abe40bf20246e7e778ac61eaa557cf004b034b6e68e85006e5add68e17eb5",
}


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


def fetch(split: str, dest: Path) -> Path:
    """
    Download, verify, and extract one split.

    Parameters:
        - split (str): "val", "train", or "test-dev"
        - dest (Path): directory that holds the zip and the extracted folder

    Return: the extracted split folder, ``dest/VisDrone2019-MOT-<split>``
    """
    dest.mkdir(parents=True, exist_ok=True)
    zip_name = f"VisDrone2019-MOT-{split}.zip"
    zip_path = dest / zip_name
    extracted = dest / f"VisDrone2019-MOT-{split}"
    if extracted.is_dir():
        print(f"already extracted: {extracted}")
        return extracted

    if zip_path.is_file() and sha256_of(zip_path) == SHA256[split]:
        print(f"already downloaded and verified: {zip_path}")
    else:
        print(f"downloading {zip_name} from {REPO_ID} ...")
        hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=zip_name, local_dir=dest)
        actual = sha256_of(zip_path)
        if actual != SHA256[split]:
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


def main(argv: list[str] | None = None) -> None:
    """
    Parse arguments and fetch one split.

    Parameters:
        - argv (list[str] | None): arguments without the program name; None means sys.argv

    Return: void
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", choices=sorted(SHA256), default="val")
    parser.add_argument("--dest", type=Path, default=Path("data") / "visdrone")
    args = parser.parse_args(argv)
    fetch(args.split, args.dest)


if __name__ == "__main__":
    main()

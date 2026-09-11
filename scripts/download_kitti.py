#!/usr/bin/env python3
"""
Download KITTI Odometry Sequences
===================================
Usage:
    python scripts/download_kitti.py --sequences 00 05 07 --dest data/kitti

The KITTI Odometry dataset requires **free registration** at:
    https://www.cvlibs.net/datasets/kitti/eval_odometry.php

This script downloads:
  • Greyscale images (image_0) for each sequence
  • Ground-truth poses for sequences 00-10
  • Calibration files

Files are large (~22 GB for all, ~2 GB for seq 00 images only).
You must accept the KITTI license before downloading.

Alternatively, download manually and extract to:
    data/kitti/sequences/XX/
    data/kitti/poses/
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

# ── KITTI download URLs ────────────────────────────────────────────────────────
# These are the direct download URLs from the KITTI server.
# Note: KITTI requires registration and provides direct links after login.
# If these fail, download manually from https://www.cvlibs.net/datasets/kitti/eval_odometry.php

KITTI_BASE_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti"

KITTI_FILES = {
    "odometry_gray":   "data_odometry_gray.zip",    # ~22 GB greyscale images all sequences
    "odometry_color":  "data_odometry_color.zip",   # ~65 GB colour images
    "odometry_poses":  "data_odometry_poses.zip",   # ~4 MB ground truth (sequences 00-10)
    "odometry_calib":  "data_odometry_calib.zip",   # ~1 MB calibration files
}

# ── Sequence-level download (individual sequences via academic torrents) ───────
# The Karlsruhe Institute dataset mirrors allow per-sequence downloads.
SEQUENCE_URLS = {
    "00": f"{KITTI_BASE_URL}/data_odometry_gray.zip",
    "poses": f"{KITTI_BASE_URL}/data_odometry_poses.zip",
    "calib": f"{KITTI_BASE_URL}/data_odometry_calib.zip",
}


def download_file(url: str, dest: Path, desc: str = "") -> None:
    """Download with a progress bar."""
    print(f"\n{'─'*60}")
    print(f"Downloading: {desc or url}")
    print(f"        To:  {dest}")
    print(f"{'─'*60}")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, int(100 * downloaded / total_size))
            bar = "█" * (pct // 2) + "░" * (50 - pct // 2)
            mb_done = downloaded / 1e6
            mb_total = total_size / 1e6
            print(f"\r  [{bar}] {pct:3d}%  {mb_done:.1f}/{mb_total:.1f} MB", end="", flush=True)
        else:
            print(f"\r  Downloaded {downloaded / 1e6:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=reporthook)
    print()  # newline after progress bar


def extract_zip(zip_path: Path, dest: Path) -> None:
    """Extract a ZIP archive."""
    import zipfile
    print(f"\nExtracting {zip_path.name} → {dest}/")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)
    print("  Done.")


def check_sequence_exists(data_dir: Path, seq_id: str) -> bool:
    """Return True if the sequence images are already present."""
    seq_path = data_dir / "sequences" / seq_id / "image_0"
    if not seq_path.exists():
        return False
    imgs = list(seq_path.glob("*.png"))
    return len(imgs) > 0


def print_manual_instructions() -> None:
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║              KITTI Manual Download Instructions                      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  1. Go to: https://www.cvlibs.net/datasets/kitti/eval_odometry.php  ║
║  2. Register (free) and accept the license.                          ║
║  3. Download:                                                        ║
║     • "odometry data set (grayscale, 22 GB)"  → images              ║
║     • "odometry ground truth poses"            → poses               ║
║     • "odometry calibrations"                  → calibration         ║
║  4. Extract all files so the layout matches:                         ║
║                                                                      ║
║     data/kitti/                                                      ║
║     ├── sequences/                                                   ║
║     │   ├── 00/                                                      ║
║     │   │   ├── image_0/   ← PNG frames                             ║
║     │   │   ├── calib.txt                                            ║
║     │   │   └── times.txt                                            ║
║     │   └── ...                                                      ║
║     └── poses/                                                       ║
║         ├── 00.txt         ← ground-truth 4×4 poses                 ║
║         └── ...                                                      ║
║                                                                      ║
║  5. Then run:                                                        ║
║       python scripts/run_vo.py --sequence 00 --data data/kitti      ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download KITTI Odometry dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dest", default="data/kitti", help="Directory to place the dataset (default: data/kitti)"
    )
    parser.add_argument(
        "--sequences", nargs="+", default=["00"],
        help="Which sequence IDs to verify (default: 00)",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Only check if sequences exist without downloading",
    )
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    print("\n🛰  KITTI Odometry Dataset Manager")
    print("=" * 60)

    all_present = True
    for seq_id in args.sequences:
        sid = f"{int(seq_id):02d}"
        exists = check_sequence_exists(dest, sid)
        status = "✓ FOUND" if exists else "✗ MISSING"
        print(f"  Sequence {sid}: {status}")
        if not exists:
            all_present = False

    if all_present:
        print("\n✅ All requested sequences are present.")
        print(f"   Dataset root: {dest.resolve()}")
        return

    if args.check_only:
        print("\n❌ Some sequences are missing. Run without --check-only to download.")
        sys.exit(1)

    # Try automatic download
    print("\n⚡ Attempting automatic download from KITTI server...")
    print("   (This requires the dataset files to be accessible without login)")
    print("   If download fails, follow the manual instructions below.\n")

    try:
        # Download calibration (small, always needed)
        calib_zip = dest / "data_odometry_calib.zip"
        if not calib_zip.exists():
            download_file(
                f"{KITTI_BASE_URL}/data_odometry_calib.zip",
                calib_zip,
                "Calibration files (~1 MB)",
            )
            extract_zip(calib_zip, dest)

        # Download poses (small, always needed)
        poses_zip = dest / "data_odometry_poses.zip"
        if not poses_zip.exists():
            download_file(
                f"{KITTI_BASE_URL}/data_odometry_poses.zip",
                poses_zip,
                "Ground truth poses (~4 MB)",
            )
            extract_zip(poses_zip, dest)

        # Download greyscale images (large)
        images_zip = dest / "data_odometry_gray.zip"
        if not images_zip.exists():
            print("\n⚠️  The full greyscale image archive is ~22 GB.")
            ans = input("   Download now? [y/N] ").strip().lower()
            if ans == "y":
                download_file(
                    f"{KITTI_BASE_URL}/data_odometry_gray.zip",
                    images_zip,
                    "Greyscale images (~22 GB)",
                )
                extract_zip(images_zip, dest)
            else:
                print("   Skipping image download.")
                print_manual_instructions()
                sys.exit(0)

        print("\n✅ Download complete! Dataset layout:")
        print(f"   {dest.resolve()}/")
        print("   ├── sequences/00/image_0/*.png")
        print("   └── poses/00.txt")

    except Exception as exc:
        print(f"\n❌ Automatic download failed: {exc}")
        print("\n   KITTI requires registration. Please follow these instructions:")
        print_manual_instructions()
        sys.exit(1)


if __name__ == "__main__":
    main()

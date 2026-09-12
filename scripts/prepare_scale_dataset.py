#!/usr/bin/env python3
"""
Prepare ScaleNet Training Dataset from KITTI Ground Truth
==========================================================

Extract (frame_i, frame_i+1, ground_truth_scale) triplets from KITTI sequences.

Ground Truth Scale Computation
-------------------------------
For consecutive frames at indices i and i+1:

    scale = ||t_{i+1} - t_i||  (Euclidean distance between camera positions)

where t_i, t_{i+1} are the translation vectors from ground truth poses.

Dataset Structure
-----------------
Output: JSON Lines format (.jsonl)

Each line contains:
    {
        "seq_id": "01",
        "frame_idx": 42,
        "frame_curr": "data/kitti/sequences/01/image_0/000042.png",
        "frame_next": "data/kitti/sequences/01/image_0/000043.png",
        "scale": 1.234,
        "pose_curr": [12 floats],  # 3x4 matrix flattened
        "pose_next": [12 floats]
    }

Train/Val/Test Split
--------------------
Sequences 01, 02, 05 → train (80%)
Sequence 03 → validation (10%)
Sequences 06, 08 → test (10%)

Usage
-----
    python scripts/prepare_scale_dataset.py \\
        --data data/kitti \\
        --output data/scale_dataset.jsonl \\
        --sequences 01 02 03 05 06 08
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from vo.data.kitti_loader import KITTISequence


def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare ScaleNet training dataset from KITTI ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--data", "-d", default="data/kitti",
        help="Path to KITTI root directory (default: data/kitti)",
    )
    p.add_argument(
        "--sequences", "-s", nargs="+", default=["01", "02", "03", "05", "06", "08"],
        help="Space-separated list of KITTI sequence IDs (default: 01 02 03 05 06 08)",
    )
    p.add_argument(
        "--output", "-o", default="data/scale_dataset.jsonl",
        help="Output .jsonl file path (default: data/scale_dataset.jsonl)",
    )
    p.add_argument(
        "--min_scale", type=float, default=0.01,
        help="Skip frames with scale < this (stationary camera, default: 0.01)",
    )
    p.add_argument(
        "--log_level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def extract_scale_labels(
    data_root: Path,
    seq_id: str,
    min_scale: float,
    log: logging.Logger,
) -> List[dict]:
    """
    Extract (frame_i, frame_i+1, scale) triplets from one KITTI sequence.

    Args:
        data_root: Path to KITTI root directory
        seq_id: Sequence ID (e.g. "01")
        min_scale: Skip frames with scale below this threshold
        log: Logger

    Returns:
        List of dicts, each containing frame paths and ground truth scale
    """
    log.info(f"Processing sequence {seq_id} ...")

    try:
        seq = KITTISequence(data_root, seq_id)
    except FileNotFoundError as exc:
        log.warning(f"Skipping {seq_id}: {exc}")
        return []

    gt_poses = seq.ground_truth()
    if not gt_poses:
        log.warning(f"Sequence {seq_id}: no ground truth poses available")
        return []

    dataset = []
    n_total = len(gt_poses) - 1
    n_skipped_stationary = 0
    n_skipped_missing = 0

    for i in range(len(gt_poses) - 1):
        T_curr = gt_poses[i]
        T_next = gt_poses[i + 1]

        # Compute translation magnitude (ground truth scale)
        t_curr = T_curr[:3, 3]
        t_next = T_next[:3, 3]
        scale = float(np.linalg.norm(t_next - t_curr))

        # Skip near-zero motion (camera stationary / stopped at traffic light)
        if scale < min_scale:
            n_skipped_stationary += 1
            continue

        # Frame paths
        frame_curr_path = data_root / "sequences" / seq_id / "image_0" / f"{i:06d}.png"
        frame_next_path = data_root / "sequences" / seq_id / "image_0" / f"{i+1:06d}.png"

        # Verify files exist
        if not frame_curr_path.exists() or not frame_next_path.exists():
            n_skipped_missing += 1
            continue

        # Store as dict
        entry = {
            "seq_id": seq_id,
            "frame_idx": i,
            "frame_curr": str(frame_curr_path.relative_to(data_root.parent)),
            "frame_next": str(frame_next_path.relative_to(data_root.parent)),
            "scale": scale,
            "pose_curr": T_curr[:3, :].flatten().tolist(),  # 12 floats (3x4 matrix)
            "pose_next": T_next[:3, :].flatten().tolist(),
        }
        dataset.append(entry)

    log.info(
        f"  Sequence {seq_id}: {len(dataset)} samples extracted "
        f"({n_skipped_stationary} stationary, {n_skipped_missing} missing)"
    )

    return dataset


def compute_statistics(dataset: List[dict], log: logging.Logger) -> None:
    """Compute and log dataset statistics."""
    if not dataset:
        log.warning("Empty dataset, no statistics to compute")
        return

    scales = np.array([entry["scale"] for entry in dataset])

    log.info("=" * 60)
    log.info("Dataset Statistics")
    log.info(f"  Total samples: {len(dataset):,}")
    log.info(f"  Scale distribution:")
    log.info(f"    min    : {scales.min():.4f} m")
    log.info(f"    p1     : {np.percentile(scales, 1):.4f} m")
    log.info(f"    median : {np.median(scales):.4f} m")
    log.info(f"    mean   : {scales.mean():.4f} m")
    log.info(f"    p99    : {np.percentile(scales, 99):.4f} m")
    log.info(f"    max    : {scales.max():.4f} m")
    log.info(f"  Sequences represented:")
    seq_counts = {}
    for entry in dataset:
        seq_counts[entry["seq_id"]] = seq_counts.get(entry["seq_id"], 0) + 1
    for seq_id, count in sorted(seq_counts.items()):
        log.info(f"    {seq_id}: {count:5d} samples")
    log.info("=" * 60)


def main():
    args = parse_args()
    setup_logging(args.log_level)
    log = logging.getLogger("prepare_scale_dataset")

    data_root = Path(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("ScaleNet Dataset Preparation")
    log.info(f"  Data root  : {data_root}")
    log.info(f"  Sequences  : {args.sequences}")
    log.info(f"  Output     : {output_path}")
    log.info(f"  Min scale  : {args.min_scale} m")
    log.info("=" * 60)

    # Extract data from all sequences
    all_data = []
    for seq_id in args.sequences:
        seq_id = f"{int(seq_id):02d}"
        seq_data = extract_scale_labels(data_root, seq_id, args.min_scale, log)
        all_data.extend(seq_data)

    if not all_data:
        log.error("No data extracted. Check your --data path and --sequences.")
        sys.exit(1)

    # Compute statistics
    compute_statistics(all_data, log)

    # Write to JSON Lines format
    log.info(f"Writing {len(all_data)} samples to {output_path} ...")
    with open(output_path, "w") as f:
        for entry in all_data:
            f.write(json.dumps(entry) + "\n")

    log.info("✅ Done. Dataset preparation complete.")
    log.info(f"   Next step: python scripts/train_scale_net.py --dataset {output_path}")


if __name__ == "__main__":
    main()

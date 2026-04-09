#!/usr/bin/env python3
"""
Train and Save a BoW Vocabulary from KITTI Sequences
======================================================

Usage
-----
Train on sequences 00–10 (all KITTI training sequences):

    python scripts/train_bow_vocab.py \\
        --data data/kitti \\
        --sequences 00 01 02 03 04 05 06 07 08 09 10 \\
        --output data/bow_vocab.npz \\
        --n_words 1000 \\
        --max_frames 200

Train on a single sequence quickly for testing:

    python scripts/train_bow_vocab.py --data data/kitti --sequences 00 \\
        --n_words 500 --max_frames 100 --output data/bow_vocab_small.npz

Then pass the vocabulary to the VO pipeline at runtime:

    python scripts/run_vo.py --sequence 00 --vocab data/bow_vocab.npz

Theory
------
An offline-trained vocabulary generalises far better than one trained
at runtime on a single sequence because:

  1. The training set covers diverse appearance conditions (weather,
     time of day, urban vs rural, highways vs intersections).
  2. The TF-IDF inverse document frequencies are calibrated against a
     large corpus, giving meaningful similarity scores.
  3. The LoopClosureDetector does not incur a cold-start phase — it
     can detect loops from the very first keyframe.

K-means++ initialisation (cv2.KMEANS_PP_CENTERS) is used for quality
cluster centres, run with multiple attempts to avoid local minima.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from vo.data.kitti_loader import KITTISequence
from vo.features import FeatureFrontend
from vo.loop_closure import BagOfWords


def parse_args():
    p = argparse.ArgumentParser(
        description="Train a BoW vocabulary from KITTI sequences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--data", "-d", default="data/kitti",
        help="Path to KITTI root directory (default: data/kitti)",
    )
    p.add_argument(
        "--sequences", "-s", nargs="+", default=["00"],
        help="Space-separated list of KITTI sequence IDs (default: 00)",
    )
    p.add_argument(
        "--output", "-o", default="data/bow_vocab.npz",
        help="Output .npz file path (default: data/bow_vocab.npz)",
    )
    p.add_argument(
        "--n_words", "-k", type=int, default=1000,
        help="Vocabulary size K (number of k-means cluster centres, default: 1000)",
    )
    p.add_argument(
        "--max_frames", "-n", type=int, default=None,
        help="Max frames per sequence (None = all, use small number for quick test)",
    )
    p.add_argument(
        "--max_features", "-f", type=int, default=500,
        help="Max ORB features per frame to extract (default: 500)",
    )
    p.add_argument(
        "--stride", type=int, default=5,
        help="Process every N-th frame (default: 5, reduces redundancy)",
    )
    p.add_argument(
        "--kmeans_attempts", type=int, default=5,
        help="K-means restarts for quality centres (default: 5)",
    )
    p.add_argument(
        "--log_level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def collect_descriptors(
    data_root: Path,
    seq_ids: list[str],
    max_frames: int | None,
    n_features: int,
    stride: int,
    log: logging.Logger,
) -> np.ndarray:
    """
    Extract ORB descriptors from KITTI sequences.

    Parameters
    ----------
    data_root : Path
    seq_ids : list of sequence IDs
    max_frames : int | None
    n_features : int — ORB features per frame
    stride : int — process every N-th frame
    log : Logger

    Returns
    -------
    np.ndarray, shape (M, 32), dtype uint8
        All extracted descriptors concatenated.
    """
    frontend = FeatureFrontend(n_features=n_features)
    all_descriptors: list[np.ndarray] = []

    for seq_id in seq_ids:
        seq_id = f"{int(seq_id):02d}"
        log.info("Processing sequence %s ...", seq_id)
        try:
            seq = KITTISequence(data_root, seq_id, max_frames=max_frames)
        except FileNotFoundError as exc:
            log.warning("Skipping %s: %s", seq_id, exc)
            continue

        n_frames = len(seq)
        n_descriptors_before = sum(len(d) for d in all_descriptors)

        for frame_idx, (fid, image, _) in enumerate(seq):
            if frame_idx % stride != 0:
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            kps, descs = frontend.detect(gray)

            if descs is not None and len(descs) > 0:
                all_descriptors.append(descs)

        n_added = sum(len(d) for d in all_descriptors) - n_descriptors_before
        log.info(
            "  Sequence %s: %d frames → %d descriptors",
            seq_id, n_frames // stride + 1, n_added,
        )

    if not all_descriptors:
        raise RuntimeError(
            "No descriptors collected. Check your --data path and --sequences."
        )

    combined = np.vstack(all_descriptors)
    log.info("Total descriptors collected: %d", len(combined))
    return combined


def train_vocabulary(
    descriptors: np.ndarray,
    n_words: int,
    n_attempts: int,
    log: logging.Logger,
) -> np.ndarray:
    """
    Run K-means++ on the descriptor pool to produce cluster centres.

    Parameters
    ----------
    descriptors : (M, 32) uint8
    n_words : int — vocabulary size K
    n_attempts : int — number of random restarts

    Returns
    -------
    centres : np.ndarray, shape (K, 32), dtype float32
    """
    log.info(
        "Running K-means (K=%d, attempts=%d) on %d descriptors ...",
        n_words, n_attempts, len(descriptors),
    )

    # cv2.kmeans needs float32
    data_f32 = descriptors.astype(np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        100,   # max iterations
        1.0,   # epsilon
    )

    compactness, labels, centres = cv2.kmeans(
        data_f32,
        n_words,
        None,
        criteria,
        attempts=n_attempts,
        flags=cv2.KMEANS_PP_CENTERS,
    )

    log.info(
        "K-means converged. Compactness: %.2e  |  Centres: %s",
        compactness, centres.shape,
    )
    return centres


def main():
    args = parse_args()
    setup_logging(args.log_level)
    log = logging.getLogger("train_bow_vocab")

    data_root = Path(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("BoW Vocabulary Training")
    log.info("  Data root  : %s", data_root)
    log.info("  Sequences  : %s", args.sequences)
    log.info("  Vocab size : %d words", args.n_words)
    log.info("  Output     : %s", output_path)
    log.info("=" * 60)

    # Step 1: collect descriptors
    descriptors = collect_descriptors(
        data_root=data_root,
        seq_ids=args.sequences,
        max_frames=args.max_frames,
        n_features=args.max_features,
        stride=args.stride,
        log=log,
    )

    # Step 2: train vocabulary
    centres = train_vocabulary(
        descriptors=descriptors,
        n_words=args.n_words,
        n_attempts=args.kmeans_attempts,
        log=log,
    )

    # Step 3: save via BagOfWords.save()
    bow = BagOfWords(n_words=args.n_words)
    bow._vocabulary = centres
    bow.save(str(output_path))

    log.info("✅ Done. Vocabulary saved to: %s", output_path.resolve())
    log.info(
        "   Use with: python scripts/run_vo.py --vocab %s ...",
        output_path,
    )


if __name__ == "__main__":
    main()

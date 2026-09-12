#!/usr/bin/env python3
"""
Train ScaleNet: Learned Metric Scale Recovery
==============================================

This script trains a CNN to predict metric scale from consecutive KITTI frames,
replacing the ground-plane RANSAC heuristic with a learned model.

Dataset Format
--------------
Input: data/scale_dataset.jsonl — one JSON object per line, containing:
  - frame_curr, frame_next: image paths
  - scale: ground truth metric displacement (meters)
  - pose_curr, pose_next: ground truth poses

Loss Function
-------------
Negative log-likelihood with learned uncertainty:

    L = precision * (scale_pred - scale_gt)^2 + log_var

where precision = exp(-log_var), allowing the model to learn its own confidence.

Training Strategy
-----------------
- Optimizer: Adam (lr=1e-3)
- Batch size: 16 (CPU training on 14K samples → ~1 hour)
- Validation split: 10% of data
- Early stopping: stop if val loss plateaus for 3 epochs
- Learning rate schedule: decay by 0.5 if val loss doesn't improve

Usage
-----
    python scripts/train_scale_net.py \\
        --dataset data/scale_dataset.jsonl \\
        --output models/scale_net_v1.pth \\
        --epochs 15 \\
        --batch_size 16 \\
        --lr 1e-3

Usage (quick test)
------------------
    python scripts/train_scale_net.py \\
        --dataset data/scale_dataset.jsonl \\
        --output models/scale_net_test.pth \\
        --epochs 2 \\
        --batch_size 32 \\
        --max_samples 1000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from vo.scale_net import ScaleNet


logger = logging.getLogger(__name__)


class ScaleDataset(Dataset):
    """PyTorch dataset for scale regression from KITTI frame pairs."""

    def __init__(
        self,
        jsonl_path: Path | str,
        max_samples: int | None = None,
        cache_frames: bool = False,
    ):
        """
        Args:
            jsonl_path: Path to .jsonl file with {frame_curr, frame_next, scale} entries
            max_samples: Limit dataset size (for quick testing)
            cache_frames: Cache loaded frames in memory (only for small datasets)
        """
        self.jsonl_path = Path(jsonl_path)
        self.cache_frames = cache_frames
        self.frame_cache = {}

        # Load metadata
        self.entries = []
        with open(self.jsonl_path) as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                entry = json.loads(line)
                self.entries.append(entry)

        logger.info(f"Loaded {len(self.entries)} samples from {self.jsonl_path}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: (4, H, W) tensor with [prev_gray, curr_gray, flow_x, flow_y]
            scale: (1,) tensor with ground truth scale
        """
        # Skip missing entries - find next valid index
        while True:
            entry = self.entries[idx]

            # Load frames (from disk or cache)
            frame_curr_path = Path(entry["frame_curr"])
            frame_next_path = Path(entry["frame_next"])

            if self.cache_frames and idx in self.frame_cache:
                frame_curr, frame_next = self.frame_cache[idx]
            else:
                frame_curr = cv2.imread(str(frame_curr_path), cv2.IMREAD_GRAYSCALE)
                frame_next = cv2.imread(str(frame_next_path), cv2.IMREAD_GRAYSCALE)

                if frame_curr is None or frame_next is None:
                    # Frame missing, skip this entry and try next
                    idx = (idx + 1) % len(self)
                    logger.warning(f"Skipping missing frame at index {idx}")
                    continue

                if self.cache_frames:
                    self.frame_cache[idx] = (frame_curr, frame_next)

            break

        # Prepare input: resize FIRST, then compute flow
        # MEMORY FIX: Farneback optical flow allocates large pyramid buffers.
        # Computing it on full-resolution KITTI frames (376x1241) exhausted
        # system memory. The network only consumes the resized flow anyway,
        # so compute flow at target resolution (~4x less memory, much faster).
        target_size = (640, 192)  # (width, height) for cv2.resize
        frame_curr = cv2.resize(frame_curr, target_size)
        frame_next = cv2.resize(frame_next, target_size)

        flow = cv2.calcOpticalFlowFarneback(
            frame_curr, frame_next,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        frame_curr = frame_curr.astype(np.float32) / 255.0
        frame_next = frame_next.astype(np.float32) / 255.0

        # Normalize flow by global statistics (not per-sample max!)
        flow_x = flow[:, :, 0] / 20.0  # std ≈ 20px
        flow_y = flow[:, :, 1] / 20.0

        # Stack into 4-channel input
        x = np.stack([frame_curr, frame_next, flow_x, flow_y], axis=0)
        x = torch.from_numpy(x).float()

        # Ground truth scale
        scale = torch.tensor([entry["scale"]], dtype=torch.float32)

        return x, scale


def parse_args():
    p = argparse.ArgumentParser(
        description="Train ScaleNet for metric scale prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset", "-d", default="data/scale_dataset.jsonl",
        help="Path to .jsonl dataset (default: data/scale_dataset.jsonl)",
    )
    p.add_argument(
        "--output", "-o", default="models/scale_net_v1.pth",
        help="Output model path (default: models/scale_net_v1.pth)",
    )
    p.add_argument(
        "--epochs", "-e", type=int, default=15,
        help="Number of training epochs (default: 15)",
    )
    p.add_argument(
        "--batch_size", "-b", type=int, default=16,
        help="Batch size (default: 16, CPU → use 16-32)",
    )
    p.add_argument(
        "--lr", type=float, default=1e-3,
        help="Initial learning rate (default: 1e-3)",
    )
    p.add_argument(
        "--val_split", type=float, default=0.1,
        help="Validation set fraction (default: 0.1)",
    )
    p.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit dataset (for quick testing, default: None = all)",
    )
    p.add_argument(
        "--log_level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    p.add_argument(
        "--device", default="cpu",
        choices=["cpu", "cuda"],
        help="Device to train on (default: cpu)",
    )
    return p.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0

    for x, scale_gt in loader:
        x = x.to(device)
        scale_gt = scale_gt.to(device)

        # Forward
        scale_pred, log_var = model(x)

        # Loss: negative log-likelihood with uncertainty
        precision = torch.exp(-log_var)
        loss = torch.mean(precision * (scale_pred - scale_gt) ** 2 + log_var)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate model, return (loss, scale_mae)."""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0

    with torch.no_grad():
        for x, scale_gt in loader:
            x = x.to(device)
            scale_gt = scale_gt.to(device)

            scale_pred, log_var = model(x)

            precision = torch.exp(-log_var)
            loss = torch.mean(precision * (scale_pred - scale_gt) ** 2 + log_var)
            mae = torch.mean(torch.abs(scale_pred - scale_gt))

            total_loss += loss.item()
            total_mae += mae.item()

    return total_loss / len(loader), total_mae / len(loader)


def main():
    args = parse_args()
    setup_logging(args.log_level)
    log = logging.getLogger("train_scale_net")

    device = torch.device(args.device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("ScaleNet Training")
    log.info(f"  Dataset     : {args.dataset}")
    log.info(f"  Output      : {output_path}")
    log.info(f"  Epochs      : {args.epochs}")
    log.info(f"  Batch size  : {args.batch_size}")
    log.info(f"  Learning rate: {args.lr}")
    log.info(f"  Device      : {device}")
    log.info("=" * 70)

    # Load dataset
    log.info("Loading dataset ...")
    dataset = ScaleDataset(args.dataset, max_samples=args.max_samples)

    # Train/val split
    n_val = max(1, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    log.info(f"  Train: {len(train_set)} samples")
    log.info(f"  Val  : {len(val_set)} samples")

    # Create model
    log.info("Creating model ...")
    model = ScaleNet(input_channels=4, device=str(device)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"  Parameters: {n_params:,} ({n_params / 1e6:.2f}M)")

    # Optimizer + LR scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # Training loop
    log.info("=" * 70)
    best_val_loss = float("inf")
    patience_counter = 0

    t0 = time.time()
    for epoch in range(args.epochs):
        t_epoch = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_mae = validate(model, val_loader, device)

        dt = time.time() - t_epoch

        log.info(
            f"Epoch {epoch + 1:2d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_mae={val_mae:.4f}m | "
            f"{dt:.1f}s"
        )

        # Early stopping + checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_path)
            log.info(f"  ✓ Checkpoint saved to {output_path}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Learning rate schedule
        scheduler.step(val_loss)

        # Early stopping
        if patience_counter >= 5:
            log.warning(f"Early stopping after {epoch + 1} epochs (no improvement for 5 epochs)")
            break

    total_time = time.time() - t0
    log.info("=" * 70)
    log.info(f"Training complete! Total time: {total_time:.1f}s ({total_time / 60:.1f}m)")
    log.info(f"Best validation loss: {best_val_loss:.4f}")
    log.info(f"Model saved to: {output_path.resolve()}")
    log.info("=" * 70)

    # Final validation on best model
    log.info("Loading best model for final evaluation ...")
    model.load_state_dict(torch.load(output_path, map_location=device))
    val_loss, val_mae = validate(model, val_loader, device)
    log.info(f"Final validation MAE: {val_mae:.4f}m")

    log.info("\n✅ ScaleNet training complete!")
    log.info(f"   Next step: python scripts/run_vo.py --use_learned_scale --scale_model {output_path}")


if __name__ == "__main__":
    main()

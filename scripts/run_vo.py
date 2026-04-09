#!/usr/bin/env python3
"""
Run the Monocular Visual Odometry Pipeline
===========================================

Usage:
    python scripts/run_vo.py --sequence 00 --data data/kitti
    python scripts/run_vo.py --sequence 00 --max_frames 200 --no_viewer
    python scripts/run_vo.py --help
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vo.camera import Camera
from vo.data.kitti_loader import KITTISequence
from vo.odometry import VisualOdometry
from vo.evaluation.metrics import compute_ate, compute_rpe
from vo.visualization.trajectory_plot import plot_trajectory_2d, plot_trajectory_3d, plot_error_over_time


def parse_args():
    p = argparse.ArgumentParser(
        description="Monocular Visual Odometry on KITTI Odometry Sequences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--sequence", "-s", default="00", help="KITTI sequence ID (default: 00)")
    p.add_argument("--data", "-d", default="data/kitti", help="Path to KITTI root directory")
    p.add_argument("--max_frames", "-n", type=int, default=None, help="Limit frames (for testing)")
    p.add_argument("--output", "-o", default=None, help="Directory to save results (default: results/<seq>)")
    p.add_argument("--no_viewer", action="store_true", help="Disable Open3D real-time viewer")
    p.add_argument("--vocab", default=None, help="Path to pre-trained BoW vocabulary .npz (recommended)")
    p.add_argument("--kf_parallax", type=float, default=2.0, help="Keyframe parallax threshold in pixels/deg (default: 2.0)")
    p.add_argument("--kf_track_ratio", type=float, default=0.80, help="Keyframe tracked-feature ratio floor (default: 0.80)")
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose per-frame logging")
    p.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    args = parse_args()
    setup_logging(args.log_level)
    log = logging.getLogger("run_vo")

    # ── Load Sequence ────────────────────────────────────────────────────────
    seq_id = f"{int(args.sequence):02d}"
    data_root = Path(args.data)
    output_dir = Path(args.output) if args.output else Path("results") / seq_id
    output_dir.mkdir(parents=True, exist_ok=True)
    traj_path = str(output_dir / "trajectory.txt")

    log.info("=" * 60)
    log.info("Monocular Visual Odometry — Sequence %s", seq_id)
    log.info("Data root : %s", data_root)
    log.info("Output    : %s", output_dir)
    log.info("=" * 60)

    try:
        sequence = KITTISequence(data_root, seq_id, max_frames=args.max_frames)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        log.error("Run  python scripts/download_kitti.py  to download the dataset.")
        sys.exit(1)

    camera = Camera.from_kitti_sequence(sequence)
    log.info("Camera: %s", camera)
    log.info("Sequence length: %d frames", len(sequence))

    # ── Build Pipeline ───────────────────────────────────────────────────────
    vo = VisualOdometry(
        camera=camera,
        camera_height=sequence.camera_height(),
        kf_parallax_deg=args.kf_parallax,
        kf_track_ratio=args.kf_track_ratio,
        vocab_path=args.vocab,
        verbose=args.verbose,
    )

    # ── Optional Real-time Viewer ────────────────────────────────────────────
    viewer = None
    if not args.no_viewer:
        try:
            from vo.visualization.realtime_viewer import RealtimeViewer
            viewer = RealtimeViewer()
            viewer.start()
            log.info("Open3D viewer started")
        except Exception as exc:
            log.warning("Could not start viewer: %s — running headless", exc)
            viewer = None

    # ── Run Pipeline ─────────────────────────────────────────────────────────
    gt_poses = sequence.ground_truth()

    for frame_id, image, timestamp in sequence:
        vo.process_frame(image, frame_id)

        if viewer is not None:
            viewer.update(
                trajectory=vo.trajectory,
                point_cloud=vo.local_map.points,
                gt_poses=gt_poses[:frame_id + 1] if gt_poses else None,
            )
            if not viewer.tick():
                log.info("Viewer closed by user — stopping.")
                break

    # ── Save Trajectory ──────────────────────────────────────────────────────
    from vo.evaluation.metrics import save_trajectory_kitti
    save_trajectory_kitti(vo.trajectory, traj_path)
    log.info("Trajectory saved: %s  (%d poses)", traj_path, len(vo.trajectory))

    # ── Evaluate ─────────────────────────────────────────────────────────────
    if gt_poses:
        n = min(len(vo.trajectory), len(gt_poses))
        est = vo.trajectory[:n]
        gt  = gt_poses[:n]

        ate_rmse, ate_mean = compute_ate(est, gt, align=True, with_scale=True)
        rpe_rmse, rpe_mean = compute_rpe(est, gt, delta=1)

        log.info("─" * 40)
        log.info("EVALUATION RESULTS (Sequence %s, %d frames)", seq_id, n)
        log.info("  ATE RMSE : %.3f m  (mean %.3f m)", ate_rmse, ate_mean)
        log.info("  RPE RMSE : %.3f m  (mean %.3f m)", rpe_rmse, rpe_mean)
        log.info("─" * 40)

        # Save metric summary
        with open(output_dir / "metrics.txt", "w") as f:
            f.write(f"Sequence:  {seq_id}\n")
            f.write(f"Frames:    {n}\n")
            f.write(f"ATE RMSE:  {ate_rmse:.4f} m\n")
            f.write(f"ATE Mean:  {ate_mean:.4f} m\n")
            f.write(f"RPE RMSE:  {rpe_rmse:.4f} m\n")
            f.write(f"RPE Mean:  {rpe_mean:.4f} m\n")
    else:
        gt = None
        log.warning("No ground-truth poses found — skipping evaluation")

    # ── Plots ────────────────────────────────────────────────────────────────
    log.info("Generating trajectory plots ...")
    plot_trajectory_2d(
        vo.trajectory, gt,
        title=f"KITTI Sequence {seq_id} — Top View",
        save_path=str(output_dir / "trajectory_2d.png"),
    )
    plot_trajectory_3d(
        vo.trajectory, gt,
        title=f"KITTI Sequence {seq_id} — 3D View",
        save_path=str(output_dir / "trajectory_3d.png"),
    )

    # Per-frame ATE if GT available
    if gt_poses:
        from vo.evaluation.metrics import _extract_positions, umeyama_alignment
        est_pos = _extract_positions(vo.trajectory[:n])
        gt_pos  = _extract_positions(gt_poses[:n])
        R, t, s = umeyama_alignment(est_pos, gt_pos)
        aligned = (s * R @ est_pos.T).T + t
        per_frame_ate = list(np.linalg.norm(gt_pos - aligned, axis=1))
        plot_error_over_time(
            per_frame_ate,
            title=f"ATE per Frame — Sequence {seq_id}",
            save_path=str(output_dir / "ate_per_frame.png"),
        )

    if viewer is not None:
        viewer.save_screenshot(str(output_dir / "viewer_screenshot.png"))
        viewer.close()

    log.info("✅ Done. Results in: %s", output_dir.resolve())


if __name__ == "__main__":
    main()

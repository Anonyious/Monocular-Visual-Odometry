#!/usr/bin/env python3
"""
Ablation Study: RANSAC Ground-Plane Scale vs. Learned Scale (ScaleNet)
=======================================================================

Runs both scale recovery methods on all available KITTI sequences and
produces a comparison table with ATE/RPE metrics.

Usage:
    python scripts/ablation_study.py --data data/kitti --output results/ablation
    python scripts/ablation_study.py --sequences 00 05 --quick
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from vo.camera import Camera
from vo.data.kitti_loader import KITTISequence
from vo.odometry import VisualOdometry
from vo.evaluation.metrics import compute_ate, compute_rpe, _extract_positions


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("ablation")


def run_sequence(
    data_root: Path,
    seq_id: str,
    variant: str,
    max_frames: Optional[int] = None,
    use_learned_scale: bool = False,
    scale_model_path: str = "models/scale_net_v1.pth",
) -> Optional[dict]:
    """Run VO pipeline on one sequence with given configuration."""
    seq_id = f"{int(seq_id):02d}"
    output_dir = Path("results") / "ablation" / variant / seq_id
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        seq = KITTISequence(data_root, seq_id, max_frames=max_frames)
        camera = Camera.from_kitti_sequence(seq)
        gt_poses = seq.ground_truth()

        log.info(f"  [{variant:15s}] Seq {seq_id}: {len(seq)} frames")

        kwargs = dict(
            camera=camera,
            camera_height=seq.camera_height(),
            verbose=False,
        )
        if use_learned_scale:
            kwargs["use_learned_scale"] = True
            kwargs["scale_model_path"] = scale_model_path

        vo = VisualOdometry(**kwargs)

        t0 = time.perf_counter()
        for frame_id, image, timestamp in seq:
            vo.process_frame(image, frame_id)
        elapsed = time.perf_counter() - t0

        if not gt_poses:
            log.warning(f"  No ground truth for {seq_id}")
            return None

        n = min(len(vo.trajectory), len(gt_poses))
        est = vo.trajectory[:n]
        gt = gt_poses[:n]

        ate_rmse, ate_mean = compute_ate(est, gt, align=True, with_scale=True)
        rpe_rmse, rpe_mean = compute_rpe(est, gt, delta=1)

        # Scale drift: ratio of estimated total distance to GT total distance
        est_pos = _extract_positions(est)
        gt_pos = _extract_positions(gt)
        est_dist = np.linalg.norm(est_pos[-1] - est_pos[0]) if len(est_pos) > 1 else 0
        gt_dist = np.linalg.norm(gt_pos[-1] - gt_pos[0]) if len(gt_pos) > 1 else 0
        scale_drift = abs(est_dist / gt_dist - 1.0) if gt_dist > 0 else float("nan")

        fps = n / elapsed if elapsed > 0 else 0

        result = {
            "sequence": seq_id,
            "variant": variant,
            "n_frames": n,
            "n_keyframes": vo._n_keyframes,
            "ate_rmse": float(ate_rmse),
            "ate_mean": float(ate_mean),
            "rpe_rmse": float(rpe_rmse),
            "rpe_mean": float(rpe_mean),
            "fps": float(fps),
            "time_s": float(elapsed),
            "map_size": len(vo.local_map),
            "loop_closures": vo.pose_graph.n_loop_closures,
            "scale_drift": float(scale_drift),
        }

        log.info(
            f"    ATE={ate_rmse:.3f}m  RPE={rpe_rmse:.3f}m  "
            f"FPS={fps:.1f}  Loops={result['loop_closures']}  Drift={scale_drift:.1%}"
        )

        # Save trajectory
        from vo.evaluation.metrics import save_trajectory_kitti
        save_trajectory_kitti(vo.trajectory, str(output_dir / "trajectory.txt"))

        return result

    except Exception as exc:
        log.error(f"  Error on seq {seq_id} ({variant}): {exc}")
        import traceback
        traceback.print_exc()
        return None


def print_comparison_table(results: List[dict]) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 110)
    print("ABALATION STUDY: RANSAC Ground-Plane Scale vs. Learned Scale (ScaleNet)")
    print("=" * 110)

    # Group by sequence
    seqs = sorted(set(r["sequence"] for r in results))
    variants = sorted(set(r["variant"] for r in results))

    # Header
    print(f"\n{'Seq':>4} {'Frames':>7} {'Method':>15} {'ATE RMSE':>10} {'ATE Mean':>10} "
          f"{'RPE RMSE':>10} {'RPE Mean':>10} {'Drift':>8} {'Loops':>6} {'FPS':>7}")
    print("-" * 110)

    for seq in seqs:
        seq_results = [r for r in results if r["sequence"] == seq]
        first = True
        for r in seq_results:
            marker = " ←" if r["variant"] == "baseline" and first else ""
            print(
                f"{r['sequence']:>4} {r['n_frames']:>7} {r['variant']:>15} "
                f"{r['ate_rmse']:>9.3f}m {r['ate_mean']:>9.3f}m "
                f"{r['rpe_rmse']:>9.3f}m {r['rpe_mean']:>9.3f}m "
                f"{r['scale_drift']:>7.1%} {r['loop_closures']:>6} "
                f"{r['fps']:>6.1f}{marker}"
            )
            first = False
        print()

    # Summary table
    print("\n" + "=" * 110)
    print("SUMMARY (Mean across all sequences)")
    print("=" * 110)
    print(f"\n{'Metric':>15} {'RANSAC':>12} {'ScaleNet':>12} {'Improvement':>14}")
    print("-" * 55)

    metrics = ["ate_rmse", "ate_mean", "rpe_rmse", "rpe_mean", "scale_drift"]
    for m in metrics:
        baseline_vals = [r[m] for r in results if r["variant"] == "baseline"]
        learned_vals = [r[m] for r in results if r["variant"] == "learned"]
        if baseline_vals and learned_vals:
            b_mean = np.mean(baseline_vals)
            l_mean = np.mean(learned_vals)
            improvement = ((b_mean - l_mean) / b_mean) * 100
            unit = "%" if m == "scale_drift" else "m"
            imp_str = f"+{improvement:.1f}%{' ' * 6}" if improvement > 0 else f"{improvement:.1f}%{' ' * 6}"
            print(
                f"{m:>15} {b_mean:>11.3f}{unit} {l_mean:>11.3f}{unit} "
                f"{imp_str:>14}"
            )

    print("\n" + "=" * 110)


def generate_latex_table(results: List[dict]) -> str:
    """Generate LaTeX table for the paper."""
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Ablation Study: RANSAC Ground-Plane Scale vs. Learned Scale (ScaleNet) on KITTI Odometry Benchmark}")
    lines.append(r"  \label{tab:ablation}")
    lines.append(r"  \begin{tabular}{crrrrrrr}")
    lines.append(r"    \hline")
    lines.append(r"    \textbf{Seq} & \textbf{Frames} & \textbf{ATE RMSE (m)} & \textbf{RPE RMSE (m)} & "
                 r"\textbf{Scale Drift} & \textbf{Loops} & \textbf{FPS} \\")
    lines.append(r"    \hline")

    seqs = sorted(set(r["sequence"] for r in results))
    for seq in seqs:
        seq_results = {r["variant"]: r for r in results if r["sequence"] == seq}
        if "baseline" in seq_results:
            r = seq_results["baseline"]
            lines.append(
                f"    {seq} & {r['n_frames']} & {r['ate_rmse']:.3f} & {r['rpe_rmse']:.3f} & "
                f"{r['scale_drift']:.1%} & {r['loop_closures']} & {r['fps']:.1f} \\\\"
            )
        if "learned" in seq_results:
            r = seq_results["learned"]
            lines.append(
                f"    {seq} & {r['n_frames']} & {r['ate_rmse']:.3f} & {r['rpe_rmse']:.3f} & "
                f"{r['scale_drift']:.1%} & {r['loop_closures']} & {r['fps']:.1f} \\\\"
            )

    # Mean row
    baseline = [r for r in results if r["variant"] == "baseline"]
    learned = [r for r in results if r["variant"] == "learned"]
    if baseline and learned:
        lines.append(r"    \midrule")
        b_ate = np.mean([r["ate_rmse"] for r in baseline])
        b_rpe = np.mean([r["rpe_rmse"] for r in baseline])
        b_drift = np.mean([r["scale_drift"] for r in baseline])
        b_loops = np.mean([r["loop_closures"] for r in baseline])
        b_fps = np.mean([r["fps"] for r in baseline])
        l_ate = np.mean([r["ate_rmse"] for r in learned])
        l_rpe = np.mean([r["rpe_rmse"] for r in learned])
        l_drift = np.mean([r["scale_drift"] for r in learned])
        l_loops = np.mean([r["loop_closures"] for r in learned])
        l_fps = np.mean([r["fps"] for r in learned])
        lines.append(
            f"    \\textbf{{Mean}} & --- & "
            f"{b_ate:.3f} & {b_rpe:.3f} & "
            f"{b_drift:.1%} & {b_loops:.0f} & {b_fps:.1f} \\\\"
        )
        lines.append(
            f"    \\textbf{{Mean}} & --- & "
            f"{l_ate:.3f} & {l_rpe:.3f} & "
            f"{l_drift:.1%} & {l_loops:.0f} & {l_fps:.1f} \\\\"
        )
    lines.append(r"    \hline")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Ablation study: RANSAC vs. learned scale recovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", "-d", default="data/kitti", help="Path to KITTI root directory")
    parser.add_argument(
        "--sequences", "-s", nargs="+",
        default=["00", "01", "02", "03", "05", "06", "08"],
        help="KITTI sequences to evaluate (default: 00 01 02 03 05 06 08)",
    )
    parser.add_argument("--max_frames", "-n", type=int, default=None,
                        help="Limit frames per sequence (for quick testing)")
    parser.add_argument("--quick", action="store_true", help="Quick run (200 frames per sequence)")
    parser.add_argument("--scale_model", default="models/scale_net_v1.pth",
                        help="Path to trained ScaleNet checkpoint")
    args = parser.parse_args()

    if args.quick:
        args.max_frames = 200

    data_root = Path(args.data)
    if not data_root.exists():
        log.error(f"KITTI data not found at {data_root}")
        log.error("Run: python scripts/download_kitti.py")
        sys.exit(1)

    results = []

    # Run baseline (RANSAC) on all sequences
    log.info("=" * 60)
    log.info("Running BASELINE (RANSAC ground-plane scale)")
    log.info("=" * 60)
    for seq in args.sequences:
        r = run_sequence(data_root, seq, variant="baseline", max_frames=args.max_frames)
        if r:
            results.append(r)

    # Run learned scale on all sequences
    log.info("\n" + "=" * 60)
    log.info("Running LEARNED SCALE (ScaleNet)")
    log.info("=" * 60)
    for seq in args.sequences:
        r = run_sequence(
            data_root, seq, variant="learned", max_frames=args.max_frames,
            use_learned_scale=True, scale_model_path=args.scale_model,
        )
        if r:
            results.append(r)

    # Print comparison
    print_comparison_table(results)

    # Save results
    out_dir = Path("results") / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {out_dir / 'results.json'}")

    # LaTeX table
    latex = generate_latex_table(results)
    with open(out_dir / "latex_table.tex", "w") as f:
        f.write(latex)
    log.info(f"LaTeX table saved to {out_dir / 'latex_table.tex'}")

    log.info("\n✅ Ablation study complete!")


if __name__ == "__main__":
    main()

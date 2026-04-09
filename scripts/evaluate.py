#!/usr/bin/env python3
"""
Quantitative Evaluation Script
================================
Runs evo_ape and evo_rpe against KITTI ground truth and prints a
LaTeX-formatted results table.

Usage:
    python scripts/evaluate.py --sequence 00 --data data/kitti
    python scripts/evaluate.py --sequence 00 05 07 --latex
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vo.evaluation.metrics import compute_ate, compute_rpe, load_trajectory_kitti


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate VO trajectory against KITTI ground truth")
    p.add_argument("--sequence", "-s", nargs="+", default=["00"])
    p.add_argument("--data", "-d", default="data/kitti")
    p.add_argument("--results", "-r", default="results")
    p.add_argument("--latex", action="store_true", help="Print LaTeX table")
    p.add_argument("--evo", action="store_true", help="Also run evo CLI tools")
    return p.parse_args()


def evaluate_sequence(seq_id: str, data_root: Path, results_dir: Path) -> dict:
    """Compute ATE and RPE for a single sequence."""
    gt_file  = data_root / "poses" / f"{seq_id}.txt"
    est_file = results_dir / seq_id / "trajectory.txt"

    if not gt_file.exists():
        print(f"  [SKIP] Ground truth not found: {gt_file}")
        return {}
    if not est_file.exists():
        print(f"  [SKIP] Estimated trajectory not found: {est_file}")
        print(f"         Run:  python scripts/run_vo.py --sequence {seq_id}")
        return {}

    gt  = load_trajectory_kitti(str(gt_file))
    est = load_trajectory_kitti(str(est_file))

    n = min(len(gt), len(est))
    gt  = gt[:n]
    est = est[:n]

    ate_rmse, ate_mean = compute_ate(est, gt, align=True, with_scale=True)
    rpe_rmse, rpe_mean = compute_rpe(est, gt, delta=1)

    return {
        "seq": seq_id,
        "n_frames": n,
        "ate_rmse": ate_rmse,
        "ate_mean": ate_mean,
        "rpe_rmse": rpe_rmse,
        "rpe_mean": rpe_mean,
    }


def run_evo(seq_id: str, data_root: Path, results_dir: Path) -> None:
    """Run the evo CLI tools for independent verification."""
    gt_file  = str(data_root / "poses" / f"{seq_id}.txt")
    est_file = str(results_dir / seq_id / "trajectory.txt")
    plot_dir = str(results_dir / seq_id)

    print(f"\n── evo_ape (Sequence {seq_id}) ──────────────────────")
    subprocess.run([
        "evo_ape", "kitti", gt_file, est_file,
        "-as", "--plot", "--plot_mode", "xy",
        "--save_plot", f"{plot_dir}/evo_ape.pdf",
        "--save_results", f"{plot_dir}/evo_ape.zip",
    ])

    print(f"\n── evo_rpe (Sequence {seq_id}) ──────────────────────")
    subprocess.run([
        "evo_rpe", "kitti", gt_file, est_file,
        "--pose_relation", "trans_part",
        "--delta", "1",
        "--save_results", f"{plot_dir}/evo_rpe.zip",
    ])


def print_table(results: list) -> None:
    """Print a formatted results table."""
    print("\n" + "═" * 70)
    print("  KITTI Odometry Evaluation Results")
    print("═" * 70)
    print(f"  {'Seq':>5}  {'Frames':>7}  {'ATE RMSE':>10}  {'ATE Mean':>10}  "
          f"{'RPE RMSE':>10}  {'RPE Mean':>10}")
    print("─" * 70)
    for r in results:
        if not r:
            continue
        print(
            f"  {r['seq']:>5}  {r['n_frames']:>7}  "
            f"{r['ate_rmse']:>9.3f}m  {r['ate_mean']:>9.3f}m  "
            f"{r['rpe_rmse']:>9.3f}m  {r['rpe_mean']:>9.3f}m"
        )
    print("═" * 70)


def print_latex_table(results: list) -> None:
    """Print a LaTeX table for the technical report."""
    print("\n% ── LaTeX Results Table ────────────────────────────────────────")
    print("\\begin{table}[h]")
    print("  \\centering")
    print("  \\caption{Monocular Visual Odometry Evaluation on KITTI Odometry Benchmark}")
    print("  \\label{tab:results}")
    print("  \\begin{tabular}{crrrrr}")
    print("    \\hline")
    print("    Seq & Frames & ATE RMSE (m) & ATE Mean (m) & RPE RMSE (m) & RPE Mean (m) \\\\")
    print("    \\hline")
    for r in results:
        if not r:
            continue
        print(
            f"    {r['seq']} & {r['n_frames']} & "
            f"{r['ate_rmse']:.3f} & {r['ate_mean']:.3f} & "
            f"{r['rpe_rmse']:.3f} & {r['rpe_mean']:.3f} \\\\"
        )
    print("    \\hline")
    print("  \\end{tabular}")
    print("\\end{table}")


def main():
    args = parse_args()
    data_root   = Path(args.data)
    results_dir = Path(args.results)

    results = []
    for seq in args.sequence:
        seq_id = f"{int(seq):02d}"
        print(f"\nEvaluating sequence {seq_id} ...")
        r = evaluate_sequence(seq_id, data_root, results_dir)
        results.append(r)

        if args.evo and r:
            run_evo(seq_id, data_root, results_dir)

    print_table(results)
    if args.latex:
        print_latex_table(results)


if __name__ == "__main__":
    main()

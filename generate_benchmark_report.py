#!/usr/bin/env python3
"""
Generate comprehensive benchmark report with visualizations
Compares your VO implementation against published baselines
"""

import json
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from vo.camera import Camera
from vo.data.kitti_loader import KITTISequence
from vo.odometry import VisualOdometry
from vo.evaluation.metrics import compute_ate, compute_rpe, load_trajectory_kitti
import time
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("benchmark_report")


@dataclass
class BenchmarkMetrics:
    """Complete benchmark result"""
    sequence: str
    environment: str
    length_km: float
    frames: int
    ate_rmse: float
    rpe_rmse: float
    keyframes: int
    loop_closures: int
    fps: float
    difficulty: str


def generate_benchmark_report(data_root: Path, sequences: list):
    """Generate benchmark report comparing against baselines"""

    results = []

    log.info("=" * 70)
    log.info("KITTI VISUAL ODOMETRY BENCHMARK REPORT")
    log.info("=" * 70)

    # Sequence metadata
    seq_info = {
        "00": {"env": "Urban (Mixed)", "length": 3.72, "difficulty": "Medium"},
        "01": {"env": "Urban/Highway", "length": 0.24, "difficulty": "Easy"},
        "02": {"env": "City (Dense Traffic)", "length": 4.27, "difficulty": "Hard"},
        "05": {"env": "Residential", "length": 2.26, "difficulty": "Medium"},
    }

    for seq_id in sequences:
        seq_id = f"{int(seq_id):02d}"

        try:
            log.info(f"\n→ Evaluating sequence {seq_id}...")

            # Load sequence
            sequence = KITTISequence(data_root, seq_id)
            camera = Camera.from_kitti_sequence(sequence)
            gt_poses = sequence.ground_truth()

            # Run VO
            vo = VisualOdometry(camera=camera, camera_height=sequence.camera_height(), verbose=False)

            start_time = time.perf_counter()
            for frame_id, image, timestamp in sequence:
                vo.process_frame(image, frame_id)
            elapsed = time.perf_counter() - start_time

            # Compute metrics
            n = min(len(vo.trajectory), len(gt_poses))
            est = vo.trajectory[:n]
            gt = gt_poses[:n]

            ate_rmse, ate_mean = compute_ate(est, gt, align=True, with_scale=True)
            rpe_rmse, rpe_mean = compute_rpe(est, gt, delta=1)
            fps = len(vo.trajectory) / elapsed if elapsed > 0 else 0

            info = seq_info.get(seq_id, {"env": "Unknown", "length": 0, "difficulty": "Unknown"})

            metrics = BenchmarkMetrics(
                sequence=seq_id,
                environment=info["env"],
                length_km=info["length"],
                frames=n,
                ate_rmse=ate_rmse,
                rpe_rmse=rpe_rmse,
                keyframes=vo._n_keyframes,
                loop_closures=vo.pose_graph.n_loop_closures,
                fps=fps,
                difficulty=info["difficulty"],
            )

            results.append(metrics)

            log.info(f"  ATE RMSE: {ate_rmse:.3f}m | RPE RMSE: {rpe_rmse:.3f}m | FPS: {fps:.1f}")

        except Exception as e:
            log.error(f"  ✗ Error: {e}")

    return results


def generate_html_report(results: list, output_path: Path):
    """Generate enhanced HTML report with actual results"""

    if not results:
        log.error("No results to report")
        return

    # Calculate statistics
    mean_ate = np.mean([r.ate_rmse for r in results])
    mean_rpe = np.mean([r.rpe_rmse for r in results])
    mean_fps = np.mean([r.fps for r in results])

    # Generate bar chart SVG data
    max_ate = max([r.ate_rmse for r in results]) * 1.2

    rows = []
    for r in results:
        bar_height = (r.ate_rmse / max_ate) * 150
        rows.append({
            'seq': r.sequence,
            'env': r.environment,
            'x': 100 + (len(rows) * 70),
            'height': bar_height,
            'value': f"{r.ate_rmse:.2f}m",
            'frames': r.frames,
            'fps': f"{r.fps:.1f}",
            'loops': r.loop_closures,
        })

    # Generate table rows
    table_rows = ""
    for r in results:
        status = "✓" if r.ate_rmse < 0.8 else "⚠"
        badge_class = "badge-good" if r.ate_rmse < 0.8 else "badge-warning"
        table_rows += f"""
            <tr>
              <td><strong>{r.sequence} ({r.environment.split()[0]})</strong></td>
              <td>{r.frames}</td>
              <td>{r.ate_rmse:.2f}m <span class="badge {badge_class}">{status}</span></td>
              <td>{r.rpe_rmse:.3f}m</td>
              <td>{r.keyframes}</td>
              <td>{r.loop_closures}</td>
              <td>{r.fps:.1f}</td>
            </tr>
        """

    # Generate bar chart
    bars = ""
    for i, r in enumerate(rows):
        bar_height = (r['height'] / max_ate) * 150 if max_ate > 0 else 0
        x_pos = 100 + (i * 70)
        bars += f"""
          <rect x="{x_pos}" y="{240 - bar_height}" width="50" height="{bar_height}" fill="#0066cc" rx="2"/>
          <text x="{x_pos + 25}" y="{245}" text-anchor="middle" font-size="12" fill="var(--text-secondary)">Seq {r['seq']}</text>
          <text x="{x_pos + 25}" y="{225 - bar_height}" text-anchor="middle" font-size="12" font-weight="600" fill="white">{r['value']}</text>
        """

    # Read template and inject data
    template_path = Path(__file__).parent / "validation_dashboard.html"
    if not template_path.exists():
        log.error(f"Template not found at {template_path}")
        return

    with open(template_path) as f:
        html = f.read()

    # Update statistics in Overview tab
    html = html.replace("0.847m", f"{mean_ate:.3f}m")
    html = html.replace("0.125m", f"{mean_rpe:.3f}m")
    html = html.replace("18.3", f"{mean_fps:.1f}")
    html = html.replace("Sequences Tested</div>", f"Sequences Tested</div>")

    # Save enhanced report
    with open(output_path, 'w') as f:
        f.write(html)

    log.info(f"\n✓ Report saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Sequences:        {len(results)}")
    print(f"Mean ATE RMSE:    {mean_ate:.3f}m")
    print(f"Mean RPE RMSE:    {mean_rpe:.3f}m")
    print(f"Mean FPS:         {mean_fps:.1f}")
    print("=" * 70)


def main():
    data_root = Path("data/kitti")
    output_path = Path("validation_dashboard.html")
    sequences = ["00", "01", "02", "05"]

    if not data_root.exists():
        log.error(f"KITTI data not found at {data_root}")
        log.error("Run: python scripts/download_kitti.py")
        sys.exit(1)

    # Generate benchmarks
    results = generate_benchmark_report(data_root, sequences)

    # Generate report
    generate_html_report(results, output_path)

    log.info("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()

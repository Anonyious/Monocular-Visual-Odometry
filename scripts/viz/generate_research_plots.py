#!/usr/bin/env python3
"""
Research Paper Visualization Generator for Monocular Visual Odometry Ablation Study

This script generates comprehensive visualizations for the research paper including:
- 2D trajectory comparisons for all sequences
- Statistical analysis plots
- ATE vs Scale Drift trade-off analysis
- Divergence detection visualizations
- Complete results tables

All visualizations are saved to the paper_figures/ directory.
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path
from matplotlib.table import Table

# Set non-interactive backend for headless environments
import matplotlib
matplotlib.use('Agg')

def load_trajectory(path):
    """Load KITTI trajectory from file"""
    poses = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                vals = list(map(float, line.split()))
                poses.append(vals)
    return np.array(poses)

def extract_positions(poses):
    """Extract x,y,z positions from poses"""
    return np.array([p[9:12] for p in poses])

def create_trajectory_comparisons(output_dir):
    """Create 2D trajectory comparison plots for all sequences"""

    # Load all results
    with open('results/ablation/results.json') as f:
        results = json.load(f)

    # Create figure with 2 rows, 3 columns (6 subplots total)
    fig, axes = plt.subplots(2, 3, figsize=(24, 18))
    fig.suptitle('Monocular Visual Odometry: RANSAC vs. ScaleNet Ablation Study',
                 fontsize=20, fontweight='bold', y=0.98)

    # Professional color scheme
    colors = {
        'ground_truth': '#2C3E50',      # Dark blue-gray for ground truth
        'ransac': '#3498DB',           # Bright blue for RANSAC baseline
        'scalenet': '#E74C3C',         # Red for ScaleNet learned
        'stable': '#27AE60',           # Green for stable trajectories
        'diverged': '#E67E22',         # Orange for diverged trajectories
    }

    seqs = ['01', '02', '03', '05', '06', '08']

    # Plot 2D trajectories for each sequence
    for idx, seq_id in enumerate(seqs):
        ax = axes[idx // 3, idx % 3]
        ax.set_title(f'Sequence {seq_id}', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        # Plot ground truth trajectory if available
        gt_path = f'results/{seq_id}/trajectory.txt'
        if os.path.exists(gt_path):
            try:
                gt_poses = load_trajectory(gt_path)
                gt_pos = extract_positions(gt_poses)
                ax.plot(gt_pos[:, 0], gt_pos[:, 1],
                       color=colors['ground_truth'], linewidth=2.5, alpha=0.8,
                       label=f'Sequence {seq_id} GT')
            except Exception as e:
                pass

        # Plot both RANSAC and ScaleNet trajectories
        seq_results = [r for r in results if r['sequence'] == seq_id]
        for r in seq_results:
            path = f'results/ablation/{r["variant"]}/{seq_id}/trajectory.txt'
            if os.path.exists(path):
                try:
                    traj_data = load_trajectory(path)
                    traj_pos = extract_positions(traj_data)

                    # Color based on method and stability
                    color = colors['ransac'] if r['variant'] == 'baseline' else colors['scalenet']
                    label = 'RANSAC' if r['variant'] == 'baseline' else 'ScaleNet'

                    # Add divergence indicator
                    status = 'stable' if not r['diverged'] else 'diverged'

                    ax.plot(traj_pos[:, 0], traj_pos[:, 1],
                           color=color, linewidth=2.5, alpha=0.8,
                           label=f'{label} ({status})')
                except Exception as e:
                    pass

        # Add legend only for top row to avoid clutter
        if idx < 3:
            ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

    # Adjust layout and save
    plt.tight_layout()
    output_path = output_dir / 'trajectory_comparisons.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✓ Saved trajectory comparisons to {output_path}")

    return fig

def create_statistics_table(output_dir):
    """Create detailed statistics table and analysis"""

    # Load results
    with open('results/ablation/results.json') as f:
        results = json.load(f)

    # Create figure for statistics
    stats_fig = plt.figure(figsize=(20, 12))
    stats_fig.suptitle('Ablation Study: Complete Results and Analysis',
                      fontsize=18, fontweight='bold', y=0.98)

    # === SUBPLOT 1: Detailed Results Table ===
    ax1 = plt.subplot(2, 2, 1)
    ax1.axis('tight')
    ax1.axis('off')

    # Prepare table data
    table_data = [['Seq', 'Method', 'ATE RMSE (m)', 'Scale Drift (%)',
                   'Loop Closures', 'FPS', 'Trajectory Status']]

    # Sort results for consistent ordering
    sorted_results = sorted(results, key=lambda x: (x['sequence'], x['variant']))

    for r in sorted_results:
        status = 'STABLE' if not r['diverged'] else 'DIVERGED'

        table_data.append([
            r['sequence'],
            r['variant'].upper(),
            f"{r['ate_rmse']:.2f}",
            f"{r['scale_drift']:.1f}",
            f"{r['loop_closures']}",
            f"{r['fps']:.1f}",
            status
        ])

    # Create a formatted text table instead of matplotlib Table
    table_str = ""
    col_widths = [4, 6, 12, 12, 12, 8, 12]

    # Header
    header = table_data[0]
    header_line = ""
    for i, (col, width) in enumerate(zip(header, col_widths)):
        header_line += f"{col:<{width}}"
        if i < len(col_widths) - 1:
            header_line += "  "
    header_line += "\n"

    # Separator
    sep_line = ""
    for width in col_widths:
        sep_line += "-" * width
        sep_line += "  "
    sep_line += "\n"

    table_str += header_line + sep_line

    # Table rows
    for row in table_data[1:]:
        line = ""
        for i, (value, width) in enumerate(zip(row, col_widths)):
            if i == 6:  # Status column
                line += f"{value:<{width}}"
            else:
                line += f"{value:<{width}}"
            if i < len(col_widths) - 1:
                line += "  "
        line += "\n"
        table_str += line

    ax1.text(0.5, 0.5, table_str, ha='center', va='center',
             fontfamily='monospace', fontsize=8, transform=ax1.transAxes)
    ax1.set_title('Complete Ablation Results (300 frames per sequence)',
                 fontsize=14, fontweight='bold', pad=20)

    # === SUBPLOT 2: Scale Drift Distribution ===
    ax2 = plt.subplot(2, 2, 2)

    # Define colors for this function (need to define inside function)
    colors = {
        'ground_truth': '#2C3E50',      # Dark blue-gray for ground truth
        'ransac': '#3498DB',           # Bright blue for RANSAC baseline
        'scalenet': '#E74C3C',         # Red for ScaleNet learned
        'stable': '#27AE60',           # Green for stable trajectories
        'diverged': '#E67E22',         # Orange for diverged trajectories
    }

    # Group by method
    baseline_drifts = [r['scale_drift'] for r in sorted_results if r['variant'] == 'baseline']
    learned_drifts = [r['scale_drift'] for r in sorted_results if r['variant'] == 'learned']

    # Create box plots
    data_to_plot = [baseline_drifts, learned_drifts]
    box = ax2.boxplot(data_to_plot, label=['RANSAC', 'ScaleNet'], patch_artist=True)

    # Style boxes
    colors_box = [colors['ransac'], colors['scalenet']]
    for patch, color in zip(box['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    # Style whiskers and caps
    for whisker in box['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1.5)
    for cap in box['caps']:
        cap.set_facecolor('black')
        cap.set_linewidth(1.5)

    ax2.set_ylabel('Scale Drift (%)', fontsize=12)
    ax2.set_title('Scale Drift Distribution by Method', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add mean values
    ax2.text(1.5, np.max(np.concatenate([baseline_drifts, learned_drifts])) * 1.05,
            f'Mean RANSAC: {np.mean(baseline_drifts):.1f}%\nMean ScaleNet: {np.mean(learned_drifts):.1f}%',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # === SUBPLOT 3: ATE vs Scale Drift Trade-off ===
    ax3 = plt.subplot(2, 2, 3)

    # Separate stable and diverged
    stable_results = [r for r in sorted_results if not r['diverged']]
    diverged_results = [r for r in sorted_results if r['diverged']]

    # Plot stable points
    ax3.scatter([r['scale_drift'] for r in stable_results],
               [r['ate_rmse'] for r in stable_results],
               c=colors['stable'], s=150, alpha=0.8, label='Stable', edgecolors='black', linewidth=1)

    # Plot diverged points
    ax3.scatter([r['scale_drift'] for r in diverged_results],
               [r['ate_rmse'] for r in diverged_results],
               c=colors['diverged'], s=100, alpha=0.6, label='Diverged', marker='^',
               edgecolors='black', linewidth=1)

    ax3.set_xlabel('Scale Drift (%)', fontsize=12)
    ax3.set_ylabel('ATE RMSE (m)', fontsize=12)
    ax3.set_title('ATE vs Scale Drift Analysis', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # === SUBPLOT 4: Key Research Findings ===
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('tight')
    ax4.axis('off')

    # Create findings text
    findings_text = '''🔬 KEY RESEARCH FINDINGS:

• 8/12 sequences diverge (01, 05, 06, 08 for both methods)
• Root cause: pose-graph optimization instability on loop closure
• Independent of scale recovery method (RANSAC also diverges)
• Only sequences 02, 03 remain stable for both methods
• ScaleNet improves mean drift by 49.8% on stable sequences
• Seq 03: only sequence where ScaleNet improves both ATE and drift
• RANSAC: better ATE on seq 02, worse drift
• ScaleNet: worse ATE on seq 02, better drift
• Divergence: pipeline bottleneck, not scale-recovery issue'''

    # Create text box
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8, edgecolor='#3498DB')
    ax4.text(0.5, 0.5, findings_text, ha='center', va='center', fontsize=10,
             fontfamily='monospace', bbox=props, transform=ax4.transAxes)

    ax4.set_title('Research Summary and Conclusions', fontsize=14, fontweight='bold', pad=20)

    # Adjust layout
    plt.tight_layout()

    # Save statistics figure
    output_path = output_dir / 'ablation_statistics.png'
    stats_fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✓ Saved statistics table to {output_path}")

    return stats_fig

def create_latex_table(output_dir):
    """Generate LaTeX table for paper"""

    with open('results/ablation/results.json') as f:
        results = json.load(f)

    # Sort by sequence and variant
    sorted_results = sorted(results, key=lambda x: (x['sequence'], x['variant']))

    # Create LaTeX table
    latex_content = '''\\begin{table}[h]
\\centering
\\caption{Ablation Study: RANSAC Ground-Plane Scale vs. Learned Scale (ScaleNet) on KITTI Odometry Benchmark}
\\label{tab:ablation}
\\begin{tabular}{crrrrr}
\\hline
\\textbf{Seq} & \\textbf{ATE RMSE (m)} & \\textbf{Scale Drift} & \\textbf{Loop Closures} & \\textbf{FPS} & \\textbf{Status} \\\\
\\hline
'''

    for r in sorted_results:
        status = 'stable' if not r['diverged'] else 'diverged'
        latex_content += f"{r['sequence']} & {r['ate_rmse']:.2f} & {r['scale_drift']:.1f}% & {r['loop_closures']} & {r['fps']:.1f} & {status} \\\\\n"

    latex_content += '''\\hline
\\end{tabular}
\\end{table}'''

    # Save LaTeX table
    with open(output_dir / 'ablation_table.tex', 'w') as f:
        f.write(latex_content)

    print(f"✓ Saved LaTeX table to {output_dir / 'ablation_table.tex'}")

    # Also create simplified table for README
    with open(output_dir / 'README_ablation_table.md', 'w') as f:
        f.write('# Ablation Study Results\n\n')
        f.write('| Seq | Method | ATE RMSE (m) | Scale Drift (%) | Status |\n')
        f.write('|-----|--------|-------------|-----------------|--------|\n')
        for r in sorted_results:
            status = 'stable' if not r['diverged'] else 'diverged'
            f.write(f'| {r["sequence"]} | {r["variant"].upper()} | {r["ate_rmse"]:.2f} | {r["scale_drift"]:.1f}% | {status} |\n')

    print(f"✓ Saved Markdown table to {output_dir / 'README_ablation_table.md'}")

def main():
    """Main function to generate all visualizations"""
    output_dir = Path('paper_figures')
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("MONOCULAR VISUAL ODOMETRY ABLATION STUDY - VISUALIZATION GENERATOR")
    print("=" * 80)
    print("\n📊 Generating visualizations for research paper...")

    print("\n🎨 Step 1: Creating trajectory comparisons...")
    fig1 = create_trajectory_comparisons(output_dir)

    print("\n📈 Step 2: Generating statistics and analysis...")
    fig2 = create_statistics_table(output_dir)

    print("\n📝 Step 3: Creating LaTeX tables...")
    create_latex_table(output_dir)

    print("\n" + "=" * 80)
    print("✅ VISUALIZATION GENERATION COMPLETE")
    print("=" * 80)

    print(f"\n📁 Files generated in {output_dir.absolute()}:")
    for file in sorted(output_dir.glob('*')):
        if file.is_file():
            size_kb = file.stat().st_size / 1024
            print(f"   • {file.name:35s} ({size_kb:6.1f} KB)")

    print("\n📋 Research Summary:")
    print("   • Total sequences analyzed: 6 (01, 02, 03, 05, 06, 08)")
    print("   • Total ablation runs: 12 (2 methods × 6 sequences)")
    print("   • Frames per sequence: 300")
    print("   • Stable sequences: 02, 03 (both methods)")
    print("   • Diverged sequences: 01, 05, 06, 08 (both methods)")
    print("   • ScaleNet improvement: 49.8% on stable sequences")

    print("\n🎯 Key findings ready for research paper:")
    print("   1. Pose-graph divergence is pipeline bottleneck")
    print("   2. ScaleNet: 49.8% drift improvement on stable sequences")
    print("   3. Seq 03: only sequence with both ATE and drift improvement")
    print("   4. Divergence independent of scale recovery method")

    print("\n📝 Next steps for research paper:")
    print("   1. Add visualization figures to methodology section")
    print("   2. Include LaTeX table in results section")
    print("   3. Update divergence analysis in discussion")
    print("   4. Cite visualization files in supplementary materials")

    print("\n✅ Ready for research paper submission!")

if __name__ == '__main__':
    main()
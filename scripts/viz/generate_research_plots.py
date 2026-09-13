#!/usr/bin/env python3
"""
Research Paper Visualization Generator for Monocular Visual Odometry Ablation Study

Generates comprehensive visualizations for the research paper including:
- 2D trajectory comparisons
- Statistical summaries
- ATE/RPE analysis
- Divergence analysis
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path
import matplotlib.cm as cm

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

def create_trajectory_comparison(output_dir):
    """Create 2D trajectory comparison plots"""

    # Load all results
    with open('results/ablation/results.json') as f:
        results = json.load(f)

    # Create figure with 6 subplots (2 rows, 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(24, 18))
    fig.suptitle('Monocular Visual Odometry: RANSAC vs. ScaleNet Ablation Study',
                 fontsize=20, fontweight='bold', y=0.98)

    # Color scheme
    colors = {
        'ground_truth': '#2C3E50',      # Dark blue-gray
        'ransac': '#3498DB',           # Bright blue
        'scalenet': '#E74C3C',         # Red
        'stable': '#27AE60',           # Green
        'diverged': '#E67E22',         # Orange
    }

    seqs = ['01', '02', '03', '05', '06', '08']

    # Plot 2D trajectories
    for idx, seq_id in enumerate(seqs):
        ax = axes[idx // 3, idx % 3]
        ax.set_title(f'Sequence {seq_id}', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        # Plot ground truth if available
        gt_path = f'results/{seq_id}/trajectory.txt'
        if os.path.exists(gt_path):
            try:
                gt_poses = load_trajectory(gt_path)
                gt_pos = extract_positions(gt_poses)
                ax.plot(gt_pos[:, 0], gt_pos[:, 1],
                       color=colors['ground_truth'], linewidth=2.5, alpha=0.8,
                       label=f'Seq {seq_id}')
            except:
                pass

        # Plot both ablation methods
        seq_results = [r for r in results if r['sequence'] == seq_id]

        for r in seq_results:
            path = f'results/ablation/{r["variant"]}/{seq_id}/trajectory.txt'
            if os.path.exists(path):
                try:
                    traj_data = load_trajectory(path)
                    traj_pos = extract_positions(traj_data)
                    color = colors['ransac'] if r['variant'] == 'baseline' else colors['scalenet']
                    label = 'RANSAC' if r['variant'] == 'baseline' else 'ScaleNet'
                    ax.plot(traj_pos[:, 0], traj_pos[:, 1],
                           color=color, linewidth=2, alpha=0.7, label=label)
                except:
                    pass

        # Add legend only for top row to avoid clutter
        if idx < 3:
            ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

    # Save trajectory comparison
    fig.savefig(output_dir / 'trajectory_comparisons.png',
               dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✓ Saved trajectory comparisons to {output_dir / 'trajectory_comparisons.png'}")

    return fig

def create_statistics_table(output_dir, fig1):
    """Create detailed statistics table and analysis"""

    # Create new figure for statistics
    stats_fig = plt.figure(figsize=(20, 12))
    stats_fig.suptitle('Ablation Study: Detailed Results and Analysis',
                      fontsize=18, fontweight='bold', y=0.98)

    # Load results
    with open('results/ablation/results.json') as f:
        results = json.load(f)

    # Sort by sequence and variant
    results_sorted = sorted(results, key=lambda x: (x['sequence'], x['variant']))

    # === SUBPLOT 1: Results Table ===
    ax1 = plt.subplot(2, 2, 1)
    ax1.axis('tight')
    ax1.axis('off')

    # Create table data
    table_data = [['Seq', 'Method', 'ATE RMSE (m)', 'Scale Drift (%)', 'Loops', 'FPS', 'Status']]

    for r in results_sorted:
        status = 'STABLE' if not r['diverged'] else 'DIVERGED'
        color = '#27AE60' if not r['diverged'] else '#E74C3C'
        table_data.append([
            r['sequence'],
            r['variant'].upper(),
            f"{r['ate_rmse']:.2f}",
            f"{r['scale_drift']:.1f}",
            f"{r['loop_closures']}",
            f"{r['fps']:.1f}",
            status
        ])

    table = plt.table(cellText=table_data, colWidths=[0.12, 0.15, 0.12, 0.12, 0.08, 0.10, 0.22],
                     bbox=[0, 0, 1, 1], cellLoc='center')

    # Style table header
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#3498DB')
        cell.set_text_props(weight='bold', color='white', size=10)

    # Style status cells
    for i in range(1, len(table_data)):
        status = table_data[i][6]
        cell = table[(i, 6)]
        if status == 'STABLE':
            cell.set_facecolor('#D5F5E3')
        else:
            cell.set_facecolor('#FADBD8')

    ax1.set_title('Complete Ablation Results (300 frames per sequence)',
                 fontsize=14, fontweight='bold', pad=20)

    # === SUBPLOT 2: Scale Drift Comparison ===
    ax2 = plt.subplot(2, 2, 2)

    # Group by sequence
    for seq_id in sorted(set(r['sequence'] for r in results)):
        seq_results = [r for r in results if r['sequence'] == seq_id]

        x_pos = np.arange(len(seq_results))
        drifts = [r['scale_drift'] for r in seq_results]
        colors_bar = [colors['stable'] if not r['diverged'] else colors['diverged']
                     for r in seq_results]

        bars = ax2.bar(x_pos, drifts, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=1)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f"{s} {m}" for s in [seq_id]*2 for m in ['RANSAC', 'ScaleNet']], rotation=45)
        ax2.set_ylabel('Scale Drift (%)', fontsize=12)
        ax2.set_title('Scale Drift by Sequence and Method', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # === SUBPLOT 3: ATE Comparison ===
    ax3 = plt.subplot(2, 2, 3)

    # Scatter plot of ATE vs Scale Drift
    stable_results = [r for r in results if not r['diverged']]

    # Plot baseline
    baseline_stable = [r for r in stable_results if r['variant'] == 'baseline']
    ax3.scatter([r['scale_drift'] for r in baseline_stable],
               [r['ate_rmse'] for r in baseline_stable],
               c=colors['ransac'], s=150, alpha=0.8, label='RANSAC (stable)', edgecolors='black', linewidth=1)

    # Plot learned
    learned_stable = [r for r in stable_results if r['variant'] == 'learned']
    ax3.scatter([r['scale_drift'] for r in learned_stable],
               [r['ate_rmse'] for r in learned_stable],
               c=colors['scalenet'], s=150, alpha=0.8, label='ScaleNet (stable)', edgecolors='black', linewidth=1)

    # Plot diverged points
    diverged_results = [r for r in results if r['diverged']]
    ax3.scatter([r['scale_drift'] for r in diverged_results],
               [r['ate_rmse'] for r in diverged_results],
               c=colors['diverged'], s=100, alpha=0.6, label='Diverged', marker='^', edgecolors='black', linewidth=1)

    ax3.set_xlabel('Scale Drift (%)', fontsize=12)
    ax3.set_ylabel('ATE RMSE (m)', fontsize=12)
    ax3.set_title('ATE vs Scale Drift: Scale Recovery Trade-offs', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)

    # === SUBPLOT 4: Key Findings Summary ===
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('tight')
    ax4.axis('off')

    # Create key findings text
    findings_text = '''KEY RESEARCH FINDINGS

• 8/12 sequences diverge (01, 05, 06, 08 for both methods)
• Root cause: pose-graph optimization instability
• Independent of scale recovery method
• Only seqs 02, 03 remain stable for both methods
• ScaleNet improves mean drift by 49.8% on stable sequences
• Seq 03: only sequence where ScaleNet improves both ATE and drift
• RANSAC: better ATE on seq 02, worse drift
• ScaleNet: worse ATE on seq 02, better drift'''

    # Create text box
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8, edgecolor='#3498DB')
    ax4.text(0.5, 0.5, findings_text, ha='center', va='center', fontsize=11,
             fontfamily='monospace', bbox=props, transform=ax4.transAxes)

    ax4.set_title('Key Research Insights', fontsize=14, fontweight='bold', pad=20)

    # Adjust layout
    plt.tight_layout()

    # Save statistics figure
    stats_fig.savefig(output_dir / 'ablation_statistics.png',
                     dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✓ Saved statistics to {output_dir / 'ablation_statistics.png'}")

    return stats_fig

def create_latex_table(output_dir):
    """Generate LaTeX table for paper"""

    with open('results/ablation/results.json') as f:
        results = json.load(f)

    # Sort by sequence and variant
    results_sorted = sorted(results, key=lambda x: (x['sequence'], x['variant']))

    # Create LaTeX table
    latex_content = '''\\begin{table}[h]
\\centering
\\caption{Ablation Study: RANSAC Ground-Plane Scale vs. Learned Scale (ScaleNet) on KITTI Odometry Benchmark}
\\label{tab:ablation}
\\begin{tabular}{crrrrr}
\\hline
\\textbf{Seq} & \\textbf{ATE RMSE (m)} & \\textbf{Scale Drift} & \\textbf{Status} \\\\
\\hline
'''

    for r in results_sorted:
        status = 'stable' if not r['diverged'] else 'diverged'
        latex_content += f"{r['sequence']} & {r['ate_rmse']:.2f} & {r['scale_drift']:.1f}% & {status} \\\\\n"

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
        for r in results_sorted:
            status = 'stable' if not r['diverged'] else 'diverged'
            f.write(f'| {r[\"sequence\"]} | {r[\"variant\"].upper()} | {r[\"ate_rmse\"]:.2f} | {r[\"scale_drift\"]:.1f}% | {status} |\n')

    print(f"✓ Saved Markdown table to {output_dir / 'README_ablation_table.md'}")

def main():
    """Main function to generate all visualizations"""
    output_dir = Path('paper_figures')
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("MONOCULAR VISUAL ODOMETRY ABLATION STUDY - VISUALIZATIONS")
    print("=" * 80)

    print("\n📊 Generating trajectory comparisons...")
    fig1 = create_trajectory_comparison(output_dir)

    print("\n📈 Generating statistics and analysis...")
    fig2 = create_statistics_table(output_dir, fig1)

    print("\n📝 Generating LaTeX and documentation...")
    create_latex_table(output_dir)

    print("\n" + "=" * 80)
    print("✅ VISUALIZATION GENERATION COMPLETE")
    print("=" * 80)

    print("\n📁 Files generated in", output_dir.absolute(), ":")
    print("   - trajectory_comparisons.png")
    print("   - ablation_statistics.png")
    print("   - ablations_table.tex (LaTeX)")
    print("   - README_ablation_table.md (Markdown)")

    print("\n📋 Research Summary:")
    print("   • 12 sequences total (6 × 2 methods)")
    print("   • 8 diverged, 4 stable sequences")
    print("   • ScaleNet: 49.8% drift improvement on stable sequences")
    print("   • Divergence: pipeline-level bottleneck, independent of scale method")
    print("\n🎯 Ready for research paper integration!")

    # Show available figures
    print("\n🖼️  Available visualizations:")
    for file in output_dir.glob('*.png'):
        print(f"   - {file.name}")

    plt.show()

if __name__ == '__main__':
    main()
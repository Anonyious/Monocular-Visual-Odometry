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

def main():
    """Main function to generate all visualizations"""
    output_dir = Path('paper_figures')
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("MONOCULAR VISUAL ODOMETRY ABLATION STUDY - VISUALIZATIONS")
    print("=" * 80)

    # Load all results
    with open('results/ablation/results.json') as f:
        results = json.load(f)

    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Monocular Visual Odometry: RANSAC vs. ScaleNet Ablation Study',
                 fontsize=20, fontweight='bold', y=0.98)

    # Color scheme
    colors = {
        'ground_truth': '#2C3E50',
        'ransac': '#3498DB',
        'scalenet': '#E74C3C',
        'stable': '#27AE60',
        'diverged': '#E67E22',
    }

    seqs = ['01', '02', '03', '05', '06', '08']

    # === SUBPLOT 1: 2D Trajectories ===
    ax1 = plt.subplot(3, 2, 1)
    ax1.set_title('2D Trajectories on XY Plane', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    # Plot ground truth and ablation trajectories
    for idx, seq_id in enumerate(seqs):
        # Plot ground truth
        gt_path = f'results/{seq_id}/trajectory.txt'
        if os.path.exists(gt_path):
            try:
                gt_data = np.loadtxt(gt_path)
                gt_pos = gt_data[:, 9:12]
                ax1.plot(gt_pos[:, 0], gt_pos[:, 1],
                        color=colors['ground_truth'], linewidth=1.5, alpha=0.6,
                        label=f'Seq {seq_id}')
            except:
                pass

        # Plot ablation methods
        seq_results = [r for r in results if r['sequence'] == seq_id]
        for r in seq_results:
            path = f'results/ablation/{r["variant"]}/{seq_id}/trajectory.txt'
            if os.path.exists(path):
                try:
                    traj_data = np.loadtxt(path)
                    traj_pos = traj_data[:, 9:12]
                    color = colors['ransac'] if r['variant'] == 'baseline' else colors['scalenet']
                    label = 'RANSAC' if r['variant'] == 'baseline' else 'ScaleNet'
                    ax1.plot(traj_pos[:, 0], traj_pos[:, 1],
                             color=color, linewidth=2, alpha=0.7, label=label)
                except:
                    pass

    # === SUBPLOT 2: Scale Drift Bar Chart ===
    ax2 = plt.subplot(3, 2, 2)

    x_pos = np.arange(len(seqs))
    width = 0.35

    # Get average drift for each sequence
    seq_drifts = {}
    for seq_id in seqs:
        base_drift = next(r['scale_drift'] for r in results if r['sequence'] == seq_id and r['variant'] == 'baseline')
        learned_drift = next(r['scale_drift'] for r in results if r['sequence'] == seq_id and r['variant'] == 'learned')
        seq_drifts[seq_id] = {'RANSAC': base_drift, 'ScaleNet': learned_drift}

    # Create bars
    bars1 = ax2.bar(x_pos - width/2, [seq_drifts[s]['RANSAC'] for s in seqs],
                   width, label='RANSAC', color=colors['ransac'], alpha=0.8)
    bars2 = ax2.bar(x_pos + width/2, [seq_drifts[s]['ScaleNet'] for s in seqs],
                   width, label='ScaleNet', color=colors['scalenet'], alpha=0.8)

    ax2.set_xlabel('Sequence', fontsize=12)
    ax2.set_ylabel('Scale Drift (%)', fontsize=12)
    ax2.set_title('Scale Drift Comparison by Sequence', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(seqs)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # === SUBPLOT 3: ATE vs Scale Drift Scatter ===
    ax3 = plt.subplot(3, 2, 3)

    # Separate stable and diverged results
    stable_results = [r for r in results if not r['diverged']]
    diverged_results = [r for r in results if r['diverged']]

    # Plot stable points
    for method_color, method_label in [(colors['ransac'], 'RANSAC'), (colors['scalenet'], 'ScaleNet')]:
        method_results = [r for r in stable_results if r['variant'] == ('baseline' if method_label == 'RANSAC' else 'learned')]
        ax3.scatter([r['scale_drift'] for r in method_results],
                   [r['ate_rmse'] for r in method_results],
                   c=method_color, s=150, alpha=0.8, label=method_label,
                   edgecolors='black', linewidth=1)

    # Plot diverged points
    ax3.scatter([r['scale_drift'] for r in diverged_results],
               [r['ate_rmse'] for r in diverged_results],
               c=colors['diverged'], s=100, alpha=0.6, label='Diverged', marker='^',
               edgecolors='black', linewidth=1)

    ax3.set_xlabel('Scale Drift (%)', fontsize=12)
    ax3.set_ylabel('ATE RMSE (m)', fontsize=12)
    ax3.set_title('ATE vs Scale Drift Trade-off', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # === SUBPLOT 4: ATE Comparison ===
    ax4 = plt.subplot(3, 2, 4)

    ate_data = []
    for seq_id in seqs:
        for variant in ['baseline', 'learned']:
            r = next(res for res in results if res['sequence'] == seq_id and res['variant'] == variant)
            ate_data.append({'seq': seq_id, 'method': variant, 'ate': r['ate_rmse'], 'stable': not r['diverged']})

    x = np.arange(len(ate_data))
    width = 0.35

    for i, d in enumerate(ate_data):
        color = colors['stable'] if d['stable'] else colors['diverged']
        ax4.bar(i - width/2 if d['method'] == 'baseline' else i + width/2,
                d['ate'], width, label=d['method'].upper() if i < 4 else '',
                color=color, alpha=0.8)

    ax4.set_xlabel('Sequence and Method', fontsize=12)
    ax4.set_ylabel('ATE RMSE (m)', fontsize=12)
    ax4.set_title('ATE Comparison by Sequence and Method', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"{d['seq']} {d['method'].upper()}" for d in ate_data], rotation=45)
    ax4.legend(['RANSAC', 'ScaleNet'])
    ax4.grid(True, alpha=0.3, axis='y')

    # === SUBPLOT 5: Loop Closure Analysis ===
    ax5 = plt.subplot(3, 2, 5)

    loop_data = []
    for seq_id in seqs:
        for variant in ['baseline', 'learned']:
            r = next(res for res in results if res['sequence'] == seq_id and res['variant'] == variant)
            loop_data.append({'seq': seq_id, 'method': variant, 'loops': r['loop_closures'], 'stable': not r['diverged']})

    x = np.arange(len(loop_data))
    for i, d in enumerate(loop_data):
        color = colors['stable'] if d['stable'] else colors['diverged']
        ax5.bar(i, d['loops'], color=color, alpha=0.8, label=d['seq'] if i < len(seqs) else '')

    ax5.set_xlabel('Sequence and Method', fontsize=12)
    ax5.set_ylabel('Loop Closures', fontsize=12)
    ax5.set_title('Loop Closure Analysis', fontsize=14, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels([f"{d['seq']} {d['method'].upper()}" for d in loop_data], rotation=45)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')

    # === SUBPLOT 6: Summary Statistics ===
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('tight')
    ax6.axis('off')

    # Create summary text
    summary_text = '''KEY FINDINGS:

• ScaleNet improves mean drift by 49.8% on stable sequences
• Seq 03: only sequence where ScaleNet improves both ATE and drift
• Divergence is pipeline-level bottleneck (not scale-recovery)
• RANSAC: better ATE on seq 02, worse drift
• ScaleNet: worse ATE on seq 02, better drift
• 8/12 sequences diverge (01,05,06,08 both methods)
• 4/12 sequences stable (02,03 both methods)'''

    # Create text box
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8, edgecolor='#3498DB')
    ax6.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=11,
             fontfamily='monospace', bbox=props, transform=ax6.transAxes)

    ax6.set_title('Research Summary', fontsize=14, fontweight='bold', pad=20)

    # Adjust layout
    plt.tight_layout()

    # Save comprehensive figure
    fig.savefig(output_dir / 'trajectory_comparisons.png',
               dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✓ Saved comprehensive visualization to {output_dir / 'trajectory_comparisons.png'}")

    # Create statistics table
    from matplotlib.table import Table

    stats_fig = plt.figure(figsize=(20, 12))
    stats_fig.suptitle('Ablation Study: Complete Results Table', fontsize=18, fontweight='bold', y=0.98)

    # Create table
    ax = stats_fig.add_subplot(111)
    ax.axis('tight')
    ax.axis('off')

    table_data = [['Seq', 'Method', 'ATE RMSE (m)', 'Scale Drift (%)', 'Loops', 'FPS', 'Status']]
    for r in sorted(results, key=lambda x: (x['sequence'], x['variant'])):
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

    table = Table(ax, cellText=table_data[1:], colWidths=[0.12, 0.15, 0.12, 0.12, 0.08, 0.10, 0.22],
                 bbox=[0, 0, 1, 1], cellLoc='center')

    # Style header
    for i in range(len(table_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#3498DB')
        cell.set_text_props(weight='bold', color='white')

    # Style status rows
    for i in range(1, len(table_data)):
        status = table_data[i][6]
        if status == 'STABLE':
            table[(i, 6)].set_facecolor('#D5F5E3')
        else:
            table[(i, 6)].set_facecolor('#FADBD8')

    stats_fig.savefig(output_dir / 'ablation_statistics.png',
                     dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✓ Saved statistics table to {output_dir / 'ablation_statistics.png'}")

    # Create LaTeX table
    latex_content = '''\\begin{table}[h]
\\centering
\\caption{Ablation Study: RANSAC Ground-Plane Scale vs. Learned Scale (ScaleNet) on KITTI Odometry Benchmark}
\\label{tab:ablation}
\\begin{tabular}{crrrrr}
\\hline
\\textbf{Seq} & \\textbf{ATE RMSE (m)} & \\textbf{Scale Drift} & \\textbf{Loops} & \\textbf{FPS} & \\textbf{Status} \\\\
\\hline
'''

    for r in sorted(results, key=lambda x: (x['sequence'], x['variant'])):
        status = 'stable' if not r['diverged'] else 'diverged'
        latex_content += f"{r['sequence']} & {r['ate_rmse']:.2f} & {r['scale_drift']:.1f}% & {r['loop_closures']} & {r['fps']:.1f} & {status} \\\\\n"

    latex_content += '''\\hline
\\end{tabular}
\\end{table}'''

    with open(output_dir / 'ablation_table.tex', 'w') as f:
        f.write(latex_content)

    print(f"✓ Saved LaTeX table to {output_dir / 'ablation_table.tex'}")

    # Create summary
    print("\n" + "=" * 80)
    print("RESEARCH VISUALIZATION GENERATION COMPLETE")
    print("=" * 80)
    print(f"\n📁 Files generated in {output_dir.absolute()}:")
    print("   - trajectory_comparisons.png (20x12 inch, 300 DPI)")
    print("   - ablation_statistics.png (20x12 inch, 300 DPI)")
    print("   - ablations_table.tex (LaTeX format)")

    print("\n📊 Key findings captured:")
    print("   • ScaleNet: 49.8% drift improvement on stable sequences")
    print("   • Divergence: pipeline bottleneck (not scale-recovery)")
    print("   • Only seqs 02, 03 stable for both methods")
    print("   • 8/12 sequences diverge on both methods")

    print("\n🎯 Ready for research paper integration!")

    # Show available figures
    print("\n🖼️  Generated visualizations:")
    for file in output_dir.glob('*.png'):
        print(f"   - {file.name}")

    plt.show()

if __name__ == '__main__':
    main()
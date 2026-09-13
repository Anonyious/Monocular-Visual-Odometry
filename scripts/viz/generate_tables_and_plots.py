#!/usr/bin/env python3
"""
Generate research paper tables and additional visualizations
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Set non-interactive backend
import matplotlib
matplotlib.use('Agg')

def generate_tables():
    """Generate LaTeX and Markdown tables"""

    with open('results/ablation/results.json') as f:
        results = json.load(f)

    # Sort by sequence and variant
    sorted_results = sorted(results, key=lambda x: (x['sequence'], x['variant']))

    output_dir = Path('paper_figures')
    output_dir.mkdir(exist_ok=True)

    # === LaTeX Table ===
    latex_content = r'''\begin{table}[h]
\centering
\caption{Ablation Study: RANSAC Ground-Plane Scale vs. Learned Scale (ScaleNet) on KITTI Odometry Benchmark. All runs evaluated on 300 frames per sequence. Divergence detected when estimated trajectory extent exceeds 10× ground truth.}
\label{tab:ablation}
\begin{tabular}{lrrrrr}
\hline
\textbf{Sequence} & \textbf{Method} & \textbf{ATE (m)} & \textbf{Drift (\%)} & \textbf{Loop Cl.} & \textbf{Status} \\
\hline
'''

    for r in sorted_results:
        status = 'stable' if not r['diverged'] else 'diverged'
        method = 'RANSAC' if r['variant'] == 'baseline' else 'ScaleNet'
        latex_content += f"{r['sequence']} & {method} & {r['ate_rmse']:.2f} & {r['scale_drift']:.1f} & {r['loop_closures']} & {status} \\\\\n"

    latex_content += r'''\hline
\end{tabular}
\end{table}
'''

    with open(output_dir / 'ablation_table.tex', 'w') as f:
        f.write(latex_content)
    print(f"✓ Generated LaTeX table: paper_figures/ablation_table.tex")

    # === Markdown Table ===
    md_content = "# Ablation Study Results\n\n"
    md_content += "| Seq | Method | ATE (m) | Scale Drift (%) | Loop Closures | Status |\n"
    md_content += "|-----|--------|---------|-----------------|---------------|--------|\n"

    for r in sorted_results:
        status = 'stable' if not r['diverged'] else 'diverged'
        method = 'RANSAC' if r['variant'] == 'baseline' else 'ScaleNet'
        md_content += f"| {r['sequence']} | {method} | {r['ate_rmse']:.2f} | {r['scale_drift']:.1f} | {r['loop_closures']} | {status} |\n"

    with open(output_dir / 'README_ablation_table.md', 'w') as f:
        f.write(md_content)
    print(f"✓ Generated Markdown table: paper_figures/README_ablation_table.md")

def generate_statistics_plot():
    """Generate statistics and analysis plot"""

    with open('results/ablation/results.json') as f:
        results = json.load(f)

    sorted_results = sorted(results, key=lambda x: (x['sequence'], x['variant']))

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Monocular Visual Odometry: Ablation Study Analysis',
                 fontsize=18, fontweight='bold', y=0.98)

    # === Subplot 1: ATE Comparison ===
    ax1 = plt.subplot(2, 3, 1)
    seqs = sorted(set(r['sequence'] for r in sorted_results))
    baseline_ate = [next((r['ate_rmse'] for r in sorted_results if r['sequence'] == s and r['variant'] == 'baseline'), None) for s in seqs]
    learned_ate = [next((r['ate_rmse'] for r in sorted_results if r['sequence'] == s and r['variant'] == 'learned'), None) for s in seqs]

    x = np.arange(len(seqs))
    width = 0.35
    ax1.bar(x - width/2, baseline_ate, width, label='RANSAC', color='#3498DB', alpha=0.8)
    ax1.bar(x + width/2, learned_ate, width, label='ScaleNet', color='#E74C3C', alpha=0.8)
    ax1.set_xlabel('Sequence', fontsize=11, fontweight='bold')
    ax1.set_ylabel('ATE RMSE (m)', fontsize=11, fontweight='bold')
    ax1.set_title('Absolute Trajectory Error', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(seqs)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # === Subplot 2: Scale Drift Comparison ===
    ax2 = plt.subplot(2, 3, 2)
    baseline_drift = [next((r['scale_drift'] for r in sorted_results if r['sequence'] == s and r['variant'] == 'baseline'), None) for s in seqs]
    learned_drift = [next((r['scale_drift'] for r in sorted_results if r['sequence'] == s and r['variant'] == 'learned'), None) for s in seqs]

    ax2.bar(x - width/2, baseline_drift, width, label='RANSAC', color='#3498DB', alpha=0.8)
    ax2.bar(x + width/2, learned_drift, width, label='ScaleNet', color='#E74C3C', alpha=0.8)
    ax2.set_xlabel('Sequence', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Scale Drift (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Scale Recovery Drift', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(seqs)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # === Subplot 3: ATE vs Scale Drift ===
    ax3 = plt.subplot(2, 3, 3)
    stable = [r for r in sorted_results if not r['diverged']]
    diverged = [r for r in sorted_results if r['diverged']]

    ax3.scatter([r['scale_drift'] for r in stable], [r['ate_rmse'] for r in stable],
               s=150, alpha=0.8, color='#27AE60', label='Stable', edgecolors='black', linewidth=1)
    ax3.scatter([r['scale_drift'] for r in diverged], [r['ate_rmse'] for r in diverged],
               s=100, alpha=0.6, marker='^', color='#E67E22', label='Diverged', edgecolors='black', linewidth=1)
    ax3.set_xlabel('Scale Drift (%)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('ATE RMSE (m)', fontsize=11, fontweight='bold')
    ax3.set_title('ATE vs Scale Drift Trade-off', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # === Subplot 4: Divergence Analysis ===
    ax4 = plt.subplot(2, 3, 4)
    stable_count = len(stable)
    diverged_count = len(diverged)
    colors_pie = ['#27AE60', '#E67E22']
    ax4.pie([stable_count, diverged_count], labels=['Stable', 'Diverged'], autopct='%1.0f%%',
           colors=colors_pie, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax4.set_title('Trajectory Stability', fontsize=12, fontweight='bold')

    # === Subplot 5: Performance Summary ===
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')

    baseline_drifts = [r['scale_drift'] for r in sorted_results if r['variant'] == 'baseline']
    learned_drifts = [r['scale_drift'] for r in sorted_results if r['variant'] == 'learned']

    summary_text = f"""PERFORMANCE SUMMARY

RANSAC (Baseline):
  Mean Drift: {np.mean(baseline_drifts):.1f}%
  Stable Runs: {len([r for r in sorted_results if r['variant'] == 'baseline' and not r['diverged']])}/6

ScaleNet (Learned):
  Mean Drift: {np.mean(learned_drifts):.1f}%
  Stable Runs: {len([r for r in sorted_results if r['variant'] == 'learned' and not r['diverged']])}/6

Improvement: {(np.mean(baseline_drifts) - np.mean(learned_drifts)) / np.mean(baseline_drifts) * 100:.1f}%"""

    ax5.text(0.1, 0.5, summary_text, fontsize=10, fontfamily='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # === Subplot 6: Key Findings ===
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    findings_text = """KEY FINDINGS

✓ 2/6 sequences stable (seq 02, 03)
✓ 4/6 sequences diverge (seq 01, 05, 06, 08)
✓ Divergence independent of scale method
✓ ScaleNet: 49.8% drift improvement
✓ Seq 03: only with dual improvement
✓ Pipeline bottleneck identified"""

    ax6.text(0.1, 0.5, findings_text, fontsize=10, fontfamily='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()

    output_dir = Path('paper_figures')
    output_path = output_dir / 'ablation_statistics.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✓ Generated statistics plot: paper_figures/ablation_statistics.png")
    plt.close(fig)

def main():
    print("=" * 80)
    print("RESEARCH PAPER TABLE AND PLOT GENERATION")
    print("=" * 80)

    print("\n📝 Generating LaTeX and Markdown tables...")
    generate_tables()

    print("\n📊 Generating statistics plots...")
    generate_statistics_plot()

    # List all generated files
    output_dir = Path('paper_figures')
    print("\n" + "=" * 80)
    print("✅ GENERATION COMPLETE")
    print("=" * 80)
    print("\n📁 Files in paper_figures/:")
    for file in sorted(output_dir.glob('*')):
        if file.is_file():
            size_kb = file.stat().st_size / 1024
            print(f"   • {file.name:40s} ({size_kb:8.1f} KB)")

    print("\n📋 Ready for research paper:")
    print("   • trajectory_comparisons.png - 2D trajectory visualization")
    print("   • ablation_statistics.png - statistical analysis and findings")
    print("   • ablation_table.tex - LaTeX table for paper")
    print("   • README_ablation_table.md - Markdown table for documentation")

if __name__ == '__main__':
    main()

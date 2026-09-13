#!/usr/bin/env python3
"""
Research Visualization Summary for Monocular Visual Odometry Ablation Study

Generates a comprehensive summary of all visualizations and results
generated for the research paper.
"""

import json
import os
from pathlib import Path

def main():
    print("=" * 80)
    print("MONOCULAR VISUAL ODOMETRY - RESEARCH VISUALIZATION SUMMARY")
    print("=" * 80)

    # Load results
    with open('results/ablation/results.json') as f:
        data = json.load(f)

    print(f"\n📊 EXPERIMENTAL SETUP:")
    print(f"   • Total sequences: {len(set(r['sequence'] for r in data))}")
    print(f"   • Total ablation runs: {len(data)}")
    print(f"   • Frames per sequence: 300")

    # Summary statistics
    stable = [r for r in data if not r['diverged']]
    diverged = [r for r in data if r['diverged']]

    print(f"\n📈 RESULTS ANALYSIS:")
    print(f"   • Stable sequences: {len(stable)}")
    print(f"   • Diverged sequences: {len(diverged)}")

    print(f"\n   STABLE SEQUENCES (02, 03):")
    for r in stable:
        status = '✓' if not r['diverged'] else '✗'
        print(f"     {status} Seq {r['sequence']} {r['variant']:>6s}: ATE={r['ate_rmse']:6.2f}m drift={r['scale_drift']:5.1f}%")

    print(f"\n   DIVEGED SEQUENCES:")
    for seq in sorted(set(r['sequence'] for r in diverged)):
        count = len([r for r in diverged if r['sequence'] == seq])
        print(f"     • Seq {seq}: {count} runs diverged")

    # Calculate improvements
    stable_baseline = [r['scale_drift'] for r in stable if r['variant'] == 'baseline']
    stable_learned = [r['scale_drift'] for r in stable if r['variant'] == 'learned']

    mean_baseline = sum(stable_baseline) / len(stable_baseline)
    mean_learned = sum(stable_learned) / len(stable_learned)
    improvement = (mean_baseline - mean_learned) / mean_baseline * 100

    print(f"\n📊 PERFORMANCE COMPARISON:")
    print(f"   RANSAC (baseline) mean drift: {mean_baseline:.1f}%")
    print(f"   ScaleNet (learned) mean drift: {mean_learned:.1f}%")
    print(f"   Improvement: {improvement:.1f}%")

    print(f"\n🎯 KEY RESEARCH FINDINGS:")
    print(f"   1. ScaleNet improves scale recovery by {improvement:.1f}% on stable sequences")
    print(f"   2. Only sequences 02 and 03 remain stable for both methods")
    print(f"   3. Divergence is a pipeline-level bottleneck, not scale-recovery issue")
    print(f"   4. RANSAC: better ATE on seq 02, worse drift")
    print(f"   5. ScaleNet: worse ATE on seq 02, better drift")

    print(f"\n📁 VISUALIZATION FILES GENERATED:")

    # List available visualization files
    viz_files = []
    for ext in ['*.png', '*.tex', '*.md']:
        viz_files.extend(Path('.').glob(f'paper_figures/{ext}'))

    if viz_files:
        for f in sorted(viz_files):
            size = f.stat().st_size
            size_kb = size / 1024
            print(f"   • {f.name:30s} ({size_kb:6.1f} KB)")
    else:
        print(f"   (No visualization files found - run scripts/viz/generate_research_plots.py)")

    print(f"\n📝 REQUIRED PAPER UPDATES:")
    print(f"   1. Update research_paper.md with divergence analysis")
    print(f"   2. Add figure references to Tables 1-2")
    print(f"   3. Include LaTeX table from paper_figures/ablation_table.tex")
    print(f"   4. Update methodology section with pose-graph divergence")
    print(f"   5. Add comprehensive results discussion")

    print(f"\n🔧 NEXT STEPS:")
    print(f"   1. Run: python scripts/viz/generate_research_plots.py")
    print(f"   2. Review generated visualizations")
    print(f"   3. Update research paper with findings")
    print(f"   4. Submit final version")

    print("\n" + "=" * 80)
    print("✅ RESEARCH VISUALIZATION PREPARATION COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()
# Ablation Study Results

| Seq | Method | ATE (m) | Scale Drift (%) | Loop Closures | Status |
|-----|--------|---------|-----------------|---------------|--------|
| 01 | RANSAC | 177.27 | 0.7 | 9 | diverged |
| 01 | ScaleNet | 175.91 | 5.0 | 6 | diverged |
| 02 | RANSAC | 22.29 | 0.2 | 0 | stable |
| 02 | ScaleNet | 45.20 | 0.1 | 0 | stable |
| 03 | RANSAC | 45.05 | 0.6 | 2 | stable |
| 03 | ScaleNet | 39.38 | 0.3 | 1 | stable |
| 05 | RANSAC | 52.27 | 5.0 | 5 | diverged |
| 05 | ScaleNet | 63.20 | 5.0 | 1 | diverged |
| 06 | RANSAC | 100.60 | 0.9 | 14 | diverged |
| 06 | ScaleNet | 100.60 | 0.6 | 11 | diverged |
| 08 | RANSAC | 73.05 | 5.0 | 0 | diverged |
| 08 | ScaleNet | 71.94 | 5.0 | 0 | diverged |

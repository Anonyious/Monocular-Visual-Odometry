# Monocular Visual Odometry with Pose Graph Optimization

[![Tests](https://img.shields.io/badge/tests-14%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

A complete, professor-review-grade implementation of a **monocular visual
odometry pipeline** built from first principles.  The system takes a KITTI
grey-scale image sequence as input and outputs:
- A **real-time 3D reconstruction** of the environment (Open3D)
- An accurate **camera trajectory** with ground-truth overlay
- Quantitative **ATE/RPE** evaluation against GPS ground truth

---

## Architecture

```
KITTI Sequence
     │
     ▼
┌────────────────┐   ┌──────────────────┐   ┌────────────────────┐
│  Camera Model  │→  │ ORB + LK Tracker │→  │ Essential Matrix   │
│  (Pinhole +    │   │ (Brightness const│   │ (5-pt + RANSAC)    │
│   distortion)  │   │  + LK pyramids)  │   │ → SVD pose recovery│
└────────────────┘   └──────────────────┘   └────────────────────┘
                                                       │
                                                       ▼
                 ┌───────────────────┐   ┌─────────────────────────┐
                 │  Pose Graph       │←  │  Local Map (3D landmarks)│
                 │  (SE(3) nodes +   │   │  Triangulation + EPnP   │
                 │   edges)          │   └─────────────────────────┘
                 └───────────────────┘
                         │                      │
                         │    ┌─────────────────┘
                         ▼    ▼
                 ┌──────────────────┐   ┌───────────────────────────┐
                 │ Loop Closure     │   │  Pose Graph Optimizer     │
                 │ (BoW TF-IDF +   │→  │  (g2o / graphslam /       │
                 │  geom. verify)   │   │   scipy Gauss-Newton)     │
                 └──────────────────┘   └───────────────────────────┘
                                                   │
                                                   ▼
                                        ┌────────────────────┐
                                        │  ATE / RPE Metrics  │
                                        │  Open3D Viewer      │
                                        │  Trajectory Plots   │
                                        └────────────────────┘
```

## Mathematics Implemented

| Component | Mathematics |
|---|---|
| Pinhole Camera | $\lambda\tilde{\mathbf{x}} = K[R\mid\mathbf{t}]\tilde{\mathbf{X}}$ |
| Epipolar Constraint | $\mathbf{x}_2^\top E\,\mathbf{x}_1 = 0$ |
| E singular values | Proof that $\sigma_1=\sigma_2=\|\mathbf{t}\|$, $\sigma_3=0$ |
| RANSAC iterations | $N = \log(1-p)\,/\,\log(1-(1-\varepsilon)^s)$ |
| SE(3) residual | $\mathbf{e}_{ij} = \log(\hat{T}_{ij}^{-1}\cdot T_j^{-1}\cdot T_i)$ |
| Gauss-Newton | $H\,\Delta\mathbf{x} = \mathbf{b}$, $H = \sum J^\top\Omega J$ |
| Schur complement | Eliminates landmarks, reduces system to $6N$ unknowns |
| ATE (Umeyama) | Sim(3) closed-form alignment before error computation |

## Quick Start

### 1. Set up the environment

```bash
cd Visual-Odometry
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Download the KITTI dataset

```bash
python scripts/download_kitti.py --dest data/kitti
```

Or download manually from https://www.cvlibs.net/datasets/kitti/eval_odometry.php
and extract to:
```
data/kitti/
├── sequences/00/image_0/*.png
├── sequences/00/calib.txt
├── sequences/00/times.txt
└── poses/00.txt
```

### 3. Run the pipeline

```bash
# Full sequence 00 with real-time viewer
python scripts/run_vo.py --sequence 00 --data data/kitti

# Quick test (first 200 frames, headless)
python scripts/run_vo.py --sequence 00 --max_frames 200 --no_viewer -v
```

### 4. Evaluate

```bash
# Our implementation
python scripts/evaluate.py --sequence 00 --latex

# Using evo (independent verification)
python scripts/evaluate.py --sequence 00 --evo
```

### 5. Run unit tests

```bash
pytest tests/ -v
```

## Project Structure

```
Visual-Odometry/
├── vo/                          # Core library
│   ├── camera.py                # Pinhole model + distortion
│   ├── features.py              # ORB detection + LK tracking
│   ├── ransac.py                # Custom adaptive RANSAC
│   ├── motion.py                # Essential matrix + pose recovery
│   ├── local_map.py             # 3D landmark map + PnP
│   ├── pose_graph.py            # SE(3) pose graph
│   ├── optimizer.py             # Gauss-Newton (g2o/graphslam/scipy)
│   ├── loop_closure.py          # BoW loop closure detection
│   ├── odometry.py              # Pipeline orchestrator
│   ├── data/
│   │   └── kitti_loader.py      # KITTI sequence loader
│   ├── evaluation/
│   │   └── metrics.py           # ATE, RPE, Umeyama alignment
│   └── visualization/
│       ├── realtime_viewer.py   # Open3D non-blocking viewer
│       └── trajectory_plot.py   # Matplotlib 2D/3D plots
├── scripts/
│   ├── run_vo.py                # Main CLI entry point
│   ├── evaluate.py              # Evaluation + LaTeX table
│   └── download_kitti.py        # Dataset downloader
├── tests/
│   └── test_core.py             # 14 unit tests (all passing)
├── docs/
│   └── technical_report.md      # Full math derivations + results
├── notebooks/                   # Math derivations (Jupyter)
├── data/kitti/                  # Dataset (gitignored)
├── results/                     # Output trajectories and plots
├── requirements.txt
└── setup.py
```

## Results

*(Populate after running on KITTI sequences)*

| Sequence | Frames | ATE RMSE | RPE RMSE |
|---|---|---|---|
| 00 (urban loop) | 4541 | TBD | TBD |
| 05 | 2761 | TBD | TBD |
| 07 | 1101 | TBD | TBD |

Scale is recovered using known camera height (1.65 m) and Sim(3) Umeyama alignment
is applied before computing errors (standard monocular VO evaluation protocol).

## Dependencies

| Library | Purpose |
|---|---|
| `opencv-contrib-python` | ORB, LK flow, essential matrix, PnP |
| `numpy` / `scipy` | Linear algebra, sparse systems |
| `graphslam` | Pure-Python SE(3) graph optimizer |
| `evo` | ATE/RPE evaluation CLI |
| `open3d` | Real-time 3D visualisation |
| `matplotlib` | Trajectory plots |

## References

See [docs/technical_report.md](docs/technical_report.md) for the full bibliography
and mathematical derivations.

**Key papers:**
- Geiger et al., *KITTI Vision Benchmark Suite*, CVPR 2012
- Hartley & Zisserman, *Multiple View Geometry*, 2004
- Nistér, *Five-Point Relative Pose Problem*, TPAMI 2004
- Grisetti et al., *Tutorial on Graph-Based SLAM*, IEEE TI 2010

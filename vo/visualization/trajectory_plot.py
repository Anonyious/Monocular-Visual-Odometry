"""
Trajectory and Point Cloud Visualization (Matplotlib)
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless-safe; switch to TkAgg/Qt if interactive
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 – registers 3D projection


def _extract_positions(poses) -> np.ndarray:
    """Pose list → (N,3) camera positions in world frame."""
    out = []
    for T in poses:
        T = np.asarray(T)
        if T.shape == (4,4):
            R, t = T[:3,:3], T[:3,3]
            out.append(-R.T @ t)
        elif T.shape == (3,):
            out.append(T)
    return np.array(out, dtype=np.float64)


def plot_trajectory_2d(
    estimated: list,
    ground_truth: Optional[list] = None,
    title: str = "Trajectory (Top View)",
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    Plot X-Z overhead view of estimated (and optionally ground-truth) trajectory.

    Parameters
    ----------
    estimated : list of 4×4 SE(3) poses
    ground_truth : list of 4×4 SE(3) poses | None
    save_path : str | None
    show : bool
    """
    est = _extract_positions(estimated)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # Colour estimated trajectory by frame index
    n = len(est)
    colours = cm.plasma(np.linspace(0, 1, n))
    for i in range(n - 1):
        ax.plot(est[i:i+2, 0], est[i:i+2, 2], color=colours[i], linewidth=1.5)
    ax.plot([], [], color=cm.plasma(0.5), label="Estimated", linewidth=2)

    if ground_truth:
        gt = _extract_positions(ground_truth)
        ax.plot(gt[:, 0], gt[:, 2], "--", color="#00ff99", linewidth=1.5,
                label="Ground Truth", alpha=0.8)

    # Start / end markers
    ax.scatter(est[0, 0], est[0, 2], c="#00ff99", s=100, zorder=5, label="Start")
    ax.scatter(est[-1, 0], est[-1, 2], c="#ff4466", s=100, marker="*", zorder=5, label="End")

    ax.set_xlabel("X (m)", color="white")
    ax.set_ylabel("Z (m)", color="white")
    ax.set_title(title, color="white", fontsize=14, fontweight="bold")
    ax.legend(facecolor="#1c2030", labelcolor="white", framealpha=0.7)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    ax.grid(True, color="#333", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_trajectory_3d(
    estimated: list,
    ground_truth: Optional[list] = None,
    title: str = "3D Trajectory",
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """3D trajectory plot (X, Y, Z axes)."""
    est = _extract_positions(estimated)

    fig = plt.figure(figsize=(12, 9))
    fig.patch.set_facecolor("#0d1117")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0d1117")

    n = len(est)
    colours = cm.plasma(np.linspace(0, 1, n))
    for i in range(n - 1):
        ax.plot(est[i:i+2, 0], est[i:i+2, 2], est[i:i+2, 1],
                color=colours[i], linewidth=1.2, alpha=0.9)

    if ground_truth:
        gt = _extract_positions(ground_truth)
        ax.plot(gt[:, 0], gt[:, 2], gt[:, 1], "--", color="#00ff99",
                linewidth=1.5, label="Ground Truth", alpha=0.7)

    ax.scatter(*[est[0, c] for c in [0,2,1]], c="#00ff99", s=80, label="Start")
    ax.scatter(*[est[-1, c] for c in [0,2,1]], c="#ff4466", s=120, marker="*", label="End")

    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#333")

    ax.set_xlabel("X (m)", color="white", labelpad=8)
    ax.set_ylabel("Z (m)", color="white", labelpad=8)
    ax.set_zlabel("Y (m)", color="white", labelpad=8)
    ax.set_title(title, color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1c2030", labelcolor="white", framealpha=0.7)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_error_over_time(
    ate_per_frame: List[float],
    rpe_per_frame: Optional[List[float]] = None,
    title: str = "Trajectory Error Over Time",
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """Plot ATE (and optionally RPE) error per frame."""
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    frames = np.arange(len(ate_per_frame))
    ax.plot(frames, ate_per_frame, color="#7c83ff", linewidth=1.5, label="ATE (m)")
    if rpe_per_frame:
        rpe_frames = np.arange(len(rpe_per_frame))
        ax.plot(rpe_frames, rpe_per_frame, color="#ff7c83", linewidth=1.5,
                label="RPE (m)", linestyle="--")

    ax.fill_between(frames, ate_per_frame, alpha=0.2, color="#7c83ff")
    ax.set_xlabel("Frame", color="white")
    ax.set_ylabel("Error (m)", color="white")
    ax.set_title(title, color="white", fontsize=14, fontweight="bold")
    ax.legend(facecolor="#1c2030", labelcolor="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444")
    ax.grid(True, color="#333", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)

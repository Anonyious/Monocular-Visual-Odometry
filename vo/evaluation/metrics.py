"""
Trajectory Evaluation: ATE and RPE
====================================

Mathematical Background
-----------------------

Absolute Trajectory Error (ATE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ATE measures the global consistency of the estimated trajectory against
ground truth by aligning the two trajectories first, then computing the
RMSE of the position error at each timestep.

Step 1 — Alignment via Umeyama (Sim(3) registration):
    Find the similarity transform (R*, t*, s*) that minimises:
        Σ_i ‖p̂_i - s*(R*·p_i + t*)‖²

    where p_i are estimated positions and p̂_i are ground-truth positions.

    This is the **Horn/Umeyama algorithm** (closed-form solution):
        μ_p   = mean(p)
        μ_p̂  = mean(p̂)
        Σ_pp̂ = (1/n) Σ (p̂_i - μ_p̂)(p_i - μ_p)ᵀ
        [U, S, Vᵀ] = SVD(Σ_pp̂)

        If det(U)·det(V) < 0: flip last column of V (reflection fix)
        R* = U diag(1,…,1,det(U)det(Vᵀ)) Vᵀ
        s* = (1/σ²_p) tr(S diag(1,…,det))     (scale)
        t* = μ_p̂ - s* R* μ_p

Step 2 — ATE computation:
    ATE = sqrt( (1/n) Σ_i ‖p̂_i - (s*R*p_i + t*)‖² )

    For monocular VO, the `scale` parameter is unknown; s* is estimated
    from the alignment (Sim(3) instead of SE(3) alignment).

Relative Pose Error (RPE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
RPE measures per-segment accuracy, averaging the error in relative
transformations over all pairs of frames separated by Δ steps:

    E_i = (T̂_{i+Δ}⁻¹ T̂_i)⁻¹ (T_{i+Δ}⁻¹ T_i)

    RPE_trans = sqrt( (1/m) Σ_i ‖trans(E_i)‖² )    (translation component)
    RPE_rot   = sqrt( (1/m) Σ_i ‖rot(E_i)‖² )       (rotation angle in rad)

where m = n - Δ is the number of valid pairs.

References
----------
• Sturm et al., "A Benchmark for the Evaluation of RGB-D SLAM Systems",
  IROS 2012  (ATE/RPE definition).
• Umeyama, "Least-Squares Estimation of Transformation Parameters Between
  Two Point Patterns", TPAMI 1991.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# ── Core Metrics ──────────────────────────────────────────────────────────────

def umeyama_alignment(
    estimated: np.ndarray,
    reference: np.ndarray,
    with_scale: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute the optimal Sim(3) (or SE(3)) alignment: aligned = s* R* p + t*.

    Parameters
    ----------
    estimated : np.ndarray, shape (N, 3)
        Estimated camera positions.
    reference : np.ndarray, shape (N, 3)
        Ground-truth camera positions.
    with_scale : bool
        If True, estimate and apply scale (Sim(3)).
        If False, constrain scale s=1 (SE(3)).

    Returns
    -------
    R : np.ndarray, shape (3, 3) — rotation
    t : np.ndarray, shape (3,)  — translation
    s : float                   — scale (1.0 if scale disabled)
    """
    assert estimated.shape == reference.shape, "Shape mismatch"
    n = len(estimated)

    mu_e = estimated.mean(axis=0)
    mu_r = reference.mean(axis=0)

    e_centered = estimated - mu_e
    r_centered = reference - mu_r

    sigma_e_sq = (e_centered ** 2).sum() / n        # variance of estimated
    cov = (r_centered.T @ e_centered) / n           # (3, 3)

    U, S, Vt = np.linalg.svd(cov)

    # Reflection fix: ensure det(UV^T) > 0
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, d])

    R = U @ D @ Vt
    s = (np.trace(np.diag(S) @ D) / sigma_e_sq) if with_scale else 1.0
    t = mu_r - s * R @ mu_e

    return R, t, s


def compute_ate(
    estimated_poses: list,
    gt_poses: list,
    align: bool = True,
    with_scale: bool = True,
) -> Tuple[float, float]:
    """
    Compute Absolute Trajectory Error (ATE).

    Parameters
    ----------
    estimated_poses : list of np.ndarray (4×4) or (N, 3)
        Estimated camera poses or positions.
    gt_poses : list of np.ndarray (4×4) or (N, 3)
        Ground-truth poses or positions.
    align : bool
        If True, align estimated to ground truth before computing error.
    with_scale : bool
        If True, estimate scale during alignment (Sim(3)).

    Returns
    -------
    ate_rmse : float — root-mean-square ATE in metres
    ate_mean : float — mean ATE in metres
    """
    est_pos = _extract_positions(estimated_poses)
    gt_pos = _extract_positions(gt_poses)

    n = min(len(est_pos), len(gt_pos))
    est_pos = est_pos[:n]
    gt_pos = gt_pos[:n]

    if align:
        R, t, s = umeyama_alignment(est_pos, gt_pos, with_scale=with_scale)
        aligned = (s * R @ est_pos.T).T + t
    else:
        aligned = est_pos

    errors = np.linalg.norm(gt_pos - aligned, axis=1)
    ate_rmse = float(np.sqrt(np.mean(errors ** 2)))
    ate_mean = float(np.mean(errors))

    return ate_rmse, ate_mean


def compute_rpe(
    estimated_poses: list,
    gt_poses: list,
    delta: int = 1,
    pose_relation: str = "translation",
) -> Tuple[float, float]:
    """
    Compute Relative Pose Error (RPE).

    Parameters
    ----------
    estimated_poses : list of 4×4 ndarray
    gt_poses : list of 4×4 ndarray
    delta : int
        Frame-step Δ for relative transform pairs.
    pose_relation : str
        'translation' (default) or 'rotation' or 'full'.

    Returns
    -------
    rpe_rmse : float
    rpe_mean : float
    """
    est = _extract_poses(estimated_poses)
    gt  = _extract_poses(gt_poses)

    n = min(len(est), len(gt))
    est = est[:n]
    gt  = gt[:n]

    errors = []
    for i in range(n - delta):
        # Ground-truth relative transform: Q̂_i = T̂_{i+Δ}⁻¹ T̂_i
        Q_gt = _se3_inv(gt[i + delta]) @ gt[i]
        # Estimated relative transform: Q_i = T_{i+Δ}⁻¹ T_i
        Q_est = _se3_inv(est[i + delta]) @ est[i]
        # Error: E_i = Q̂_i⁻¹ Q_i
        E = _se3_inv(Q_gt) @ Q_est

        if pose_relation == "translation":
            err = np.linalg.norm(E[:3, 3])
        elif pose_relation == "rotation":
            cos = np.clip((np.trace(E[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
            err = float(np.arccos(cos))
        else:
            t_err = np.linalg.norm(E[:3, 3])
            cos = np.clip((np.trace(E[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
            r_err = float(np.arccos(cos))
            err = (t_err + r_err) / 2.0

        errors.append(err)

    if not errors:
        return 0.0, 0.0

    arr = np.array(errors)
    return float(np.sqrt(np.mean(arr ** 2))), float(np.mean(arr))


# ── KITTI-Style Evaluation ────────────────────────────────────────────────────

def save_trajectory_kitti(poses: list, path: str) -> None:
    """
    Save estimated poses in KITTI format (12 values per line, row-major 3×4).

    Parameters
    ----------
    poses : list of 4×4 ndarray
    path : str
    """
    with open(path, "w") as f:
        for T in poses:
            row = T[:3, :].ravel()
            f.write(" ".join(f"{v:.8e}" for v in row) + "\n")


def load_trajectory_kitti(path: str) -> list:
    """Load a KITTI trajectory file, returning a list of 4×4 ndarray."""
    poses = []
    with open(path) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) != 12:
                continue
            T = np.eye(4, dtype=np.float64)
            T[:3, :] = np.array(vals).reshape(3, 4)
            poses.append(T)
    return poses


# ── Private Helpers ────────────────────────────────────────────────────────────

def _extract_positions(poses) -> np.ndarray:
    """Extract (N, 3) positions from a list of 4×4 SE(3) matrices or (N,3) array."""
    if isinstance(poses, np.ndarray) and poses.ndim == 2 and poses.shape[1] == 3:
        return poses
    out = []
    for p in poses:
        p = np.asarray(p)
        if p.shape == (4, 4):
            # Camera position in world = -Rᵀ t
            R = p[:3, :3]
            t = p[:3, 3]
            out.append(-R.T @ t)
        elif p.shape == (3,):
            out.append(p)
        else:
            raise ValueError(f"Unexpected pose shape: {p.shape}")
    return np.array(out, dtype=np.float64)


def _extract_poses(poses) -> list:
    """Return a list of 4×4 ndarray from various input formats."""
    result = []
    for p in poses:
        p = np.asarray(p, dtype=np.float64)
        if p.shape == (4, 4):
            result.append(p)
        elif p.shape == (3, 4):
            T = np.eye(4)
            T[:3, :] = p
            result.append(T)
        else:
            raise ValueError(f"Unexpected pose shape: {p.shape}")
    return result


def _se3_inv(T: np.ndarray) -> np.ndarray:
    """Fast SE(3) inverse."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv

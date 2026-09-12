"""
Unit Tests
===========
Run with:  pytest tests/ -v
"""
from __future__ import annotations

import numpy as np
import pytest

from vo.camera import Camera
from vo.ransac import RANSAC
from vo.pose_graph import PoseGraph, se3_exp, se3_log, se3_inverse, se3_compose
from vo.evaluation.metrics import compute_ate, compute_rpe, umeyama_alignment


# ── Camera Tests ──────────────────────────────────────────────────────────────

class TestCamera:
    def setup_method(self):
        K = np.array([[718.856, 0, 607.193],
                      [0, 718.856, 185.216],
                      [0, 0, 1.0]], dtype=np.float64)
        self.cam = Camera(K=K)

    def test_normalize_project_roundtrip(self):
        """Project 3D point → pixel → normalize → should recover normalised coords."""
        X = np.array([[5.0, 0.5, 10.0]])   # world coords
        R = np.eye(3)
        t = np.zeros(3)
        px = self.cam.project(X, R, t)
        norm = self.cam.normalize(px)

        # Expected normalised: X/Z, Y/Z
        expected = X[:, :2] / X[:, 2:]
        np.testing.assert_allclose(norm, expected, atol=1e-4)

    def test_reprojection_error_zero(self):
        """Reprojection error should be zero when using exact projection."""
        X = np.array([[1.0, -0.5, 8.0], [3.0, 2.0, 12.0]])
        R = np.eye(3)
        t = np.zeros(3)
        px = self.cam.project(X, R, t)
        errors = self.cam.reprojection_error(X, px, R, t)
        np.testing.assert_allclose(errors, 0.0, atol=1e-4)

    def test_k_shape(self):
        assert self.cam.K.shape == (3, 3)

    def test_invalid_K_raises(self):
        with pytest.raises(ValueError):
            Camera(K=np.eye(4))


# ── RANSAC Tests ──────────────────────────────────────────────────────────────

class TestRANSAC:
    def test_line_fitting(self):
        """Fit y = 2x + 1 with 30% outliers."""
        rng = np.random.default_rng(42)
        x = np.linspace(0, 10, 100)
        y_true = 2.0 * x + 1.0
        y = y_true + rng.normal(0, 0.05, 100)  # inlier noise
        # Add 30 outliers
        outlier_idx = rng.choice(100, 30, replace=False)
        y[outlier_idx] += rng.uniform(-5, 5, 30)

        data = np.column_stack([x, y])

        def model_fn(sample):
            x1, y1 = sample[0]
            x2, y2 = sample[1]
            if abs(x2 - x1) < 1e-10:
                return None
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1
            return (m, b)

        def residual_fn(model, data):
            m, b = model
            return np.abs(data[:, 1] - (m * data[:, 0] + b))

        ransac = RANSAC(model_fn, residual_fn, min_samples=2, threshold=0.2, confidence=0.99)
        model, inliers = ransac.fit(data, rng=rng)

        assert model is not None
        m, b = model
        assert abs(m - 2.0) < 0.2, f"Slope should be ~2.0, got {m}"
        assert abs(b - 1.0) < 0.2, f"Intercept should be ~1.0, got {b}"
        assert inliers.sum() >= 60  # should keep most inliers

    def test_required_iterations_formula(self):
        """Verify the RANSAC iteration count formula N = log(1-p)/log(1-(1-ε)^s)."""
        import math
        ransac = RANSAC(lambda x: x, lambda m, d: d, min_samples=8, threshold=1.0, confidence=0.99)
        epsilon = 0.5  # 50% outliers
        s = 8
        N_expected = math.ceil(math.log(1 - 0.99) / math.log(1 - (1 - epsilon) ** s))
        N_computed = ransac._required_iterations(epsilon)
        assert N_computed == N_expected


# ── SE(3) Tests ───────────────────────────────────────────────────────────────

class TestSE3:
    def test_exp_log_roundtrip(self):
        """exp(log(T)) should recover T."""
        # Random SE(3) with small rotation
        xi = np.array([0.1, -0.2, 0.05, 1.0, -0.5, 2.0])
        T = se3_exp(xi)
        xi_recovered = se3_log(T)
        np.testing.assert_allclose(xi, xi_recovered, atol=1e-8, rtol=1e-6)

    def test_inverse(self):
        """T · T⁻¹ should be identity."""
        xi = np.array([0.3, 0.1, -0.2, 2.0, 1.0, -1.0])
        T = se3_exp(xi)
        T_inv = se3_inverse(T)
        product = T @ T_inv
        np.testing.assert_allclose(product, np.eye(4), atol=1e-10)

    def test_compose(self):
        """T1 · T2 should equal matrix multiply."""
        xi1 = np.array([0.1, 0.0, 0.0, 1.0, 0.0, 0.0])
        xi2 = np.array([0.0, 0.1, 0.0, 0.0, 1.0, 0.0])
        T1 = se3_exp(xi1)
        T2 = se3_exp(xi2)
        T12_compose = se3_compose(T1, T2)
        T12_matmul = T1 @ T2
        np.testing.assert_allclose(T12_compose, T12_matmul, atol=1e-12)

    def test_pose_graph_edge_error_zero(self):
        """Edge error should be zero when graph is consistent."""
        g = PoseGraph()
        g.add_node(0, np.eye(3), np.zeros(3), fixed=True)

        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        t = np.array([1.0, 0.0, 0.0])
        g.add_node(1, R, t)
        g.add_edge(0, 1, R_ij=R, t_ij=t)

        error = g.compute_edge_error(g._edges[0])
        np.testing.assert_allclose(error, np.zeros(6), atol=1e-8)

    def test_pose_graph_edge_error_non_degenerate(self):
        """
        Edge error should be zero for a non-degenerate graph.

        This test uses a rotated first node to ensure the i→j convention
        is correct (the old implementation only passed for the degenerate
        case where node 0 was at identity).
        """
        g = PoseGraph()
        R0 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        t0 = np.array([0.5, 0.0, 0.0])
        g.add_node(0, R0, t0, fixed=True)

        R1 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)
        t1 = np.array([1.0, 0.5, 0.0])
        g.add_node(1, R1, t1)

        # Relative transform from node 0 to node 1 (i→j):
        # T_ij = T_j @ T_i^{-1}  (maps camera_i → world → camera_j)
        T0 = np.eye(4)
        T0[:3, :3] = R0
        T0[:3, 3] = t0
        T1 = np.eye(4)
        T1[:3, :3] = R1
        T1[:3, 3] = t1
        T_ij_expected = T1 @ np.linalg.inv(T0)

        g.add_edge(0, 1, R_ij=T_ij_expected[:3, :3], t_ij=T_ij_expected[:3, 3])

        error = g.compute_edge_error(g._edges[0])
        np.testing.assert_allclose(error, np.zeros(6), atol=1e-8)

# ── Evaluation Metrics Tests ──────────────────────────────────────────────────

class TestMetrics:
    def _make_linear_traj(self, n=100):
        """Generate a straight-line trajectory with identity rotations."""
        poses = []
        for i in range(n):
            T = np.eye(4)
            T[0, 3] = float(i)   # move along X
            poses.append(T)
        return poses

    def test_ate_zero_on_perfect_estimate(self):
        """ATE should be zero when estimate equals ground truth."""
        gt = self._make_linear_traj(50)
        ate_rmse, ate_mean = compute_ate(gt, gt, align=False)
        assert ate_rmse < 1e-6, f"Expected ~0 ATE, got {ate_rmse}"

    def test_rpe_zero_on_perfect_estimate(self):
        """RPE should be zero when estimate equals ground truth."""
        gt = self._make_linear_traj(50)
        rpe_rmse, rpe_mean = compute_rpe(gt, gt, delta=1)
        assert rpe_rmse < 1e-6, f"Expected ~0 RPE, got {rpe_rmse}"

    def test_umeyama_recovers_known_transform(self):
        """Umeyama should recover R, t, s used to perturb reference."""
        from scipy.spatial.transform import Rotation
        rng = np.random.default_rng(0)
        pts = rng.standard_normal((30, 3))

        R_true = Rotation.from_euler("z", 30, degrees=True).as_matrix()
        t_true = np.array([1.0, -2.0, 0.5])
        s_true = 2.5

        pts_ref = (s_true * R_true @ pts.T).T + t_true

        R_est, t_est, s_est = umeyama_alignment(pts, pts_ref, with_scale=True)

        np.testing.assert_allclose(s_est, s_true, atol=1e-6)
        np.testing.assert_allclose(R_est, R_true, atol=1e-6)
        np.testing.assert_allclose(t_est, t_true, atol=1e-6)

    def test_ate_detected_after_drift(self):
        """Adding systematic drift to estimated trajectory should raise ATE."""
        gt = self._make_linear_traj(100)
        drifted = []
        for i, T in enumerate(gt):
            T_d = T.copy()
            T_d[0, 3] += 0.1 * i   # linear drift
            drifted.append(T_d)

        ate_rmse, _ = compute_ate(drifted, gt, align=True, with_scale=False)
        assert ate_rmse > 1.0, f"Expected significant ATE due to drift, got {ate_rmse:.3f}"

"""
RANSAC Ground-Plane Scale Recovery
====================================

Mathematical Background
-----------------------

Monocular cameras cannot recover metric scale from images alone (the scene is
unobservable up to a similarity transformation). We anchor scale by exploiting
the known physical height of the camera above the ground plane.

Ground-Plane Model
~~~~~~~~~~~~~~~~~~
A plane in 3D space is parameterised by its unit normal n ∈ ℝ³ and offset d:

    nᵀ X + d = 0   for all points X on the plane.

The signed distance from the camera origin (0,0,0) to the plane is:

    dist = d / ‖n‖ = d    (since ‖n‖ = 1)

RANSAC Plane Fitting
~~~~~~~~~~~~~~~~~~~~
Given N triangulated 3D points {X_i}, a minimal sample of 3 non-collinear
points uniquely determines a plane via:

    n = (X₂ - X₁) × (X₃ - X₁) / ‖…‖
    d = -nᵀ X₁

Inlier test: |nᵀ X_i + d| < τ  (τ = inlier distance threshold in metres)

The inlier set that maximises point count is the ground plane.

Scale Recovery
~~~~~~~~~~~~~~
The camera is mounted at a known height h above the ground. In the camera
coordinate frame, the KITTI ground plane is approximately at Y = h (since the
camera Y-axis points downward).

The triangulated points have an unknown scale s — they are correct *relative*
to each other but scaled by s. The true metric plane is at distance h from
the camera origin. The estimated plane is at distance d̂. Therefore:

    s = h / |d̂|

Temporal Smoothing
~~~~~~~~~~~~~~~~~~
Raw scale estimates are noisy. We apply a 1-D exponential moving average:

    s_smooth(k) = α * s_raw(k) + (1 - α) * s_smooth(k-1)

with α = 0.3 (favour the running estimate over the current noisy sample).

References
----------
• Kitt et al., "Visual Odometry Based on Stereo Image Sequences with RANSAC-
  Based Outlier Rejection Scheme", IEEE IV 2010.
• Scaramuzza & Fraundorfer, "Visual Odometry: Part I", IEEE RA-M 2011.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Defaults
_RANSAC_THRESHOLD_M = 0.10      # 10 cm ground-plane inlier tolerance
_RANSAC_MAX_ITER    = 100
_RANSAC_CONFIDENCE  = 0.99
_EMA_ALPHA          = 0.30       # exponential moving-average weight for new samples
_MIN_GROUND_POINTS  = 20         # minimum inliers to trust a GP estimate
_SCALE_CLIP_LOW     = 0.05       # sanity clip: scale < 0.05 → reject
_SCALE_CLIP_HIGH    = 20.0       # sanity clip: scale > 20  → reject


class GroundPlaneScaleRecovery:
    """
    Estimates metric scale by fitting a ground plane to triangulated 3-D points
    via RANSAC and comparing the estimated camera height to the known physical
    height.

    Parameters
    ----------
    camera_height : float
        Known camera height above the ground in metres.
    ransac_threshold : float
        RANSAC inlier distance threshold in metres.
    ema_alpha : float
        Exponential moving-average coefficient for temporal smoothing.
    min_inliers : int
        Minimum number of inliers to accept a plane fit as the ground plane.
    """

    def __init__(
        self,
        camera_height: float = 1.65,
        ransac_threshold: float = _RANSAC_THRESHOLD_M,
        ema_alpha: float = _EMA_ALPHA,
        min_inliers: int = _MIN_GROUND_POINTS,
    ) -> None:
        self.camera_height = camera_height
        self.ransac_threshold = ransac_threshold
        self.ema_alpha = ema_alpha
        self.min_inliers = min_inliers

        self._scale_ema: float = 1.0     # current smoothed scale estimate
        self._n_updates: int = 0         # number of successful scale updates

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def scale(self) -> float:
        """Current smoothed scale estimate."""
        return self._scale_ema

    def update(self, points_3d: np.ndarray) -> float:
        """
        Update the scale estimate from newly triangulated 3-D points.

        Parameters
        ----------
        points_3d : np.ndarray, shape (N, 3)
            Triangulated 3-D points in the **camera frame** of the reference
            view. NaN values are ignored.

        Returns
        -------
        float
            Updated smoothed scale estimate. Returns the previous estimate if
            the ground plane cannot be reliably detected.
        """
        # Filter NaN / points behind camera
        valid = points_3d[
            ~np.any(np.isnan(points_3d), axis=1) & (points_3d[:, 2] > 0)
        ]

        if len(valid) < 3:
            logger.debug("GroundPlaneScale: too few valid points (%d)", len(valid))
            return self._scale_ema

        result = self._ransac_ground_plane(valid)
        if result is None:
            logger.debug("GroundPlaneScale: RANSAC plane fit failed")
            return self._scale_ema

        normal, offset, n_inliers = result

        if n_inliers < self.min_inliers:
            logger.debug(
                "GroundPlaneScale: only %d inliers (need %d)",
                n_inliers, self.min_inliers,
            )
            return self._scale_ema

        # Estimated camera height from triangulated (unscaled) points
        # Distance from camera origin to plane: |d| where nᵀX + d = 0
        estimated_height = abs(offset)
        if estimated_height < 1e-4:
            return self._scale_ema

        raw_scale = self.camera_height / estimated_height

        # Sanity check
        if not (_SCALE_CLIP_LOW < raw_scale < _SCALE_CLIP_HIGH):
            logger.debug(
                "GroundPlaneScale: raw scale %.3f out of sane range [%.2f, %.1f]",
                raw_scale, _SCALE_CLIP_LOW, _SCALE_CLIP_HIGH,
            )
            return self._scale_ema

        # Exponential moving average update
        if self._n_updates == 0:
            self._scale_ema = raw_scale
        else:
            self._scale_ema = (
                self.ema_alpha * raw_scale
                + (1.0 - self.ema_alpha) * self._scale_ema
            )

        self._n_updates += 1
        logger.debug(
            "GroundPlaneScale: raw=%.3f ema=%.3f (inliers=%d)",
            raw_scale, self._scale_ema, n_inliers,
        )
        return self._scale_ema

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _ransac_ground_plane(
        self, pts: np.ndarray
    ) -> Optional[Tuple[np.ndarray, float, int]]:
        """
        RANSAC ground-plane fitting.

        The ground plane is identified as the one whose normal is most aligned
        with the camera Y-axis (downward direction in KITTI convention).

        Returns
        -------
        (normal, offset, n_inliers) or None on failure.
        normal : np.ndarray (3,) — unit plane normal
        offset : float           — signed plane-origin distance (d in nᵀX+d=0)
        n_inliers : int
        """
        n = len(pts)
        best_normal: Optional[np.ndarray] = None
        best_offset: float = 0.0
        best_inliers: int = 0

        # Adaptive iteration count using RANSAC formula
        max_iter = _RANSAC_MAX_ITER
        rng = np.random.default_rng(seed=42)

        for iteration in range(max_iter):
            # Minimal sample: 3 points
            idx = rng.choice(n, size=3, replace=False)
            p1, p2, p3 = pts[idx]

            # Compute plane normal
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm_ = np.linalg.norm(normal)
            if norm_ < 1e-10:
                continue   # degenerate (collinear) sample
            normal = normal / norm_
            offset = float(-normal @ p1)

            # Count inliers
            distances = np.abs(pts @ normal + offset)
            inlier_mask = distances < self.ransac_threshold
            n_inliers = int(inlier_mask.sum())

            if n_inliers > best_inliers:
                best_normal = normal
                best_offset = offset
                best_inliers = n_inliers

                # Adaptive update of max_iter
                inlier_ratio = n_inliers / n
                if inlier_ratio > 1e-6:
                    eps = 1.0 - inlier_ratio
                    eps = np.clip(eps, 1e-6, 1.0 - 1e-6)
                    p_good = (1.0 - eps) ** 3
                    if p_good > 1e-10:
                        new_max = (
                            np.log(1.0 - _RANSAC_CONFIDENCE)
                            / np.log(1.0 - p_good + 1e-15)
                        )
                        max_iter = min(int(np.ceil(new_max)), _RANSAC_MAX_ITER)

        if best_normal is None:
            return None

        # Refit on inliers for better accuracy
        distances = np.abs(pts @ best_normal + best_offset)
        inlier_pts = pts[distances < self.ransac_threshold]
        if len(inlier_pts) >= 3:
            best_normal, best_offset = self._fit_plane_least_squares(inlier_pts)
            best_inliers = len(inlier_pts)

        # Prefer normals pointing roughly in the +Y or -Y direction (camera up/down)
        # In KITTI, camera Y-axis points down, so the ground-plane normal should
        # have a significant |normal[1]| component. We don't enforce this strictly
        # to stay dataset-agnostic, but log it.
        if abs(best_normal[1]) < 0.3:
            logger.debug(
                "GroundPlaneScale: fitted normal %s has small Y component — "
                "may not be the ground plane", best_normal.round(3)
            )

        return best_normal, best_offset, best_inliers

    @staticmethod
    def _fit_plane_least_squares(
        pts: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Fit a plane to pts via PCA (SVD of zero-mean point set).

        The plane normal is the right singular vector of the zero-mean point
        matrix corresponding to the smallest singular value.
        """
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        _, _, Vt = np.linalg.svd(centered)
        normal = Vt[-1]                      # smallest singular value direction
        normal /= np.linalg.norm(normal) + 1e-12
        offset = float(-normal @ centroid)
        return normal, offset

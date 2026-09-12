"""
Local 3D Landmark Map with PnP-based Pose Refinement
=====================================================

After bootstrapping the scale from pair-wise essential matrix estimation,
we maintain a sliding-window map of 3D landmarks.  For each new keyframe,
instead of re-estimating the pose from scratch via the essential matrix,
we use the *Perspective-n-Point (PnP)* problem:

    Given N 3D map points  X_i  and their corresponding 2D observations
    x_i = K[R|t]X_i in the new frame, find R and t.

This is more stable than the 2-frame essential matrix approach because:
  • It uses historical 3D points (triangulated from multiple views).
  • The 3D point positions are known, removing the scale ambiguity.
  • More observations → lower variance.

PnP via Direct Linear Transform (DLT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For N ≥ 6 correspondences, the DLT stacks rows:

    [ -xᵢ Pᵀ₃ + P₁ᵀ ]
    [ -yᵢ Pᵀ₃ + P₂ᵀ ] X_i = 0

where P₁,P₂,P₃ are the rows of P = K[R|t].  Solved as a 2N×12 system
via SVD.  The minimal case (n=3) uses the EPnP algorithm (Lepetit 2009)
implemented by OpenCV.

Keyframing Strategy
~~~~~~~~~~~~~~~~~~~~
Not every frame is a keyframe.  We insert a new keyframe when:
  1. Feature tracking degrades below a threshold (fewer than min_tracked).
  2. The median parallax since the last keyframe exceeds a threshold.
  3. A fixed number of frames have elapsed (fallback).

References
----------
• Lepetit, Moreno-Noguer & Fua, "EPnP: An Accurate O(n) Solution to
  the PnP Problem", IJCV 2009.
• Hartley & Zisserman, §7.1 (DLT), §11.1 (basic PnP).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .camera import Camera

logger = logging.getLogger(__name__)

_PNP_MIN_POINTS = 6
_TRIANGULATE_MAX_REPROJ_PX = 2.0   # pixels – reject noisy triangulations
_MAP_MAX_POINTS = 3000             # prune map when size exceeds this


@dataclass
class Landmark:
    """A single 3D map point."""
    point_id: int
    position: np.ndarray          # shape (3,) – 3D coords in world frame
    descriptor: Optional[np.ndarray] = None  # (32,) ORB descriptor
    observations: List[Tuple[int, int]] = field(default_factory=list)  # (frame_id, kp_idx)
    times_matched: int = 0


class LocalMap:
    """
    Maintains a sliding set of 3D landmarks triangulated from keyframes.

    Provides two services:
      1. ``add_keyframe``  – triangulate new 3D points from a frame pair.
      2. ``pnp_localize``  – estimate a new frame's pose from existing map points.

    Parameters
    ----------
    camera : Camera
        Calibrated camera (intrinsics + distortion).
    max_points : int
        Maximum map size before pruning.
    min_parallax_deg : float
        Minimum angular parallax (degrees) required to accept triangulation.
    max_reproj_err : float
        Maximum reprojection error (pixels) to accept a triangulated point.
    """

    def __init__(
        self,
        camera: Camera,
        max_points: int = _MAP_MAX_POINTS,
        min_parallax_deg: float = 1.0,
        max_reproj_err: float = _TRIANGULATE_MAX_REPROJ_PX,
    ) -> None:
        self.camera = camera
        self.max_points = max_points
        self.min_parallax_deg = min_parallax_deg
        self.max_reproj_err = max_reproj_err

        self._landmarks: Dict[int, Landmark] = {}
        self._next_id: int = 0
        self._keyframes: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}  # frame_id → (R, t)

    # ── Public API ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._landmarks)

    def add_keyframe(
        self,
        frame_id: int,
        R: np.ndarray,
        t: np.ndarray,
        keypoints: np.ndarray,
        descriptors: np.ndarray,
        prev_frame_id: int,
        pts_prev: np.ndarray,
        pts_curr: np.ndarray,
    ) -> int:
        """
        Triangulate new 3D points from a matched frame pair and add to map.

        Parameters
        ----------
        frame_id : int
            ID of the current (new) frame.
        R, t : ndarray
            Pose of *frame_id* (camera ← world).
        keypoints : (N, 2)
            All keypoints in the current frame.
        descriptors : (N, 32)
            ORB descriptors for keypoints.
        prev_frame_id : int
            ID of the previous keyframe.
        pts_prev : (M, 2)
            Tracked points in the previous keyframe.
        pts_curr : (M, 2)
            Corresponding tracked points in the current frame.

        Returns
        -------
        int
            Number of new landmarks added.
        """
        self._keyframes[frame_id] = (R.copy(), t.copy())

        if prev_frame_id not in self._keyframes:
            return 0

        R_prev, t_prev = self._keyframes[prev_frame_id]

        # Build projection matrices
        K = self.camera.K
        P_prev = K @ np.hstack([R_prev, t_prev.reshape(3, 1)])
        P_curr = K @ np.hstack([R, t.reshape(3, 1)])

        # Triangulate
        pts_prev_f = pts_prev.T.astype(np.float32)
        pts_curr_f = pts_curr.T.astype(np.float32)
        pts4d = cv2.triangulatePoints(P_prev, P_curr, pts_prev_f, pts_curr_f)
        pts3d = (pts4d[:3] / pts4d[3]).T   # (M, 3)

        n_added = 0
<<<<<<< HEAD
        for i, X in enumerate(pts3d):
            # Chirality check
            if X[2] <= 0:
                continue
            X_cam2 = R @ X + t
            if X_cam2[2] <= 0:
                continue

            # Reprojection error filter
            reproj_dist = self._reprojection_error(X, pts_curr[i], R, t)
            if reproj_dist > self.max_reproj_err:
                continue

            # Parallax check (angular)
            if not self._sufficient_parallax(X, t_prev, t):
=======
        # cv2.triangulatePoints(P_prev, P_curr, ...) returns points in the
        # *previous keyframe's camera frame*.  All downstream geometric
        # checks (chirality, reprojection, parallax) and the stored landmark
        # position must be expressed in the **world frame**, so convert first.
        R_prev_inv = R_prev.T
        for i, X in enumerate(pts3d):
            # Convert triangulated point (prev-camera frame) → world frame
            X_world = R_prev_inv @ (X - t_prev)

            # Chirality check (world → current camera frame)
            X_cam2 = R @ X_world + t
            if X_cam2[2] <= 0:
                continue

            # Reprojection error filter (world → current camera frame)
            reproj_dist = self._reprojection_error(X_world, pts_curr[i], R, t)
            if reproj_dist > self.max_reproj_err:
                continue

            # Parallax check (angular) — uses world-frame positions
            if not self._sufficient_parallax(X_world, t_prev, t):
>>>>>>> origin/master
                continue

            # Find best matching descriptor
            desc = None
            if descriptors is not None and len(descriptors) > 0:
                nn_idx = self._nearest_keypoint(pts_curr[i], keypoints)
                if nn_idx is not None:
                    desc = descriptors[nn_idx]

            lm = Landmark(
                point_id=self._next_id,
<<<<<<< HEAD
                position=X.copy(),
=======
                position=X_world.copy(),
>>>>>>> origin/master
                descriptor=desc,
                observations=[(prev_frame_id, i), (frame_id, i)],
            )
            self._landmarks[self._next_id] = lm
            self._next_id += 1
            n_added += 1

        logger.debug("add_keyframe %d: +%d landmarks (total %d)", frame_id, n_added, len(self))
        self.prune()
        return n_added

    def pnp_localize(
        self,
        keypoints: np.ndarray,
        descriptors: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Estimate the current frame's pose by matching to map landmarks.

        Matches the current frame's ORB descriptors against stored landmark
        descriptors, then runs EPnP+RANSAC to recover (R, t).

        Parameters
        ----------
        keypoints : (N, 2)
            Keypoints in the current frame.
        descriptors : (N, 32)
            Descriptors for those keypoints.

        Returns
        -------
        (R, t, inlier_mask) or None if PnP fails.
        """
        pts3d, pts2d = self._get_map_correspondences(keypoints, descriptors)

        if len(pts3d) < _PNP_MIN_POINTS:
            logger.debug("PnP: only %d correspondences, need %d", len(pts3d), _PNP_MIN_POINTS)
            return None

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d.astype(np.float64),
            pts2d.astype(np.float64),
            self.camera.K,
            self.camera.dist_coeffs,
            iterationsCount=200,
            reprojectionError=2.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_EPNP,
        )

        if not success or inliers is None or len(inliers) < _PNP_MIN_POINTS:
            return None

        R, _ = cv2.Rodrigues(rvec)
        t = tvec.ravel()

        inlier_mask = np.zeros(len(pts3d), dtype=bool)
        inlier_mask[inliers.ravel()] = True

        return R, t, inlier_mask

    def prune(self) -> None:
        """
        Remove the oldest / least-matched landmarks to cap map size.

        Strategy: discard landmarks with fewest observations first,
        then oldest by insertion order, until size ≤ max_points.
        """
        if len(self._landmarks) <= self.max_points:
            return

        n_remove = len(self._landmarks) - self.max_points
        # Sort by times_matched ascending, then by id ascending (oldest first)
        sorted_ids = sorted(
            self._landmarks,
            key=lambda pid: (self._landmarks[pid].times_matched, pid),
        )
        for pid in sorted_ids[:n_remove]:
            del self._landmarks[pid]

        logger.debug("Map pruned to %d landmarks", len(self._landmarks))

    @property
    def points(self) -> np.ndarray:
        """All map point positions as (N, 3) array."""
        if not self._landmarks:
            return np.empty((0, 3))
        return np.stack([lm.position for lm in self._landmarks.values()])

    # ── Private Helpers ──────────────────────────────────────────────────────

    def _get_map_correspondences(
        self,
        keypoints: np.ndarray,
        descriptors: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Match current frame descriptors against map landmark descriptors."""
        map_ids = [
            pid for pid, lm in self._landmarks.items()
            if lm.descriptor is not None
        ]
        if not map_ids or descriptors is None or len(descriptors) == 0:
            return np.empty((0, 3)), np.empty((0, 2))

        map_descs = np.stack([self._landmarks[pid].descriptor for pid in map_ids])
        map_pts = np.stack([self._landmarks[pid].position for pid in map_ids])

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        raw = matcher.knnMatch(map_descs, descriptors.astype(np.uint8), k=2)

        pts3d, pts2d = [], []
        for pair in raw:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                pts3d.append(map_pts[m.queryIdx])
                pts2d.append(keypoints[m.trainIdx])
                self._landmarks[map_ids[m.queryIdx]].times_matched += 1

        if not pts3d:
            return np.empty((0, 3)), np.empty((0, 2))
        return np.array(pts3d), np.array(pts2d)

    def _reprojection_error(
        self, X: np.ndarray, obs: np.ndarray, R: np.ndarray, t: np.ndarray
    ) -> float:
        """Euclidean reprojection error in pixels for a single 3D/2D pair."""
        projected = self.camera.project(X[np.newaxis], R, t)[0]
        return float(np.linalg.norm(projected - obs))

    def _sufficient_parallax(
        self, X: np.ndarray, t_prev: np.ndarray, t_curr: np.ndarray
    ) -> bool:
        """Check angular parallax between two camera positions."""
        ray_prev = X - t_prev
        ray_curr = X - t_curr
        cos_angle = np.dot(ray_prev, ray_curr) / (
            np.linalg.norm(ray_prev) * np.linalg.norm(ray_curr) + 1e-10
        )
        angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        return angle_deg > self.min_parallax_deg

    @staticmethod
    def _nearest_keypoint(
        pt: np.ndarray, keypoints: np.ndarray, max_dist_px: float = 5.0
    ) -> Optional[int]:
        """Index of the keypoint nearest to pt within max_dist_px pixels."""
        if len(keypoints) == 0:
            return None
        dists = np.linalg.norm(keypoints - pt, axis=1)
        idx = int(np.argmin(dists))
        return idx if dists[idx] < max_dist_px else None

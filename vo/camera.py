"""
Pinhole Camera Model with Radial-Tangential Lens Distortion
============================================================

Mathematical Background
-----------------------
The pinhole camera model relates a 3D world point X_w to a 2D image point x
through a sequence of coordinate transformations:

  1. World → Camera frame:
         X_c = R X_w + t

  2. Perspective projection (normalised image plane at Z = 1):
         x_n = (X_c / Z_c) = [X_c/Z_c, Y_c/Z_c]ᵀ

  3. Lens distortion (Brown-Conrady model):
         r² = x_n² + y_n²
         k  = 1 + k1·r² + k2·r⁴ + k3·r⁶
         x_d = k·x_n + [2p1·x_n·y_n + p2·(r²+2·x_n²)]
                      [p1·(r²+2·y_n²) + 2p2·x_n·y_n  ]

  4. Pixel projection through intrinsic matrix K:
         u = fx·x_d + cx
         v = fy·y_d + cy

     where K = [[fx, 0, cx],
                [ 0,fy, cy],
                [ 0,  0,  1]]

  Back-projection (pixel → normalised ray) inverts step 4 then 3:
         x_n = K⁻¹ · [u, v, 1]ᵀ   (no distortion → divide by Z)

References
----------
• Hartley & Zisserman, "Multiple View Geometry in Computer Vision", Ch. 6.
• Zhang, "A Flexible New Technique for Camera Calibration", TPAMI 2000.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class Camera:
    """
    Pinhole camera with optional Brown-Conrady distortion.

    Attributes
    ----------
    K : np.ndarray, shape (3, 3)
        Intrinsic matrix  [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].
    dist_coeffs : np.ndarray, shape (4,) | (5,) | (8,)
        Distortion coefficients [k1, k2, p1, p2, k3, ...].
        Pass np.zeros(5) for pre-rectified images (e.g. KITTI).
    """

    K: np.ndarray
    dist_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(5))

    def __post_init__(self) -> None:
        self.K = np.asarray(self.K, dtype=np.float64)
        self.dist_coeffs = np.asarray(self.dist_coeffs, dtype=np.float64)
        if self.K.shape != (3, 3):
            raise ValueError(f"K must be shape (3,3), got {self.K.shape}")
        # Cache K inverse for fast normalisation
        self._K_inv: np.ndarray = np.linalg.inv(self.K)

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    # ── Core Operations ───────────────────────────────────────────────────────

    def undistort(self, image: np.ndarray) -> np.ndarray:
        """
        Remove lens distortion from *image* using the Brown-Conrady model.

        Applies  cv2.undistort  which iteratively inverts the distortion
        polynomial to find the undistorted pixel for each distorted pixel
        via bilinear interpolation.

        Parameters
        ----------
        image : np.ndarray
            Input image (any channel count).

        Returns
        -------
        np.ndarray
            Undistorted image of the same shape as *image*.
        """
        if np.allclose(self.dist_coeffs, 0):
            return image  # Already rectified (KITTI case)
        return cv2.undistort(image, self.K, self.dist_coeffs)

    def normalize(self, pixel_coords: np.ndarray) -> np.ndarray:
        """
        Back-project pixel coordinates to normalised image coordinates.

        Computes  x_n = K⁻¹ · [u, v, 1]ᵀ  →  (x, y) on the Z=1 plane.

        This is the "pinhole" ideal — distortion correction should be handled
        by  undistort()  at the image level before calling this.

        Parameters
        ----------
        pixel_coords : np.ndarray, shape (N, 2)
            Pixel coordinates (u, v).

        Returns
        -------
        np.ndarray, shape (N, 2)
            Normalised coordinates (x, y) = (u', v') at Z = 1.
        """
        pts = np.asarray(pixel_coords, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts[np.newaxis, :]

        # Homogeneous: [u, v, 1]
        ones = np.ones((len(pts), 1), dtype=np.float64)
        pts_h = np.hstack([pts, ones])          # (N, 3)

        # x_n = K⁻¹ · pᵀ → shape (3, N), then transpose
        normalised_h = (self._K_inv @ pts_h.T).T   # (N, 3)
        return normalised_h[:, :2]  # drop homogeneous component (== 1)

    def project(
        self,
        points_3d: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
    ) -> np.ndarray:
        """
        Project 3D world points to pixel coordinates.

        Implements the full pipeline:
            X_c = R·X_w + t
            x_n = X_c[:2] / X_c[2]           (perspective divide)
            [u,v,1]ᵀ = K · [x_n, 1]ᵀ         (pixel projection)

        Uses cv2.projectPoints so distortion is applied if dist_coeffs is
        non-zero.

        Parameters
        ----------
        points_3d : np.ndarray, shape (N, 3)
            3D points in the world coordinate frame.
        R : np.ndarray, shape (3, 3)
            Rotation matrix (camera ← world).
        t : np.ndarray, shape (3,) | (3, 1)
            Translation vector (camera ← world).

        Returns
        -------
        np.ndarray, shape (N, 2)
            Pixel coordinates (u, v).
        """
        pts = np.asarray(points_3d, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts[np.newaxis, :]

        R_vec, _ = cv2.Rodrigues(R)
        projected, _ = cv2.projectPoints(
            pts.reshape(-1, 1, 3),
            R_vec,
            t.reshape(3, 1),
            self.K,
            self.dist_coeffs,
        )
        return projected.reshape(-1, 2)

    def project_point(
        self,
        X_c: np.ndarray,
    ) -> np.ndarray:
        """
        Project a single 3D point in the **camera** frame to pixel coords.

        Bypasses rotation/translation — assumes X_c is already in camera frame.

        Parameters
        ----------
        X_c : np.ndarray, shape (3,)
            3D point in camera coordinates.

        Returns
        -------
        np.ndarray, shape (2,)
            Pixel coordinates (u, v).
        """
        X_c = np.asarray(X_c, dtype=np.float64).ravel()
        if X_c[2] <= 0:
            raise ValueError("Point is behind the camera (Z ≤ 0)")
        x_n = X_c[:2] / X_c[2]
        uv = self.K[:2, :2] @ x_n + self.K[:2, 2]
        return uv

    # ── Utility ───────────────────────────────────────────────────────────────

    def reprojection_error(
        self,
        points_3d: np.ndarray,
        points_2d: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
    ) -> np.ndarray:
        """
        Compute per-point reprojection error (Euclidean distance in pixels).

        Parameters
        ----------
        points_3d : ndarray, (N, 3)
        points_2d : ndarray, (N, 2) — observed pixel coords
        R, t      : rotation and translation (camera ← world)

        Returns
        -------
        ndarray, shape (N,) — reprojection error per point in pixels.
        """
        projected = self.project(points_3d, R, t)
        return np.linalg.norm(projected - np.asarray(points_2d, dtype=np.float64), axis=1)

    @classmethod
    def from_kitti_sequence(cls, sequence) -> "Camera":
        """
        Construct a Camera from a :class:`~vo.data.KITTISequence` object.

        Parameters
        ----------
        sequence
            An instance of ``KITTISequence`` (duck-typed).
        """
        return cls(K=sequence.K, dist_coeffs=sequence.dist_coeffs)

    def __repr__(self) -> str:
        return (
            f"Camera(fx={self.fx:.2f}, fy={self.fy:.2f}, "
            f"cx={self.cx:.2f}, cy={self.cy:.2f})"
        )

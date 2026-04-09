"""
Essential Matrix Estimation, Pose Recovery, and Triangulation
==============================================================

Mathematical Background
-----------------------

The Essential Matrix
~~~~~~~~~~~~~~~~~~~~
For two calibrated cameras (intrinsics K₁, K₂ known), a pair of
corresponding image points x₁ and x₂ satisfy the epipolar constraint:

    x₂ᵀ E x₁ = 0

where x_i = K_i⁻¹ [u_i, v_i, 1]ᵀ are normalised camera coordinates and
E is the 3×3 Essential Matrix relating the two camera positions.

E encodes the relative rotation R and translation t between the cameras:

    E = [t]× R

where [t]× is the 3×3 skew-symmetric cross-product matrix of t:

    [t]× = [[ 0  -tz  ty]
             [ tz   0  -tx]
             [-ty  tx   0 ]]

Algebraic Properties of E
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Proposition: E has exactly two equal non-zero singular values.

Proof:
  Let E = U diag(σ₁, σ₂, σ₃) Vᵀ.  Since E = [t]× R and both [t]× and R
  are in SO(3) (or the cross-product matrix of a unit vector), one can show:

    E Eᵀ = [t]× R Rᵀ [t]×ᵀ = [t]× [t]×ᵀ

  Expanding [t]× [t]×ᵀ for t = [a,b,c]ᵀ:
    = [[b²+c²  -ab    -ac  ]
       [-ab    a²+c²  -bc  ]
       [-ac    -bc    a²+b²]]
    = ‖t‖² I - t tᵀ

  This is a rank-2 matrix with eigenvalues
    {‖t‖², ‖t‖², 0}

  Therefore the singular values of E are {‖t‖, ‖t‖, 0}, i.e. the two
  non-zero singular values are always equal.

In practice, the estimated Ê satisfies this constraint only approximately.
We project to the nearest Essential Matrix by enforcing σ₁=σ₂=(σ₁+σ₂)/2,
σ₃=0 via SVD:
    Ê* = U diag((σ₁+σ₂)/2, (σ₁+σ₂)/2, 0) Vᵀ

Pose Recovery from E
~~~~~~~~~~~~~~~~~~~~~
There are four possible decompositions of E into (R, t):

    E = U diag(1,1,0) Vᵀ   (project as above, normalised)

    Let W = [[0,-1,0],[1,0,0],[0,0,1]]   (90° rotation matrix)

    Four hypotheses:
      (R1, +t) = (U W  Vᵀ,  +U[:,2])
      (R1, -t) = (U W  Vᵀ,  -U[:,2])
      (R2, +t) = (U Wᵀ Vᵀ,  +U[:,2])
      (R2, -t) = (U Wᵀ Vᵀ,  -U[:,2])

The correct hypothesis is the one in which the triangulated 3D points have
positive depth (chirality test): Z > 0 in both camera frames.

Triangulation (DLT)
~~~~~~~~~~~~~~~~~~~~
Given projection matrices P₁ = K[I|0] and P₂ = K[R|t] and matching
image points x₁, x₂ (homogeneous), the 3D point X satisfies:

    x₁ × (P₁ X) = 0   →   Form rows of A
    x₂ × (P₂ X) = 0

This gives a 4×4 linear system AX = 0.  The minimum-norm solution is the
right singular vector of A corresponding to its smallest singular value.

References
----------
• Hartley & Zisserman, "Multiple View Geometry", §9 (essential matrix),
  §12.2 (triangulation).
• Nister, "An Efficient Solution to the Five-Point Relative Pose Problem",
  TPAMI 2004.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .camera import Camera

# Information matrix weight scales — larger = trust this DOF more.
# Rotation is typically estimated more reliably than translation.
_INFO_ROT_SCALE   = 10.0   # baseline weight for rotation DOFs
_INFO_TRANS_SCALE =  5.0   # baseline weight for translation DOFs


@dataclass
class PoseEstimate:
    """
    Relative camera pose between two frames.

    Parameters
    ----------
    R : (3, 3) rotation matrix
    t : (3,) unit translation vector
    E : (3, 3) Essential matrix
    inlier_mask : (N,) bool – RANSAC inliers
    n_inliers : int
    n_total : int
    information : (6, 6) information matrix (inverse covariance).
        Constructed from the RANSAC inlier ratio so that low-quality
        estimates contribute less to the optimizer cost. Layout is
        [rotation (3), translation (3)] matching the se(3) convention.
    """
    R: np.ndarray
    t: np.ndarray
    E: np.ndarray
    inlier_mask: np.ndarray
    n_inliers: int
    n_total: int
    information: np.ndarray = None   # (6, 6) – set post-init

    def __post_init__(self) -> None:
        if self.information is None:
            self.information = self._build_information()

    def _build_information(self) -> np.ndarray:
        """
        Build a diagonal 6×6 information matrix scaled by inlier_ratio².

        Squaring the ratio penalises low-quality estimates more aggressively:
        an estimate with 50% inliers contributes only 25% of the weight of
        a perfect estimate with 100% inliers.
        """
        q = self.inlier_ratio ** 2
        diag = np.array(
            [_INFO_ROT_SCALE] * 3 + [_INFO_TRANS_SCALE] * 3,
            dtype=np.float64,
        )
        return np.diag(q * diag)

    @property
    def inlier_ratio(self) -> float:
        return self.n_inliers / max(1, self.n_total)

    def transform_matrix(self) -> np.ndarray:
        """Return the 4×4 SE(3) transformation matrix [R | t; 0 | 1]."""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.R
        T[:3, 3] = self.t.ravel()
        return T


class MotionEstimator:
    """
    Recover relative camera motion between two frames from feature correspondences.

    Uses the 5-point algorithm (via OpenCV) seeded inside RANSAC to
    estimate the Essential Matrix, then recovers the unique (R, t) via SVD.

    Parameters
    ----------
    camera : Camera
        Calibrated camera model.
    ransac_threshold : float
        Epipolar distance threshold (pixels) for RANSAC inlier classification.
    ransac_confidence : float
        RANSAC confidence level ∈ (0, 1).
    min_inliers : int
        Minimum inlier count to accept a pose estimate.
    """

    def __init__(
        self,
        camera: Camera,
        ransac_threshold: float = 1.0,
        ransac_confidence: float = 0.999,
        min_inliers: int = 15,
    ) -> None:
        self.camera = camera
        self.ransac_threshold = ransac_threshold
        self.ransac_confidence = ransac_confidence
        self.min_inliers = min_inliers

    # ── Public API ────────────────────────────────────────────────────────────

    def estimate(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
    ) -> Optional[PoseEstimate]:
        """
        Estimate relative pose from matched feature coordinates.

        Parameters
        ----------
        pts1 : np.ndarray, shape (N, 2)
            Pixel coordinates in the first frame.
        pts2 : np.ndarray, shape (N, 2)
            Corresponding pixel coordinates in the second frame.

        Returns
        -------
        PoseEstimate | None
            Estimated pose, or None if estimation fails (too few matches,
            degenerate geometry, too few inliers).
        """
        if len(pts1) < 5:
            return None

        E, mask = self._compute_essential(pts1, pts2)
        if E is None or mask is None:
            return None

        inlier_mask = mask.ravel().astype(bool)
        n_inliers = int(inlier_mask.sum())

        if n_inliers < self.min_inliers:
            return None

        pts1_in = pts1[inlier_mask]
        pts2_in = pts2[inlier_mask]

        R, t, pose_mask = self._recover_pose(E, pts1_in, pts2_in)
        if R is None:
            return None

        return PoseEstimate(
            R=R, t=t.ravel(), E=E,
            inlier_mask=inlier_mask,
            n_inliers=n_inliers,
            n_total=len(pts1),
        )

    def triangulate(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
    ) -> np.ndarray:
        """
        Triangulate 3D points from two sets of matched pixel coordinates.

        Uses the DLT method (cv2.triangulatePoints) which forms the
        homogeneous system  A·X=0  and solves via SVD.

        Parameters
        ----------
        pts1 : (N, 2)  — points in frame 1 (pixel coords)
        pts2 : (N, 2)  — corresponding points in frame 2
        R    : (3, 3)  — rotation (frame 2 ← frame 1)
        t    : (3,)    — translation (frame 2 ← frame 1)

        Returns
        -------
        points_3d : np.ndarray, shape (N, 3)
            Triangulated 3D points in the frame-1 coordinate system.
            Points behind either camera are set to NaN.
        """
        K = self.camera.K
        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])       # K [I | 0]
        P2 = K @ np.hstack([R, t.reshape(3, 1)])                 # K [R | t]

        # triangulatePoints expects (2, N) float32
        pts1_f = pts1.T.astype(np.float32)
        pts2_f = pts2.T.astype(np.float32)

        pts4d = cv2.triangulatePoints(P1, P2, pts1_f, pts2_f)   # (4, N) homogeneous
        pts3d = (pts4d[:3] / pts4d[3]).T                         # (N, 3)

        # Chirality filter: discard points behind either camera
        # In camera 2 frame: X2 = R X1 + t
        pts3d_cam2 = (R @ pts3d.T + t.reshape(3, 1)).T
        behind = (pts3d[:, 2] <= 0) | (pts3d_cam2[:, 2] <= 0)
        pts3d[behind] = np.nan

        return pts3d

    # ── Private Helpers ──────────────────────────────────────────────────────

    def _compute_essential(
        self, pts1: np.ndarray, pts2: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Estimate the Essential Matrix using the 5-point algorithm with RANSAC.

        Returns (E, inlier_mask) or (None, None) on failure.
        """
        E, mask = cv2.findEssentialMat(
            pts1.astype(np.float32),
            pts2.astype(np.float32),
            self.camera.K,
            method=cv2.RANSAC,
            prob=self.ransac_confidence,
            threshold=self.ransac_threshold,
        )

        if E is None or E.shape != (3, 3):
            return None, None

        # Project to nearest valid Essential Matrix:
        # Force singular values to (σ, σ, 0)
        E = self._project_to_essential(E)
        return E, mask

    @staticmethod
    def _project_to_essential(E: np.ndarray) -> np.ndarray:
        """
        Project a matrix to the manifold of valid Essential Matrices.

        Enforces the two-equal-singular-values constraint:
            σ₁ = σ₂ = (σ₁_raw + σ₂_raw) / 2,  σ₃ = 0

        This is the closest Essential Matrix in Frobenius norm (Eq. 9.13
        in Hartley & Zisserman).
        """
        U, s, Vt = np.linalg.svd(E)
        sigma = (s[0] + s[1]) / 2.0
        E_proj = U @ np.diag([sigma, sigma, 0.0]) @ Vt
        return E_proj

    def _recover_pose(
        self, E: np.ndarray, pts1: np.ndarray, pts2: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Recover (R, t) from E.  Returns (R, t, mask) or (None, None, None).

        OpenCV's recoverPose implements the chirality test internally:
        it tests all four (R, t) hypotheses and selects the one with the
        most points having positive depth in both cameras.
        """
        n_inliers, R, t, mask = cv2.recoverPose(
            E,
            pts1.astype(np.float32),
            pts2.astype(np.float32),
            self.camera.K,
        )

        if n_inliers < self.min_inliers:
            return None, None, None

        return R, t.ravel(), mask

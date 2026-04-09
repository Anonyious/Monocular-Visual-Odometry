"""
Feature Detection and Optical Flow Tracking
============================================

Mathematical Background
-----------------------

ORB (Oriented FAST and Rotated BRIEF)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ORB combines two algorithms:

  1. FAST Keypoint Detector (Features from Accelerated Segment Test):
     A pixel p is a corner if there exist N ≥ 9 contiguous pixels on a
     circle of radius 3 around p that are all brighter than I(p) + τ
     or all darker than I(p) - τ.  Harris score is computed for each
     FAST keypoint and used to rank and retain the strongest N.

  2. BRIEF Descriptor with Orientation Normalization:
     BRIEF builds a 256-bit binary string by comparing intensities of
     nd pairs of pixels sampled from a steerable Gaussian window:
         b_i = 1 if I(p_a_i) < I(p_b_i) else 0
     ORB rotates the sampling patch by the keypoint's dominant gradient
     orientation θ (computed from the image intensity centroid), making
     the descriptor rotation-invariant.

Lucas-Kanade Optical Flow with Image Pyramids
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The brightness constancy (or aperture) assumption states that the
intensity of a physical point does not change between frames:
    I(x, y, t) = I(x + u·δt, y + v·δt, t + δt)

Taylor-expanding the right side and dropping higher-order terms:
    ∂I/∂x · u + ∂I/∂y · v + ∂I/∂t = 0
    ∇I · [u v]ᵀ = -I_t           (the optical flow equation)

This is one equation in two unknowns (the aperture problem). LK solves
it by assuming the flow is constant within a small window Ω of W×W pixels:

    Σ_{(x,y)∈Ω} [Ix Iy] · [u v]ᵀ = -It   for each pixel

In matrix form:  A v = b
    A = Σ [[Ix²   IxIy ]    b = -Σ [[Ix·It]]
           [IxIy  Iy²  ]]          [[Iy·It]]

The least-squares solution is: v = (AᵀA)⁻¹ Aᵀb
where AᵀA is the 2×2 structure tensor.

Image pyramids allow tracking large displacements by coarse-to-fine
refinement:  compute at the coarsest scale first, propagate as initializer
to the next finer scale.

References
----------
• Rublee et al., "ORB: An Efficient Alternative to SIFT or SURF", ICCV 2011.
• Lucas & Kanade, "An Iterative Image Registration Technique ...", DARPA 1981.
• Bouguet, "Pyramidal Implementation of the Lucas-Kanade Feature Tracker", 2001.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np


# ── LK Optical Flow Parameters ────────────────────────────────────────────────
_LK_PARAMS = dict(
    winSize=(21, 21),          # window size Ω for the structure tensor
    maxLevel=3,                # pyramid depth (2^3 = 8× coarse at top)
    criteria=(
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        30,                    # max iterations per level
        0.01,                  # ε for convergence
    ),
)

# ── ORB Parameters ─────────────────────────────────────────────────────────────
_ORB_N_FEATURES = 3000          # max keypoints per frame
_ORB_SCALE_FACTOR = 1.2         # pyramid scale factor
_ORB_N_LEVELS = 8               # pyramid levels
_ORB_EDGE_THRESHOLD = 31        # border to exclude
_ORB_PATCH_SIZE = 31            # BRIEF patch size

# ── Matching ──────────────────────────────────────────────────────────────────
_LOWE_RATIO = 0.75              # ratio test threshold (Lowe 2004)
_MIN_MATCHES = 8                # minimum matches to declare success


@dataclass
class TrackingResult:
    """Outcome of one round of Lucas-Kanade tracking."""
    pts_prev: np.ndarray          # (N, 2) – source points
    pts_curr: np.ndarray          # (N, 2) – tracked points
    status: np.ndarray            # (N,)   – 1 = tracked OK, 0 = lost
    good_prev: np.ndarray         # (M, 2) – successful source points
    good_curr: np.ndarray         # (M, 2) – successful target points
    n_tracked: int                # M – number of successfully tracked pts

    @property
    def inlier_ratio(self) -> float:
        return self.n_tracked / max(1, len(self.pts_prev))


class FeatureFrontend:
    """
    Two-stage feature front-end:
      1. ORB detection   – find keypoints and descriptors in a frame.
      2. LK tracking     – track a set of points to the next frame.
      3. ORB matching    – cross-frame descriptor matching (optional path).

    The pipeline uses LK as the primary tracker (faster, no descriptor
    recomputation) and falls back to ORB matching when tracking degrades.

    Parameters
    ----------
    n_features : int
        Maximum number of ORB features to detect per frame.
    min_tracked : int
        Minimum features before triggering a full ORB re-detect.
    lk_params : dict | None
        Override Lucas-Kanade parameters (see cv2.calcOpticalFlowPyrLK).
    """

    def __init__(
        self,
        n_features: int = _ORB_N_FEATURES,
        min_tracked: int = 500,
        lk_params: Optional[dict] = None,
    ) -> None:
        self.n_features = n_features
        self.min_tracked = min_tracked
        self.lk_params = lk_params or _LK_PARAMS

        # ORB detector — uses FAST + Harris score + steered BRIEF
        self._orb = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=_ORB_SCALE_FACTOR,
            nlevels=_ORB_N_LEVELS,
            edgeThreshold=_ORB_EDGE_THRESHOLD,
            patchSize=_ORB_PATCH_SIZE,
        )

        # Hamming distance matcher for binary ORB descriptors
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect ORB keypoints and compute descriptors.

        Parameters
        ----------
        frame : np.ndarray
            Greyscale or BGR image.

        Returns
        -------
        keypoints : np.ndarray, shape (N, 2)
            Detected pixel coordinates (u, v).
        descriptors : np.ndarray, shape (N, 32)
            32-byte (256-bit) ORB descriptors.
        """
        gray = self._to_gray(frame)
        kps, descs = self._orb.detectAndCompute(gray, None)

        if not kps:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 32), dtype=np.uint8)

        # Convert cv2.KeyPoint list to (N,2) array
        pts = np.array([kp.pt for kp in kps], dtype=np.float32)
        descs = descs if descs is not None else np.empty((0, 32), dtype=np.uint8)
        return pts, descs

    def track(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        pts_prev: np.ndarray,
    ) -> TrackingResult:
        """
        Track keypoints from *prev_frame* to *curr_frame* with LK pyramid flow.

        Implements forward-backward (FB) error checking:
          - Track prev → curr  (forward)
          - Track curr → prev  (backward)
          - Discard points where FB error > threshold

        Parameters
        ----------
        prev_frame, curr_frame : np.ndarray
            Greyscale or BGR images of consecutive frames.
        pts_prev : np.ndarray, shape (N, 2)
            Pixel coordinates to track from the previous frame.

        Returns
        -------
        TrackingResult
            Named tuple of filtered source / target point sets.
        """
        if len(pts_prev) == 0:
            empty = np.empty((0, 2), dtype=np.float32)
            return TrackingResult(empty, empty, np.array([], dtype=np.uint8), empty, empty, 0)

        gray_prev = self._to_gray(prev_frame)
        gray_curr = self._to_gray(curr_frame)

        pts_prev_f32 = pts_prev.astype(np.float32).reshape(-1, 1, 2)

        # Forward pass: prev → curr
        pts_curr_f, st_f, _ = cv2.calcOpticalFlowPyrLK(
            gray_prev, gray_curr, pts_prev_f32, None, **self.lk_params
        )

        # Backward pass: curr → prev (error check)
        pts_back, st_b, _ = cv2.calcOpticalFlowPyrLK(
            gray_curr, gray_prev, pts_curr_f, None, **self.lk_params
        )

        # Forward-backward error: L2 distance from original to back-tracked point
        fb_error = np.linalg.norm(
            pts_prev_f32.reshape(-1, 2) - pts_back.reshape(-1, 2), axis=1
        )

        # Status: both forward and backward must succeed, FB error < 1 pixel
        valid = (
            (st_f.ravel() == 1) &
            (st_b.ravel() == 1) &
            (fb_error < 1.0)
        )

        pts_curr_out = pts_curr_f.reshape(-1, 2)
        pts_prev_flat = pts_prev_f32.reshape(-1, 2)

        good_prev = pts_prev_flat[valid]
        good_curr = pts_curr_out[valid]

        return TrackingResult(
            pts_prev=pts_prev_flat,
            pts_curr=pts_curr_out,
            status=valid.astype(np.uint8),
            good_prev=good_prev,
            good_curr=good_curr,
            n_tracked=int(valid.sum()),
        )

    def match(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
        ratio_threshold: float = _LOWE_RATIO,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match ORB descriptors between two frames using Lowe's ratio test.

        For each descriptor d1 in desc1, find the two nearest neighbours in
        desc2.  Accept the match only if:
            dist(d1, nn1) < ratio_threshold × dist(d1, nn2)

        This filters ambiguous matches (where two candidates are similarly
        close) that are likely incorrect.

        Parameters
        ----------
        desc1, desc2 : np.ndarray, shape (N, 32) and (M, 32)
            ORB binary descriptors (uint8).
        ratio_threshold : float
            Lowe ratio test threshold (default 0.75).

        Returns
        -------
        pts1 : np.ndarray, shape (K, 2)  — matched indices from desc1 frame
        pts2 : np.ndarray, shape (K, 2)  — matched indices from desc2 frame

        Note: the caller must pass keypoint positions separately.  This
        method returns index arrays; use match_keypoints() for coordinates.
        """
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return np.empty((0,), np.int32), np.empty((0,), np.int32)

        # kNN with k=2 for ratio test
        raw = self._matcher.knnMatch(desc1, desc2, k=2)

        good_idx1, good_idx2 = [], []
        for pair in raw:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < ratio_threshold * n.distance:
                good_idx1.append(m.queryIdx)
                good_idx2.append(m.trainIdx)

        return np.array(good_idx1, dtype=np.int32), np.array(good_idx2, dtype=np.int32)

    def match_keypoints(
        self,
        kp1: np.ndarray,
        desc1: np.ndarray,
        kp2: np.ndarray,
        desc2: np.ndarray,
        ratio_threshold: float = _LOWE_RATIO,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match and return matched coordinate pairs directly.

        Returns
        -------
        pts1 : (K, 2)
        pts2 : (K, 2)
        """
        idx1, idx2 = self.match(desc1, desc2, ratio_threshold)
        if len(idx1) == 0:
            return np.empty((0, 2)), np.empty((0, 2))
        return kp1[idx1], kp2[idx2]

    def detect_and_track(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        pts_prev: Optional[np.ndarray] = None,
    ) -> TrackingResult:
        """
        Combined detect + track.

        If *pts_prev* is None or has fewer than min_tracked points,
        re-detects ORB keypoints in *prev_frame* first, then tracks.

        Parameters
        ----------
        prev_frame, curr_frame : np.ndarray
            Consecutive frames.
        pts_prev : ndarray, shape (N, 2) | None
            Existing points to track. Re-detect if None or too few.

        Returns
        -------
        TrackingResult
        """
        if pts_prev is None or len(pts_prev) < self.min_tracked:
            pts_prev, _ = self.detect(prev_frame)

        return self.track(prev_frame, curr_frame, pts_prev)

    # ── Visualisation Helper ─────────────────────────────────────────────────

    def draw_tracks(
        self,
        frame: np.ndarray,
        result: TrackingResult,
        colour: Tuple[int, int, int] = (0, 255, 0),
        radius: int = 3,
    ) -> np.ndarray:
        """
        Draw tracked feature points and motion vectors on *frame*.

        Returns a BGR copy of *frame* with:
          - Green dots for tracked points.
          - Red lines showing motion from prev to curr.
        """
        out = frame.copy()
        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

        for (px, py), (cx, cy) in zip(result.good_prev, result.good_curr):
            cv2.line(out, (int(px), int(py)), (int(cx), int(cy)), (0, 0, 200), 1)
            cv2.circle(out, (int(cx), int(cy)), radius, colour, -1)

        label = f"Tracked: {result.n_tracked}  Ratio: {result.inlier_ratio:.2f}"
        cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return out

    # ── Internal Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        """Convert BGR or greyscale image to single-channel uint8."""
        if frame.ndim == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

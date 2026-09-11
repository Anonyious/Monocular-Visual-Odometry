"""
Adaptive RANSAC (RANdom SAmple Consensus)
==========================================

Mathematical Background
-----------------------
RANSAC estimates a model M* from noisy, outlier-contaminated data by
iterating three steps:
  1. Sample s points at random (the minimal sample).
  2. Fit a model to those s points.
  3. Count inliers: points whose residual < threshold τ.
  4. Store the model if its inlier count exceeds the current best.

Required Iterations
~~~~~~~~~~~~~~~~~~~
Let  ε  = fraction of outliers in the data,
     s  = minimal sample size,
     p  = desired probability of finding a correct model.

The probability that a single sample of s points contains NO outlier is:
     P(all s from inliers) = (1 - ε)^s

The probability that after N iterations we have found at least one
all-inlier sample is:
     p = 1 - (1 - (1-ε)^s)^N

Solving for N:
     N = log(1 - p) / log(1 - (1-ε)^s)

Adaptive RANSAC
~~~~~~~~~~~~~~~
Since ε is unknown in advance, we estimate it from the best inlier set
found so far and update N after each iteration:

     ε̂  = 1 - (best_inliers / n_points)
     N  ← log(1-p) / log(1 - (1-ε̂)^s)

This usually terminates far earlier than the worst-case fixed N.

References
----------
• Fischler & Bolles, "Random Sample Consensus", CACM 1981.
• Hartley & Zisserman, "Multiple View Geometry", §4.7.1.
• Torr & Zisserman, "MLESAC: A New Robust Estimator ...", CVIU 2000.
"""

from __future__ import annotations

import math
import logging
from typing import Callable, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class RANSAC:
    """
    Generic adaptive RANSAC for any model that can be fit from s samples.

    Parameters
    ----------
    model_fn : Callable[[np.ndarray], any]
        Fits a model to a (s, ...) array of minimal samples.
        Returns a model object (or None if degenerate).
    residual_fn : Callable[[model, np.ndarray], np.ndarray]
        Computes per-point residuals for a fitted model.
        Shape: (model, data (N,...)) → (N,) float.
    refit_fn : Callable[[np.ndarray], any], optional
        Fits a model to a (N, ...) array of all inliers. If provided,
        RANSAC refits the model over all best inliers at the end.
    min_samples : int
        Minimal sample size s.
    threshold : float
        Inlier residual threshold τ.
    confidence : float
        Desired probability p that the result contains an all-inlier sample.
        Typical value: 0.99.
    max_iterations : int
        Hard cap on iterations (prevents infinite loops when ε is large).
    """

    def __init__(
        self,
        model_fn: Callable,
        residual_fn: Callable,
        min_samples: int,
        threshold: float,
        confidence: float = 0.99,
        max_iterations: int = 2000,
        refit_fn: Optional[Callable] = None,
    ) -> None:
        self.model_fn = model_fn
        self.residual_fn = residual_fn
        self.refit_fn = refit_fn
        self.min_samples = min_samples
        self.threshold = threshold
        self.confidence = confidence
        self.max_iterations = max_iterations

    def fit(
        self, data: np.ndarray, rng: Optional[np.random.Generator] = None
    ) -> Tuple[any, np.ndarray]:
        """
        Run adaptive RANSAC on *data*.

        Parameters
        ----------
        data : np.ndarray, shape (N, ...) or list of length N
            Complete dataset.
        rng : np.random.Generator | None
            Random number generator (for reproducibility).

        Returns
        -------
        best_model : any
            Model fitted to the best inlier set.  ``None`` if RANSAC fails.
        inlier_mask : np.ndarray, shape (N,), dtype bool
            True for points consistent with *best_model*.
        """
        if rng is None:
            rng = np.random.default_rng()

        data = np.asarray(data)
        n = len(data)

        if n < self.min_samples:
            logger.warning("RANSAC: data size %d < min_samples %d", n, self.min_samples)
            return None, np.zeros(n, dtype=bool)

        best_model = None
        best_inlier_mask = np.zeros(n, dtype=bool)
        best_n_inliers = 0

        # Start with max iterations, decrease it as we find inlier ratios
        n_iter = self.max_iterations

        iteration = 0
        while iteration < min(n_iter, self.max_iterations):
            # 1. Random minimal sample
            sample_idx = rng.choice(n, size=self.min_samples, replace=False)
            sample = data[sample_idx]

            # 2. Fit model
            try:
                model = self.model_fn(sample)
            except Exception as exc:
                logger.debug("RANSAC model_fn raised %s; skipping sample", exc)
                iteration += 1
                continue

            if model is None:
                iteration += 1
                continue

            # 3. Count inliers
            residuals = self.residual_fn(model, data)
            inlier_mask = residuals < self.threshold
            n_inliers = int(inlier_mask.sum())

            # 4. Update best
            if n_inliers > best_n_inliers:
                best_n_inliers = n_inliers
                best_inlier_mask = inlier_mask
                best_model = model

                # Adaptive N update: refit inlier ratio estimate
                eps_hat = 1.0 - n_inliers / n
                eps_hat = np.clip(eps_hat, 1e-6, 1.0 - 1e-6)
                n_iter = self._required_iterations(eps_hat)
                logger.debug(
                    "RANSAC iter %d: %d inliers (%.1f%%), new N=%d",
                    iteration, n_inliers, 100 * (1 - eps_hat), n_iter,
                )

            iteration += 1

        if best_model is None:
            logger.warning("RANSAC failed to find a valid model after %d iterations", iteration)
            return None, np.zeros(n, dtype=bool)

        # Optional: re-fit on all inliers for a more accurate final model
        if best_n_inliers >= self.min_samples and self.refit_fn is not None:
            try:
                refined = self.refit_fn(data[best_inlier_mask])
                if refined is not None:
                    best_model = refined
            except Exception as exc:
                logger.debug("refit_fn raised %s", exc)

        return best_model, best_inlier_mask

    # ── Private Helpers ──────────────────────────────────────────────────────

    def _required_iterations(self, epsilon: float) -> int:
        """
        Compute required RANSAC iterations given outlier ratio ε.

        N = ceil( log(1-p) / log(1 - (1-ε)^s) )

        Parameters
        ----------
        epsilon : float
            Estimated outlier ratio ∈ (0, 1).

        Returns
        -------
        int
            Number of iterations (capped at max_iterations).
        """
        p_all_inlier = (1.0 - epsilon) ** self.min_samples
        if p_all_inlier < 1e-10:
            return self.max_iterations
            
        denom = math.log1p(-p_all_inlier) # Better numerical stability than log(1-p)
        if abs(denom) < 1e-15:
            return self.max_iterations
            
        # Clip confidence to avoid math.log(0) if passed 1.0
        conf = np.clip(self.confidence, 1e-6, 1.0 - 1e-6)
        n_float = math.log1p(-conf) / denom
        return min(int(math.ceil(n_float)), self.max_iterations)

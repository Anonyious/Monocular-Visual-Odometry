"""
Bag-of-Words Loop Closure Detection
=====================================

Mathematical Background
-----------------------

Visual Bag-of-Words (BoW)
~~~~~~~~~~~~~~~~~~~~~~~~~~
A visual vocabulary V = {w₁, …, w_K} is learned by k-means clustering
a large set of feature descriptors d ∈ ℝ^(D) (for binary ORB, D=32 bytes).

Assignment: each descriptor is mapped to its nearest visual word:
    a(d) = argmin_{k} dist(d, w_k)

For binary descriptors, Hamming distance is used:
    dist(d₁, d₂) = popcount(d₁ XOR d₂)

Image Representation (BoW Histogram)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Given a set of N descriptors from an image, the BoW histogram h ∈ ℝ^K is:
    h[k] = TF(k) × IDF(k)

where:
    TF(k) = count of descriptors assigned to word k in this image / N
    IDF(k) = log(|DB| / |{images containing word k}|)

TF-IDF (Term Frequency – Inverse Document Frequency) downweights words
that appear in many images (hence uninformative) and upweights rare words.

Similarity Score
~~~~~~~~~~~~~~~~~~
Two images are compared by L1-normalised vector similarity:

    s(h_query, h_cand) = 1 - ½‖h_query/‖h_query‖₁ - h_cand/‖h_cand‖₁‖₁

This is equivalent to a normalised histogram intersection and is in [0, 1].
A score close to 1 means the images look very similar (loop closure candidate).

Geometric Verification
~~~~~~~~~~~~~~~~~~~~~~~
Appearance similarity is a necessary but not sufficient condition for a
loop closure.  We verify each candidate geometrically by:
  1. Match ORB descriptors between the query and candidate frames.
  2. Estimate the Essential Matrix with RANSAC.
  3. Accept the loop if the inlier ratio > min_inlier_ratio.

References
----------
• Galvez-Lopez & Tardos, "Bags of Binary Words for Fast Place Recognition
  in Image Sequences", IEEE TRO 2012 (DBoW2).
• Sivic & Zisserman, "Video Google: A Text Retrieval Approach to Object
  Matching in Videos", ICCV 2003.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .camera import Camera
from .motion import MotionEstimator

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_MIN_LOOP_SCORE = 0.15       # minimum BoW similarity to consider candidate
_MIN_LOOP_INLIERS = 20       # minimum RANSAC inliers to confirm loop closure
_MIN_LOOP_INLIER_RATIO = 0.3 # minimum ratio
_MIN_FRAMES_APART = 30       # don't detect loops between very recent frames


@dataclass
class LoopCandidate:
    """A potential loop closure."""
    query_id: int
    candidate_id: int
    score: float                          # BoW similarity ∈ [0, 1]
    R: Optional[np.ndarray] = None       # verified relative rotation
    t: Optional[np.ndarray] = None       # verified relative translation
    n_inliers: int = 0
    verified: bool = False


class BagOfWords:
    """
    Visual Bag-of-Words database for place recognition.

    Vocabulary is trained on ORB descriptors using k-means.
    New frames are added to the database incrementally.

    Parameters
    ----------
    n_words : int
        Vocabulary size K.
    max_iter_kmeans : int
        Maximum k-means iterations for vocabulary training.
    """

    def __init__(self, n_words: int = 500, max_iter_kmeans: int = 100) -> None:
        self.n_words = n_words
        self.max_iter_kmeans = max_iter_kmeans

        self._vocabulary: Optional[np.ndarray] = None   # (K, 32) float32 cluster centres
        self._database: Dict[int, np.ndarray] = {}       # frame_id → L1-norm BoW histogram
        self._word_doc_freq: np.ndarray = np.zeros(n_words, dtype=np.float64)  # IDF numerator
        self._n_docs: int = 0

    @property
    def is_trained(self) -> bool:
        return self._vocabulary is not None

    def train(self, descriptor_lists: List[np.ndarray]) -> None:
        """
        Learn a visual vocabulary from a list of descriptor arrays.

        Parameters
        ----------
        descriptor_lists : list of ndarray, each shape (M_i, 32)
            ORB descriptors from training images.
        """
        all_desc = np.vstack([d for d in descriptor_lists if d is not None and len(d) > 0])
        all_desc = all_desc.astype(np.float32)

        logger.info("Training BoW vocabulary: %d descriptors → %d words ...", len(all_desc), self.n_words)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
            self.max_iter_kmeans,
            1.0,
        )
        _, _, centres = cv2.kmeans(
            all_desc,
            self.n_words,
            None,
            criteria,
            attempts=3,
            flags=cv2.KMEANS_PP_CENTERS,
        )
        self._vocabulary = centres  # (K, 32) float32
        # IDF denominator reset
        self._word_doc_freq = np.zeros(self.n_words, dtype=np.float64)
        self._n_docs = 0
        logger.info("Vocabulary trained.")

    def save(self, path: str) -> None:
        """
        Serialize the trained vocabulary to a NumPy .npz file.

        Parameters
        ----------
        path : str
            Destination file path (will be created/overwritten).
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save: vocabulary has not been trained.")
        np.savez_compressed(
            path,
            vocabulary=self._vocabulary,
            n_words=np.array([self.n_words]),
        )
        logger.info("BoW vocabulary saved to %s  (%d words)", path, self.n_words)

    def load(self, path: str) -> None:
        """
        Load a vocabulary from a .npz file produced by :meth:`save`.

        Parameters
        ----------
        path : str
            Path to the .npz file.
        """
        data = np.load(path)
        self._vocabulary = data["vocabulary"].astype(np.float32)
        self.n_words = int(data["n_words"][0])
        self._word_doc_freq = np.zeros(self.n_words, dtype=np.float64)
        self._n_docs = 0
        logger.info("BoW vocabulary loaded from %s  (%d words)", path, self.n_words)

    def add_image(self, frame_id: int, descriptors: np.ndarray) -> np.ndarray:
        """
        Compute and store the TF-IDF BoW histogram for a frame.

        Parameters
        ----------
        frame_id : int
        descriptors : (N, 32) uint8 ORB descriptors

        Returns
        -------
        histogram : np.ndarray, shape (K,) — the normalised TF-IDF vector
        """
        if not self.is_trained:
            raise RuntimeError("Call train() before add_image()")

        hist = self._build_histogram(descriptors)
        self._database[frame_id] = hist

        # Update IDF: count how many documents contain each word
        nonzero = hist > 0
        self._word_doc_freq[nonzero] += 1
        self._n_docs += 1

        return hist

    def query(
        self,
        descriptors: np.ndarray,
        top_k: int = 5,
        exclude_ids: Optional[List[int]] = None,
    ) -> List[Tuple[int, float]]:
        """
        Find the *top_k* most similar images in the database.

        Parameters
        ----------
        descriptors : (N, 32) ORB descriptors of the query image.
        top_k : int
        exclude_ids : list[int] | None
            Frame IDs to exclude (e.g., recent frames too close in time).

        Returns
        -------
        list of (frame_id, score) sorted by descending score.
        """
        if not self._database:
            return []

        query_hist = self._build_histogram(descriptors, use_idf=True)
        q_norm = query_hist / (np.linalg.norm(query_hist, ord=1) + 1e-10)

        scores = {}
        for fid, h in self._database.items():
            if exclude_ids and fid in exclude_ids:
                continue
            h_norm = h / (np.linalg.norm(h, ord=1) + 1e-10)
            # L1-norm similarity: 1 - ½‖q - h‖₁
            sim = 1.0 - 0.5 * np.linalg.norm(q_norm - h_norm, ord=1)
            scores[fid] = float(sim)

        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        return ranked[:top_k]

    # ── Private Helpers ──────────────────────────────────────────────────────

    def _build_histogram(
        self, descriptors: np.ndarray, use_idf: bool = False
    ) -> np.ndarray:
        """Quantise descriptors and compute (optionally TF-IDF) histogram."""
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.n_words)

        desc_f = descriptors.astype(np.float32)
        # Assign each descriptor to nearest visual word (Hamming via L2 on float)
        vocab_f = self._vocabulary.astype(np.float32)
        # BFMatcher for binary descriptors if available
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = matcher.match(descriptors.astype(np.uint8), self._vocabulary.astype(np.uint8))
        word_ids = np.array([m.trainIdx for m in matches], dtype=np.int32)

        # TF: term frequency
        tf = np.bincount(word_ids, minlength=self.n_words).astype(np.float64)
        tf /= max(len(descriptors), 1)

        if use_idf and self._n_docs > 0:
            idf = np.log(
                (self._n_docs + 1) / (self._word_doc_freq + 1)
            )
            return tf * idf
        return tf


class LoopClosureDetector:
    """
    Complete loop closure detector: BoW retrieval + geometric verification.

    Parameters
    ----------
    camera : Camera
    n_words : int
        BoW vocabulary size.
    min_score : float
        Minimum BoW similarity to trigger geometric verification.
        If ``vocab_path`` is provided, this is treated as a **floor**;
        an adaptive threshold based on the running score distribution will
        be used whenever the database is large enough.
    min_inliers : int
        Minimum RANSAC inliers to accept a loop closure.
    min_frames_apart : int
        Minimum frame separation to avoid trivial near-duplicate loops.
    vocab_path : str | None
        Path to a pre-trained vocabulary .npz file.  If supplied, the
        vocabulary is loaded immediately and online training is skipped.
        This is the recommended production mode.
    """

    def __init__(
        self,
        camera: Camera,
        n_words: int = 500,
        min_score: float = _MIN_LOOP_SCORE,
        min_inliers: int = _MIN_LOOP_INLIERS,
        min_frames_apart: int = _MIN_FRAMES_APART,
        vocab_path: Optional[str] = None,
    ) -> None:
        self.camera = camera
        self.min_score = min_score
        self.min_inliers = min_inliers
        self.min_frames_apart = min_frames_apart

        self._bow = BagOfWords(n_words=n_words)
        self._keyframe_descriptors: Dict[int, np.ndarray] = {}
        self._keyframe_keypoints: Dict[int, np.ndarray] = {}
        self._motion_estimator = MotionEstimator(camera, min_inliers=min_inliers)

        # --- Adaptive threshold tracking ---
        # We keep a running mean and standard deviation of *all* BoW similarity
        # scores returned by the database query (including non-matches).  Once
        # we have enough observations we raise the effective threshold to
        #   mu + k * sigma
        # where k is a user-tunable multiplier.  This prevents repetitive
        # environments (parking garages, tunnels) from flooding the graph with
        # false positives while still detecting true loop closures in varied scenes.
        self._score_history: List[float] = []
        self._adaptive_k: float = 1.5   # multiplier for adaptive threshold
        self._adaptive_min_obs: int = 50  # observations before enabling adaptive mode

        # --- Vocabulary loading / training ---
        self._training_buffer: List[np.ndarray] = []
        self._TRAIN_THRESHOLD = 20   # train vocab after N keyframes accumulated

        if vocab_path is not None:
            try:
                self._bow.load(vocab_path)
                self._trained = True
                logger.info("LoopClosureDetector: loaded vocab from %s", vocab_path)
            except Exception as exc:
                logger.warning(
                    "LoopClosureDetector: failed to load vocab from %s (%s). "
                    "Falling back to online training.", vocab_path, exc
                )
                self._trained = False
        else:
            self._trained = False

    def add_keyframe(
        self,
        frame_id: int,
        keypoints: np.ndarray,
        descriptors: np.ndarray,
    ) -> Optional[LoopCandidate]:
        """
        Add a keyframe to the BoW database and check for loop closures.

        Parameters
        ----------
        frame_id : int
        keypoints : (N, 2)
        descriptors : (N, 32)

        Returns
        -------
        LoopCandidate | None — verified loop closure if found.
        """
        self._keyframe_descriptors[frame_id] = descriptors
        self._keyframe_keypoints[frame_id] = keypoints

        # Accumulate training data
        if not self._trained:
            self._training_buffer.append(descriptors)
            if len(self._training_buffer) >= self._TRAIN_THRESHOLD:
                self._bow.train(self._training_buffer)
                # Add all accumulated frames to database
                for fid, desc in self._keyframe_descriptors.items():
                    self._bow.add_image(fid, desc)
                self._trained = True
            return None

        # Add this frame to BoW database
        self._bow.add_image(frame_id, descriptors)

        # Query for candidates (exclude very recent frames)
        recent_ids = [
            fid for fid in self._keyframe_descriptors
            if frame_id - fid < self.min_frames_apart
        ]
        candidates = self._bow.query(descriptors, top_k=5, exclude_ids=recent_ids)

        # Compute adaptive threshold from running score statistics
        effective_threshold = self._adaptive_threshold(candidates)

        for cand_id, score in candidates:
            # Track all top-5 scores for adaptive thresholding
            self._score_history.append(score)

            if score < effective_threshold:
                break  # sorted descending; remaining will be worse

            loop = self._verify_geometrically(frame_id, cand_id, score)
            if loop is not None and loop.verified:
                logger.info(
                    "Loop closure detected: frame %d ↔ %d (score=%.3f, inliers=%d, thresh=%.3f)",
                    frame_id, cand_id, score, loop.n_inliers, effective_threshold,
                )
                return loop

        return None

    def _adaptive_threshold(self, candidates: List[Tuple[int, float]]) -> float:
        """
        Compute the effective loop closure score threshold.

        In environments with many repetitive visual elements (tunnels, corridors),
        many frames will have high BoW similarity scores, making a fixed threshold
        too permissive. We adapt by raising the floor to mean + k * std.

        Returns the fixed :attr:`min_score` until enough observations have been
        collected (:attr:`_adaptive_min_obs`).
        """
        if len(self._score_history) < self._adaptive_min_obs:
            return self.min_score

        arr = np.array(self._score_history[-500:], dtype=np.float64)  # rolling window
        mu = float(arr.mean())
        sigma = float(arr.std())
        adaptive = mu + self._adaptive_k * sigma
        # Never go below the hard floor
        return max(self.min_score, adaptive)

    def _verify_geometrically(
        self, query_id: int, cand_id: int, score: float
    ) -> Optional[LoopCandidate]:
        """
        Verify a loop closure candidate by fitting the Essential Matrix.

        Returns a LoopCandidate with verified=True if confirmed, else None.
        """
        kp_q = self._keyframe_keypoints.get(query_id)
        kp_c = self._keyframe_keypoints.get(cand_id)
        desc_q = self._keyframe_descriptors.get(query_id)
        desc_c = self._keyframe_descriptors.get(cand_id)

        if any(x is None or len(x) == 0 for x in [kp_q, kp_c, desc_q, desc_c]):
            return None

        # ORB match with ratio test
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        raw = matcher.knnMatch(desc_q.astype(np.uint8), desc_c.astype(np.uint8), k=2)
        good = []
        for pair in raw:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append((m.queryIdx, m.trainIdx))

        if len(good) < self.min_inliers:
            return None

        pts_q = np.array([kp_q[i] for i, _ in good])
        pts_c = np.array([kp_c[j] for _, j in good])

        pose = self._motion_estimator.estimate(pts_q, pts_c)
        if pose is None:
            return None

        return LoopCandidate(
            query_id=query_id,
            candidate_id=cand_id,
            score=score,
            R=pose.R,
            t=pose.t,
            n_inliers=pose.n_inliers,
            verified=True,
        )

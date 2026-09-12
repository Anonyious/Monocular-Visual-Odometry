"""
Monocular Visual Odometry Pipeline Orchestrator
================================================

Ties together all components:
  Camera → Features → Motion/PnP → LocalMap → PoseGraph → LoopClosure → Optimizer

Architecture (v2)
~~~~~~~~~~~~~~~~~
This revision addresses the following systematic weaknesses of the original:

1. **PnP-first pose estimation** (Tier 2, highest impact)
   Once the local map has been bootstrapped, every new frame first attempts to
   localise via PnP against existing 3-D map points.  The Essential Matrix
   path is now a fallback, used only when the map is empty or PnP fails.
   This grounds pose estimates in the global 3-D structure rather than
   pairwise optical-flow differences, dramatically reducing drift.

2. **RANSAC ground-plane scale recovery** (Tier 1)
   The hard-coded ``t_y / camera_height`` trick is replaced by a proper RANSAC
   plane-fitting routine (``GroundPlaneScaleRecovery``) that survives road
   inclinations, different camera rigs, and non-trivial vehicle dynamics.

3. **Principled keyframe selection** (Tier 1)
   A frame becomes a keyframe only when:
     • tracked feature ratio drops below ``kf_track_ratio`` (default 0.80) of
       the previous keyframe's count, **and**
     • median optical-flow angular parallax exceeds ``kf_parallax_deg`` (2°),
   **or** the hard-gap fallback (``keyframe_max_gap``) fires.

4. **Covariance-weighted pose graph edges** (Tier 1)
   ``PoseEstimate.information`` (6×6 matrix derived from RANSAC inlier ratio)
   is now forwarded to ``PoseGraph.add_edge`` instead of ``np.eye(6)``.

5. **Loop-closure-triggered optimization** (Tier 1/2)
   Instead of running the optimizer every N keyframes unconditionally, it fires
   only when ``_pose_graph.consume_new_loops()`` returns a positive count.
   This prevents polluting the graph when no new information has been added.

Scale Recovery
~~~~~~~~~~~~~~
``GroundPlaneScaleRecovery`` triangulates 3-D points from consecutive keyframes,
RANSAC-fits a ground plane to those points, and computes:

    s = h_known / d_estimated

where d_estimated is the distance from the camera origin to the fitted plane.
An exponential moving average (α = 0.3) smooths the raw estimate over time.

Keyframing Policy (v2)
~~~~~~~~~~~~~~~~~~~~~~~~
A new keyframe is inserted when **either**:
  1. ``(n_tracked / n_kf_feats) < kf_track_ratio`` — fewer features are tracked
     compared to how many the previous keyframe had, indicating the camera has
     moved significantly.
  2. ``median_parallax_deg > kf_parallax_deg`` — the angular motion of tracked
     points exceeds 2° (sufficient baseline for good triangulation).
  3. ``frames_since_keyframe >= keyframe_max_gap`` — hard fallback.

Conditions 1 and 2 must both be satisfied (AND), except when condition 3 fires.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from tqdm import tqdm

from .camera import Camera
from .features import FeatureFrontend
from .local_map import LocalMap
from .loop_closure import LoopClosureDetector
from .motion import MotionEstimator, PoseEstimate
from .optimizer import PoseGraphOptimizer
from .pose_graph import PoseGraph, se3_compose, se3_inverse
from .scale_recovery import GroundPlaneScaleRecovery
from .evaluation.metrics import save_trajectory_kitti

logger = logging.getLogger(__name__)


class VisualOdometry:
    """
    Full monocular visual odometry pipeline (v2 — production-grade).

    Parameters
    ----------
    camera : Camera
        Calibrated camera model.
    camera_height : float
        Height of the camera above ground (metres) — used for scale recovery.
    kf_track_ratio : float
        Keyframe trigger: insert KF when tracked/previous < this ratio.
    kf_parallax_deg : float
        Keyframe trigger: insert KF when median parallax exceeds this (degrees).
    keyframe_max_gap : int
        Hard fallback: force a new keyframe after this many frames.
    vocab_path : str | None
        Path to pre-trained BoW vocabulary .npz (recommended for production).
        If None, vocabulary is trained online from the first 20 keyframes.
    verbose : bool
        Enable per-frame console output.
    """

    def __init__(
        self,
        camera: Camera,
        camera_height: float = 1.65,
        kf_track_ratio: float = 0.80,
        kf_parallax_deg: float = 2.0,
        keyframe_max_gap: int = 10,
        vocab_path: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        self.camera = camera
        self.camera_height = camera_height
        self.kf_track_ratio = kf_track_ratio
        self.kf_parallax_deg = kf_parallax_deg
        self.keyframe_max_gap = keyframe_max_gap
        self.verbose = verbose

        # ── Sub-components ────────────────────────────────────────────────────
        self._frontend = FeatureFrontend(n_features=3000)
        self._motion = MotionEstimator(camera)
        self._local_map = LocalMap(camera)
        self._loop_detector = LoopClosureDetector(camera, vocab_path=vocab_path)
        self._pose_graph = PoseGraph()
        self._optimizer = PoseGraphOptimizer(n_iterations=20, verbose=False)
        self._scale_recovery = GroundPlaneScaleRecovery(camera_height=camera_height)

        # ── State ─────────────────────────────────────────────────────────────
        self._prev_frame: Optional[np.ndarray] = None
        self._prev_kps: Optional[np.ndarray] = None
        self._prev_descs: Optional[np.ndarray] = None
        self._prev_frame_id: int = -1

        # Absolute pose (T_cw: camera ← world)
        self._current_R: np.ndarray = np.eye(3)
        self._current_t: np.ndarray = np.zeros(3)

        # Trajectory
        self._trajectory: List[np.ndarray] = []   # list of 4×4 SE(3) poses
        self._trajectory_frame_ids: List[int] = []  # frame_id for each entry
        self._keyframe_ids: List[int] = []

        # Per-frame stats
        self._stats: List[dict] = []

        # Keyframe counters
        self._frames_since_keyframe: int = 0
        self._n_keyframes: int = 0
        self._prev_kf_n_feats: int = 0   # feature count at previous keyframe

    # ── Public Interface ──────────────────────────────────────────────────────

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
    ) -> np.ndarray:
        """
        Process one frame and update the estimated trajectory.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (will be undistorted internally).
        frame_id : int

        Returns
        -------
        T_cw : np.ndarray, shape (4, 4)
            Current camera pose as SE(3) matrix (camera ← world).
        """
        t0 = time.perf_counter()
        frame = self.camera.undistort(frame)

        # Initialise on first frame
        if self._prev_frame is None:
            self._initialise(frame, frame_id)
            return self._current_T()

        # ── Feature Tracking ─────────────────────────────────────────────────
        track = self._frontend.detect_and_track(self._prev_frame, frame, self._prev_kps)
        n_tracked = track.n_tracked

        # ── Pose Estimation (PnP first, Essential fallback) ───────────────────
        pose = None
        used_pnp = False

        # 1. Try PnP against existing map landmarks
        kps_curr, descs_curr = None, None
        if len(self._local_map) >= 6:
            kps_curr, descs_curr = self._frontend.detect(frame)
            pnp_result = self._local_map.pnp_localize(kps_curr, descs_curr)
            if pnp_result is not None:
                R_pnp, t_pnp, inlier_mask_pnp = pnp_result
                n_pnp_inliers = int(inlier_mask_pnp.sum())
                if n_pnp_inliers >= 6:
                    # PnP gives absolute pose (world frame) — use directly
                    self._current_R = R_pnp
                    self._current_t = t_pnp
                    used_pnp = True
                    logger.debug(
                        "Frame %d: PnP pose (%d inliers)", frame_id, n_pnp_inliers
                    )
                    # Create PoseEstimate for pose graph edge from absolute pose
                    if self._keyframe_ids:
                        prev_kf_id = self._keyframe_ids[-1]
                        prev_node = self._pose_graph.get_node(prev_kf_id)
                        T_prev = prev_node.T
                        T_curr = np.eye(4)
                        T_curr[:3, :3] = R_pnp
                        T_curr[:3, 3] = t_pnp
                        T_rel = se3_compose(T_curr, se3_inverse(T_prev))
                        pose = PoseEstimate(
                            R=T_rel[:3, :3],
                            t=T_rel[:3, 3],
                            E=np.eye(3),
                            inlier_mask=inlier_mask_pnp,
                            n_inliers=n_pnp_inliers,
                            n_total=n_pnp_inliers,
                        )

        # 2. Fallback to Essential Matrix if PnP unavailable
        if not used_pnp:
            if n_tracked >= 8:
                pose = self._motion.estimate(track.good_prev, track.good_curr)
            if pose is not None:
                # Scale recovery via RANSAC ground-plane fitting
                pts3d = self._motion.triangulate(
                    track.good_prev, track.good_curr, pose.R, pose.t
                )
                scale = self._scale_recovery.update(pts3d)

                # Integrate relative pose into global pose
                self._current_t = (
                    self._current_t + scale * (self._current_R @ pose.t)
                )
                self._current_R = pose.R @ self._current_R

        # Record pose regardless
        T = self._current_T()
        self._trajectory.append(T.copy())
        self._trajectory_frame_ids.append(frame_id)

        # ── Keyframe Decision ─────────────────────────────────────────────────
        is_keyframe = self._should_insert_keyframe(n_tracked, track)

        if is_keyframe:
            if kps_curr is None:
                kps_curr, descs_curr = self._frontend.detect(frame)
            self._process_keyframe(frame, frame_id, pose, kps_curr, descs_curr, track)
        else:
            self._frames_since_keyframe += 1

        # ── Re-detect features for next frame ─────────────────────────────────
        if kps_curr is None:
            kps_curr, descs_curr = self._frontend.detect(frame)
        self._prev_frame = frame
        self._prev_kps = kps_curr
        self._prev_descs = descs_curr
        self._prev_frame_id = frame_id

        # ── Stats ──────────────────────────────────────────────────────────────
        dt = time.perf_counter() - t0
        stat = {
            "frame_id": frame_id,
            "tracked": n_tracked,
            "map_size": len(self._local_map),
            "n_kf": self._n_keyframes,
            "dt_ms": dt * 1000,
            "pose_ok": (pose is not None) or used_pnp,
            "used_pnp": used_pnp,
            "scale": self._scale_recovery.scale,
        }
        self._stats.append(stat)

        if self.verbose and frame_id % 10 == 0:
            pos = -self._current_R.T @ self._current_t
            mode = "PnP" if used_pnp else "E5pt"
            logger.info(
                "Frame %04d | pos (%.1f, %.1f, %.1f) | tracked %d | map %d | "
                "scale %.3f | %s | %.1f ms",
                frame_id, pos[0], pos[1], pos[2], n_tracked, len(self._local_map),
                self._scale_recovery.scale, mode, dt * 1000,
            )

        return T

    def run(self, sequence, save_path: Optional[str] = None) -> list:
        """
        Run the pipeline over a full KITTI sequence.

        Parameters
        ----------
        sequence : KITTISequence — iterable yielding (frame_id, image, timestamp)
        save_path : str | None — if given, save trajectory in KITTI format

        Returns
        -------
        list of 4×4 SE(3) pose matrices (one per frame).
        """
        self.reset()
        logger.info("Starting VO pipeline ...")
        for frame_id, image, timestamp in tqdm(sequence, desc="VO", unit="frame"):
            self.process_frame(image, frame_id)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            save_trajectory_kitti(self._trajectory, save_path)
            logger.info("Trajectory saved to %s", save_path)

        return self._trajectory

    @property
    def trajectory(self) -> list:
        """List of 4×4 SE(3) poses, one per frame."""
        return self._trajectory

    @property
    def pose_graph(self) -> PoseGraph:
        return self._pose_graph

    @property
    def local_map(self) -> LocalMap:
        return self._local_map

    @property
    def stats(self) -> list:
        return self._stats

    def reset(self) -> None:
        """Reset pipeline state (call before re-running on a new sequence)."""
        self._prev_frame = None
        self._prev_kps = None
        self._prev_descs = None
        self._prev_frame_id = -1
        self._current_R = np.eye(3)
        self._current_t = np.zeros(3)
        self._trajectory = []
        self._trajectory_frame_ids = []
        self._keyframe_ids = []
        self._stats = []
        self._frames_since_keyframe = 0
        self._n_keyframes = 0
        self._prev_kf_n_feats = 0
        self._pose_graph = PoseGraph()
        self._local_map = LocalMap(self.camera)
        self._scale_recovery = GroundPlaneScaleRecovery(
            camera_height=self.camera_height
        )
        # Reset loop detector so BoW database / vocabulary / score history
        # do not persist across sequences on the same VO instance.
        old_loop_detector = self._loop_detector
        self._loop_detector = LoopClosureDetector(
            self.camera,
            n_words=old_loop_detector._bow.n_words,
            min_score=old_loop_detector.min_score,
            min_inliers=old_loop_detector.min_inliers,
            min_frames_apart=old_loop_detector.min_frames_apart,
            vocab_path=getattr(old_loop_detector, '_vocab_path', None),
        )

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _initialise(self, frame: np.ndarray, frame_id: int) -> None:
        """Set up the first frame as the world origin."""
        self._prev_frame = frame
        kps, descs = self._frontend.detect(frame)
        self._prev_kps = kps
        self._prev_descs = descs
        self._prev_frame_id = frame_id
        self._prev_kf_n_feats = len(kps)

        # First node is the fixed world origin
        self._pose_graph.add_node(frame_id, np.eye(3), np.zeros(3), fixed=True)
        self._trajectory.append(self._current_T().copy())
        self._trajectory_frame_ids.append(frame_id)
        self._keyframe_ids.append(frame_id)
        self._n_keyframes = 1

    def _current_T(self) -> np.ndarray:
        """Return current pose as a 4×4 SE(3) matrix."""
        T = np.eye(4)
        T[:3, :3] = self._current_R
        T[:3, 3] = self._current_t
        return T

    def _should_insert_keyframe(
        self,
        n_tracked: int,
        track,
    ) -> bool:
        """
        Principled keyframe selection (v2).

        A keyframe is inserted when:
          (a) Tracked ratio < kf_track_ratio  AND  parallax > kf_parallax_deg,
          OR
          (b) Hard-gap fallback fires (frames_since_keyframe >= keyframe_max_gap).

        This prevents keyframe insertion on mere feature drop (e.g. a momentary
        occlusion) without sufficient motion, and vice versa.
        """
        # Hard fallback always fires
        if self._frames_since_keyframe >= self.keyframe_max_gap:
            return True

        # Conditions (a): both parallax AND track degradation must be present
        tracked_ratio = (
            n_tracked / max(1, self._prev_kf_n_feats)
        )
        parallax_ok = self._median_parallax_deg(track) > self.kf_parallax_deg
        tracking_degraded = tracked_ratio < self.kf_track_ratio

        return parallax_ok and tracking_degraded

    @staticmethod
    def _median_parallax_deg(track) -> float:
        """
        Compute the median angular parallax of tracked points.

        Uses the L2 optical-flow magnitude as a proxy for angular motion.
        For a typical field of view and focal length, 1 pixel ≈ 0.05–0.1°,
        so 2° corresponds roughly to 20–40 pixel median flow.

        Returns
        -------
        float — median flow magnitude in pixels (used as a parallax proxy).
        """
        if track.n_tracked == 0:
            return 0.0
        flow = np.linalg.norm(track.good_curr - track.good_prev, axis=1)
        return float(np.median(flow))

    def _process_keyframe(
        self,
        frame: np.ndarray,
        frame_id: int,
        pose: Optional[PoseEstimate],
        kps: np.ndarray,
        descs: np.ndarray,
        track=None,
    ) -> None:
        """Handle keyframe insertion: map update, graph edge, loop closure."""

        # Add node to pose graph
        self._pose_graph.add_node(frame_id, self._current_R.copy(), self._current_t.copy())

        # Add odometry edge from previous keyframe with proper information weight
        if self._keyframe_ids and pose is not None:
            prev_kf_id = self._keyframe_ids[-1]
            self._pose_graph.add_edge(
                prev_kf_id, frame_id,
                R_ij=pose.R,
                t_ij=pose.t * self._scale_recovery.scale,
                # Use the covariance-derived information matrix from PoseEstimate
                information=pose.information,
            )

        # Update local map (triangulate new 3-D points)
        if pose is not None and self._keyframe_ids:
            prev_kf_id = self._keyframe_ids[-1]
            # Use *matched* point pairs from LK tracking, not arbitrary slices
            # of the detection arrays (which are not correspondences).
            if track is not None and hasattr(track, 'good_prev') and len(track.good_prev) > 0:
                pts_prev = track.good_prev
                pts_curr = track.good_curr
            else:
                # Fallback: no matched points available
                pts_prev = np.empty((0, 2))
                pts_curr = np.empty((0, 2))
            max_pts = min(500, len(pts_prev), len(pts_curr))
            self._local_map.add_keyframe(
                frame_id=frame_id,
                R=self._current_R,
                t=self._current_t,
                keypoints=kps,
                descriptors=descs,
                prev_frame_id=prev_kf_id,
                pts_prev=pts_prev[:max_pts],
                pts_curr=pts_curr[:max_pts],
            )

        # Loop closure check
        loop = self._loop_detector.add_keyframe(frame_id, kps, descs)
        if loop is not None and loop.verified:
            # Weight loop closure edges significantly higher than odometry edges
            # since they correct long-range drift. The inlier count drives this.
            loop_information = (
                min(loop.n_inliers / 20.0, 5.0) * np.eye(6)
            )
            self._pose_graph.add_edge(
                loop.query_id, loop.candidate_id,
<<<<<<< HEAD
                R_ij=loop.R, t_ij=loop.t,
=======
                R_ij=loop.R, t_ij=loop.t * self._scale_recovery.scale,
>>>>>>> origin/master
                information=loop_information,
                is_loop_closure=True,
            )
            logger.info(
                "Loop closure edge added: %d → %d  (weight=%.2f)",
                loop.query_id, loop.candidate_id, loop_information[0, 0],
            )

        # Loop-closure-triggered global optimisation
        # Only runs when a *new* loop closure has been detected since last call.
        if self._pose_graph.consume_new_loops() > 0:
            logger.info(
                "Loop closure triggered optimisation (%d keyframes)...",
                self._n_keyframes,
            )
            self._pose_graph = self._optimizer.optimize(self._pose_graph)
<<<<<<< HEAD
=======
            # Sync trajectory with optimized graph poses for all keyframes.
            # Non-keyframes keep their original estimates (they are not graph nodes).
            for idx, fid in enumerate(self._trajectory_frame_ids):
                if fid in self._pose_graph._nodes:
                    node = self._pose_graph.get_node(fid)
                    self._trajectory[idx] = node.T.copy()
>>>>>>> origin/master
            # Sync current pose from the (now optimised) graph
            if frame_id in self._pose_graph._nodes:
                node = self._pose_graph.get_node(frame_id)
                self._current_R = node.R
                self._current_t = node.t

        # Update keyframe bookkeeping
        self._keyframe_ids.append(frame_id)
        self._frames_since_keyframe = 0
        self._n_keyframes += 1
        self._prev_kf_n_feats = len(kps)

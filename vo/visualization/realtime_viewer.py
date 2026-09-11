"""
Real-time Open3D Trajectory and Point Cloud Viewer
====================================================
"""
from __future__ import annotations

import logging
import threading
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _try_import_open3d():
    try:
        import open3d as o3d
        return o3d
    except ImportError:
        logger.warning("open3d not installed — real-time viewer disabled")
        return None


class RealtimeViewer:
    """
    Non-blocking Open3D window that updates the camera trajectory and point
    cloud in real time as the VO pipeline processes frames.

    Usage:
        viewer = RealtimeViewer()
        viewer.start()
        for frame_id, img, ts in sequence:
            pose = vo.process_frame(img, frame_id)
            viewer.update(vo.trajectory, vo.local_map.points)
        viewer.close()
    """

    def __init__(
        self,
        window_name: str = "Monocular VO — Real-time View",
        width: int = 1280,
        height: int = 720,
        point_size: float = 2.0,
    ) -> None:
        self._o3d = _try_import_open3d()
        self._window_name = window_name
        self._width = width
        self._height = height
        self._point_size = point_size

        self._vis = None
        self._traj_lines = None
        self._pcd = None
        self._gt_lines = None
        self._running = False
        self._lock = threading.Lock()

        # Pending updates (set from VO thread, consumed in GUI thread)
        self._pending_traj: Optional[List[np.ndarray]] = None
        self._pending_pts: Optional[np.ndarray] = None
        self._pending_gt: Optional[List[np.ndarray]] = None

    def start(self) -> None:
        """Open the visualisation window."""
        o3d = self._o3d
        if o3d is None:
            return

        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(
            window_name=self._window_name,
            width=self._width,
            height=self._height,
        )
        opt = self._vis.get_render_option()
        opt.background_color = np.array([0.05, 0.05, 0.08])
        opt.point_size = self._point_size

        # Empty geometries (updated later)
        self._pcd = o3d.geometry.PointCloud()
        self._traj_lines = o3d.geometry.LineSet()
        self._gt_lines = o3d.geometry.LineSet()

        self._vis.add_geometry(self._pcd)
        self._vis.add_geometry(self._traj_lines)
        self._vis.add_geometry(self._gt_lines)

        self._running = True

    def update(
        self,
        trajectory: List[np.ndarray],
        point_cloud: Optional[np.ndarray] = None,
        gt_poses: Optional[List[np.ndarray]] = None,
    ) -> None:
        """
        Queue a visualisation update.  Thread-safe; called from the VO thread.

        Parameters
        ----------
        trajectory : list of 4×4 SE(3) poses
        point_cloud : (N, 3) ndarray | None
        gt_poses : list of 4×4 SE(3) poses | None
        """
        with self._lock:
            self._pending_traj = trajectory
            self._pending_pts = point_cloud
            self._pending_gt = gt_poses

    def tick(self) -> bool:
        """
        Process one GUI event and apply pending updates.

        Call this from the **main thread** in a loop if running synchronously.
        Returns False if the window has been closed.
        """
        if self._vis is None or not self._running:
            return False

        with self._lock:
            traj = self._pending_traj
            pts = self._pending_pts
            gt = self._pending_gt
            self._pending_traj = None
            self._pending_pts = None
            self._pending_gt = None

        if traj is not None:
            self._update_trajectory(traj, gt)

        if pts is not None and len(pts) > 0:
            self._update_point_cloud(pts)

        self._vis.poll_events()
        self._vis.update_renderer()
        return self._running

    def run_blocking(self) -> None:
        """Block until the user closes the window."""
        if self._vis is None:
            return
        self._vis.run()
        self._vis.destroy_window()

    def close(self) -> None:
        """Destroy the window."""
        if self._vis is not None:
            self._vis.destroy_window()
        self._running = False

    def save_screenshot(self, path: str) -> None:
        """Save current view to a PNG file."""
        if self._vis is not None:
            self._vis.capture_screen_image(path)
            logger.info("Screenshot saved: %s", path)

    # ── Private Helpers ──────────────────────────────────────────────────────

    def _extract_positions(self, poses) -> np.ndarray:
        out = []
        for T in poses:
            T = np.asarray(T)
            if T.shape == (4, 4):
                R, t = T[:3, :3], T[:3, 3]
                out.append(-R.T @ t)
        return np.array(out, dtype=np.float64) if out else np.empty((0, 3))

    def _update_trajectory(
        self, traj: List[np.ndarray], gt: Optional[List[np.ndarray]]
    ) -> None:
        o3d = self._o3d
        est_pos = self._extract_positions(traj)

        if len(est_pos) < 2:
            return

        pts = est_pos.tolist()
        lines = [[i, i + 1] for i in range(len(pts) - 1)]
        colours_est = [[0.4, 0.5, 1.0]] * len(lines)  # blue-ish

        if gt:
            gt_pos = self._extract_positions(gt)
            if len(gt_pos) >= 2:
                gt_start = len(pts)
                pts += gt_pos.tolist()
                gt_lines = [[gt_start + i, gt_start + i + 1] for i in range(len(gt_pos) - 1)]
                lines += gt_lines
                colours_est += [[0.0, 1.0, 0.5]] * len(gt_lines)  # green

        self._traj_lines.points = o3d.utility.Vector3dVector(pts)
        self._traj_lines.lines = o3d.utility.Vector2iVector(lines)
        self._traj_lines.colors = o3d.utility.Vector3dVector(colours_est)
        self._vis.update_geometry(self._traj_lines)

    def _update_point_cloud(self, pts: np.ndarray) -> None:
        o3d = self._o3d
        valid = pts[~np.any(np.isnan(pts), axis=1)]
        if len(valid) == 0:
            return
        self._pcd.points = o3d.utility.Vector3dVector(valid[:5000])  # cap for speed
        n = len(self._pcd.points)
        # Depth-coloured: closer = warmer
        depths = valid[:n, 2]
        d_min, d_max = depths.min(), depths.max() + 1e-6
        t = (depths - d_min) / (d_max - d_min)
        colours = np.stack([t, 0.5 * np.ones(n), 1 - t], axis=1)
        self._pcd.colors = o3d.utility.Vector3dVector(colours)
        self._vis.update_geometry(self._pcd)

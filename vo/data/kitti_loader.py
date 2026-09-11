"""
KITTI Odometry Dataset Loader
==============================
Reads KITTI odometry sequences in their standard directory layout:

    data/kitti/
    └── sequences/
        └── 00/
            ├── image_0/          <- left greyscale camera frames
            │   ├── 000000.png
            │   └── ...
            ├── image_2/          <- left colour camera frames (optional)
            ├── calib.txt         <- camera projection matrices
            └── times.txt         <- per-frame timestamps (seconds)
    └── poses/
        └── 00.txt                <- ground-truth 4×4 SE(3) poses (12 values/row)

Reference
---------
Geiger et al., "Are we ready for Autonomous Driving?  The KITTI Vision
Benchmark Suite", CVPR 2012.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import numpy as np


class KITTISequence:
    """
    Iterates over a single KITTI odometry sequence and exposes calibration
    data and ground-truth poses.

    Parameters
    ----------
    dataset_root : str | Path
        Root directory containing ``sequences/`` and ``poses/`` subdirs.
    seq_id : str | int
        Sequence identifier, e.g. ``'00'`` or ``0``.
    camera : int
        Camera index:  0 = left greyscale, 1 = right greyscale,
                       2 = left colour, 3 = right colour.  Defaults to 0.
    max_frames : int | None
        Limit the number of frames for quick testing.  ``None`` = all.
    """

    # ── KITTI Camera Intrinsics for sequence 00 (from calib.txt P0) ──────────
    # These are embedded as a reference; the loader always reads calib.txt.
    _KNOWN_SHAPES = {
        "00": (376, 1241),   # (H, W) – used for sanity check only
        "02": (376, 1241),
        "05": (376, 1241),
        "07": (376, 1241),
        "08": (376, 1241),
    }

    def __init__(
        self,
        dataset_root: str | Path,
        seq_id: str | int,
        camera: int = 0,
        max_frames: Optional[int] = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.seq_id = f"{int(seq_id):02d}"
        self.camera = camera
        self.max_frames = max_frames

        self._seq_dir = self.dataset_root / "sequences" / self.seq_id
        self._poses_file = self.dataset_root / "poses" / f"{self.seq_id}.txt"
        self._image_dir = self._seq_dir / f"image_{camera}"
        self._calib_file = self._seq_dir / "calib.txt"
        self._times_file = self._seq_dir / "times.txt"

        self._validate_paths()
        self._K, self._P, self._dist_coeffs = self._load_calibration()
        self._timestamps = self._load_timestamps()
        self._image_paths = sorted(self._image_dir.glob("*.png"))
        if max_frames is not None:
            self._image_paths = self._image_paths[:max_frames]

        self._gt_poses: Optional[list[np.ndarray]] = None  # lazy load

    # ── Public Interface ─────────────────────────────────────────────────────

    @property
    def K(self) -> np.ndarray:
        """3×3 camera intrinsic matrix for the selected camera."""
        return self._K.copy()

    @property
    def P(self) -> np.ndarray:
        """3×4 full projection matrix P = K[R|t] from calib.txt."""
        return self._P.copy()

    @property
    def dist_coeffs(self) -> np.ndarray:
        """
        Distortion coefficients.

        KITTI images are already rectified (undistorted), so this returns
        zeros.  Kept in the API so the Camera class doesn't need special-casing.
        """
        return self._dist_coeffs.copy()

    @property
    def timestamps(self) -> np.ndarray:
        """Array of per-frame timestamps in seconds, shape (N,)."""
        return self._timestamps[: len(self._image_paths)]

    def __len__(self) -> int:
        return len(self._image_paths)

    def __iter__(self) -> Generator[Tuple[int, np.ndarray, float], None, None]:
        """
        Yield ``(frame_id, image_bgr, timestamp)`` for every frame.

        Images are loaded as 8-bit greyscale converted to BGR for consistency
        with the rest of the pipeline (OpenCV convention).
        """
        for idx, img_path in enumerate(self._image_paths):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise IOError(f"Cannot read image: {img_path}")
            # Convert to BGR so all pipeline modules receive consistent format
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            ts = float(self._timestamps[idx]) if idx < len(self._timestamps) else float(idx)
            yield idx, img_bgr, ts

    def ground_truth(self) -> list[np.ndarray]:
        """
        Return ground-truth camera poses as a list of 4×4 SE(3) matrices.

        Each matrix T_i represents the transformation from the camera frame
        at time i to the world frame.  If the poses file does not exist
        (e.g., for test sequences 11–21), returns an empty list.
        """
        if self._gt_poses is not None:
            return self._gt_poses

        if not self._poses_file.exists():
            self._gt_poses = []
            return self._gt_poses

        poses = []
        with open(self._poses_file) as f:
            for line in f:
                vals = list(map(float, line.strip().split()))
                if len(vals) != 12:
                    continue
                T = np.eye(4, dtype=np.float64)
                T[:3, :] = np.array(vals, dtype=np.float64).reshape(3, 4)
                poses.append(T)

        n = len(self._image_paths)
        self._gt_poses = poses[:n]
        return self._gt_poses

    def camera_height(self) -> float:
        """
        Return the approximate height of camera 0 above the ground plane
        in metres.  Used for monocular scale recovery.

        KITTI sequences use a roughly 1.65 m mounting height.  This is the
        value used in the Geiger et al. reference implementation.
        """
        return 1.65  # metres

    # ── Private Helpers ──────────────────────────────────────────────────────

    def _validate_paths(self) -> None:
        if not self._seq_dir.exists():
            raise FileNotFoundError(
                f"KITTI sequence directory not found: {self._seq_dir}\n"
                f"Run  python scripts/download_kitti.py  to download the dataset."
            )
        if not self._image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self._image_dir}\n"
                f"Make sure camera index {self.camera} is correct."
            )
        if not self._calib_file.exists():
            raise FileNotFoundError(f"Calibration file not found: {self._calib_file}")

    def _load_calibration(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Parse calib.txt and return (K, P, dist_coeffs).

        calib.txt format (one line per camera):
            P0: f00 f01 f02 f03 f10 f11 ... f23   (12 floats = 3×4 matrix)

        The intrinsic matrix K is extracted from P as:
            K = P[:3, :3]  (valid when principal point offset baseline is zero)

        For camera 0 (left greyscale), the baseline is 0 so K = P[:3,:3].
        """
        cam_key = f"P{self.camera}:"
        P = None
        with open(self._calib_file) as f:
            for line in f:
                if line.startswith(cam_key):
                    vals = list(map(float, line.strip().split()[1:]))
                    P = np.array(vals, dtype=np.float64).reshape(3, 4)
                    break

        if P is None:
            raise ValueError(f"Could not find {cam_key} in {self._calib_file}")

        K = P[:3, :3].copy()
        # KITTI images are pre-rectified; distortion is zero.
        dist_coeffs = np.zeros(5, dtype=np.float64)
        return K, P, dist_coeffs

    def _load_timestamps(self) -> np.ndarray:
        if not self._times_file.exists():
            # Fall back to integer frame indices
            return np.arange(10_000, dtype=np.float64)
        with open(self._times_file) as f:
            times = [float(line.strip()) for line in f if line.strip()]
        return np.array(times, dtype=np.float64)


# ── Convenience factory ────────────────────────────────────────────────────────

def load_sequence(
    dataset_root: str | Path,
    seq_id: str | int = "00",
    camera: int = 0,
    max_frames: Optional[int] = None,
) -> KITTISequence:
    """Create and return a :class:`KITTISequence` instance."""
    return KITTISequence(dataset_root, seq_id, camera, max_frames)

"""
Generic Video Loader for Arbitrary Footage
==========================================

Reads any video source supported by OpenCV:
  - Local video files: MP4, AVI, MOV, MKV, WebM, etc.
  - Live webcam: device index (0, 1, ...)
  - IP camera / RTSP streams: rtsp://...
  - YouTube and other streams via URL (requires opencv-contrib or yt-dlp)

Camera Calibration
------------------
Unlike KITTI, arbitrary videos don't come with calibration files.
This loader accepts calibration in three ways (priority order):

  1. A .json or .npz file produced by `scripts/calibrate_camera.py`
  2. Explicit fx, fy, cx, cy, dist_coeffs keyword arguments
  3. Automatic estimation from image dimensions:

       fx = fy ≈ max(width, height)   (assumes ~53° diagonal FOV, common in phone/DSLRs)
       cx = width / 2,  cy = height / 2
       dist_coeffs = zeros

     This estimate produces usable results on typical consumer cameras.
     ATE will be higher than with a calibrated camera, but the pipeline
     will run and produce a visually plausible trajectory.

References
----------
• Zhang, "A Flexible New Technique for Camera Calibration", TPAMI 2000.
• Hartley & Zisserman, "Multiple View Geometry", §6.1 (camera model).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import numpy as np

from vo.camera import Camera

logger = logging.getLogger(__name__)


# ── Calibration Helpers ────────────────────────────────────────────────────────

def estimate_K(width: int, height: int) -> np.ndarray:
    """
    Estimate the intrinsic matrix from image dimensions alone.

    Uses the heuristic  fx = fy = max(w, h)  which corresponds to a
    horizontal field of view of roughly 53° for a 4:3 image — a reasonable
    approximation for most smartphones and consumer cameras.

    For best results, calibrate with a checkerboard using
    ``scripts/calibrate_camera.py`` and pass the resulting file via
    ``--calibration``.

    Parameters
    ----------
    width, height : int

    Returns
    -------
    K : np.ndarray, shape (3, 3)
    """
    f = float(max(width, height))
    cx = width / 2.0
    cy = height / 2.0
    K = np.array([
        [f,   0., cx],
        [0.,  f,  cy],
        [0.,  0., 1.],
    ], dtype=np.float64)
    logger.warning(
        "Using estimated camera intrinsics (fx=fy=%.0f, cx=%.0f, cy=%.0f). "
        "For better accuracy, calibrate with scripts/calibrate_camera.py",
        f, cx, cy,
    )
    return K


def load_calibration(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load camera calibration from a .json or .npz file.

    JSON format (produced by calibrate_camera.py):
        {
          "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
          "dist_coeffs": [k1, k2, p1, p2, k3]
        }

    NPZ format: numpy arrays with keys "K" and "dist_coeffs".

    Parameters
    ----------
    path : str

    Returns
    -------
    K : np.ndarray (3, 3)
    dist_coeffs : np.ndarray (4,) or (5,) or (8,)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")

    if p.suffix == ".json":
        with open(p) as f:
            data = json.load(f)
        K = np.array(data["K"], dtype=np.float64)
        dist = np.array(data["dist_coeffs"], dtype=np.float64)
    elif p.suffix in (".npz", ".npy"):
        data = np.load(p)
        K = data["K"].astype(np.float64)
        dist = data["dist_coeffs"].astype(np.float64)
    else:
        raise ValueError(f"Unsupported calibration format: {p.suffix}. Use .json or .npz")

    logger.info(
        "Loaded calibration from %s  fx=%.1f fy=%.1f cx=%.1f cy=%.1f",
        path, K[0,0], K[1,1], K[0,2], K[1,2],
    )
    return K, dist


# ── Video Loader ───────────────────────────────────────────────────────────────

class VideoLoader:
    """
    Frame-by-frame reader for any OpenCV-compatible video source.

    Parameters
    ----------
    source : str | int
        Video file path, webcam device index (0, 1 ...), or RTSP URL.
    calibration_path : str | None
        Path to a .json or .npz calibration file. If None, intrinsics are
        estimated from the first frame's dimensions.
    fx, fy, cx, cy : float | None
        Manual intrinsic overrides. Used only when calibration_path is None.
    dist_coeffs : array-like | None
        Lens distortion coefficients [k1, k2, p1, p2, k3].
        Defaults to zeros (no distortion, e.g. already-corrected footage).
    max_frames : int | None
        Stop after this many frames.
    resize : tuple (W, H) | None
        Resize every frame to this resolution before yielding.
        Useful for speeding up processing on large footage (e.g. 4K → 1280×720).
    skip_frames : int
        Yield every N-th frame (default 1 = every frame, 2 = every other, etc.).
        Useful for slow-motion or high-FPS footage.
    camera_height : float
        Height of the camera above the ground plane in metres.
        Used for scale recovery. Default 1.65 m (car-mounted).
        Use 1.2–1.5 m for hand-held, 0.5–1.0 m for drone footage, etc.
    """

    def __init__(
        self,
        source: str | int,
        calibration_path: Optional[str] = None,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        dist_coeffs: Optional[list] = None,
        max_frames: Optional[int] = None,
        resize: Optional[Tuple[int, int]] = None,
        skip_frames: int = 1,
        camera_height: float = 1.65,
    ) -> None:
        self.source = source
        self.max_frames = max_frames
        self.resize = resize
        self.skip_frames = max(1, skip_frames)
        self.camera_height = camera_height

        # Open video
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise IOError(
                f"Cannot open video source: {source!r}\n"
                f"Supported: file paths (MP4/AVI/MOV/MKV), webcam index (0), "
                f"RTSP URLs (rtsp://...)"
            )

        # Probe frame dimensions
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # If resizing, adjust dimensions for calibration
        if resize is not None:
            w_out, h_out = resize
        else:
            w_out, h_out = w, h

        # Resolve calibration
        if calibration_path is not None:
            K, dist = load_calibration(calibration_path)
            # Scale K if we are resizing
            if resize is not None and (w_out != w or h_out != h):
                K = self._scale_K(K, w, h, w_out, h_out)
        elif all(v is not None for v in [fx, fy, cx, cy]):
            K = np.array([
                [fx,  0., cx],
                [0.,  fy, cy],
                [0.,  0., 1.],
            ], dtype=np.float64)
            dist = np.array(dist_coeffs or [0., 0., 0., 0., 0.], dtype=np.float64)
            if resize is not None and (w_out != w or h_out != h):
                K = self._scale_K(K, w, h, w_out, h_out)
            logger.info("Using manually supplied intrinsics")
        else:
            K = estimate_K(w_out, h_out)
            dist = np.zeros(5, dtype=np.float64)

        self._camera = Camera(K=K, dist_coeffs=dist)

        logger.info(
            "VideoLoader: source=%s  %dx%d @ %.1f fps  total_frames=%s",
            source, w_out, h_out, self._fps,
            str(self._total_frames) if self._total_frames > 0 else "unknown (live)",
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def camera(self) -> Camera:
        """Calibrated :class:`~vo.camera.Camera` for this video."""
        return self._camera

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> int:
        return self._total_frames

    def __len__(self) -> int:
        if self._total_frames > 0 and self.max_frames is not None:
            return min(self._total_frames, self.max_frames * self.skip_frames)
        return self._total_frames

    def __iter__(self) -> Generator[Tuple[int, np.ndarray, float], None, None]:
        """
        Yield ``(frame_id, image_bgr, timestamp_seconds)`` for each frame.

        frame_id is the sequential output frame index (0-based, accounting for
        skip_frames), not the raw video frame index.
        """
        raw_idx = 0
        output_idx = 0
        max_out = self.max_frames if self.max_frames is not None else float("inf")

        while output_idx < max_out:
            ret, frame = self._cap.read()
            if not ret:
                break   # end of file or stream disconnected

            # Skip frames for high-FPS or slow-motion footage
            if raw_idx % self.skip_frames != 0:
                raw_idx += 1
                continue

            # Resize if requested
            if self.resize is not None:
                frame = cv2.resize(frame, self.resize, interpolation=cv2.INTER_LINEAR)

            timestamp = raw_idx / self._fps if self._fps > 0 else float(raw_idx)
            yield output_idx, frame, timestamp

            raw_idx += 1
            output_idx += 1

        self._cap.release()

    def ground_truth(self) -> list:
        """
        Compatibility shim with KITTISequence API.
        Generic video has no ground truth — returns empty list.
        """
        return []

    def camera_height(self) -> float:
        return self.camera_height   # type: ignore[return-value]

    def release(self) -> None:
        """Release the underlying VideoCapture."""
        if self._cap.isOpened():
            self._cap.release()

    def __del__(self):
        self.release()

    # ── Private Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _scale_K(
        K: np.ndarray,
        orig_w: int, orig_h: int,
        new_w: int, new_h: int,
    ) -> np.ndarray:
        """
        Scale the intrinsic matrix K to match a resized image.

        When you resize an image by scale (sx, sy):
            fx' = fx * sx,  fy' = fy * sy
            cx' = cx * sx,  cy' = cy * sy
        """
        sx = new_w / orig_w
        sy = new_h / orig_h
        K_scaled = K.copy()
        K_scaled[0, 0] *= sx   # fx
        K_scaled[1, 1] *= sy   # fy
        K_scaled[0, 2] *= sx   # cx
        K_scaled[1, 2] *= sy   # cy
        logger.info(
            "Scaled K for resize %dx%d -> %dx%d  (sx=%.3f, sy=%.3f)",
            orig_w, orig_h, new_w, new_h, sx, sy,
        )
        return K_scaled

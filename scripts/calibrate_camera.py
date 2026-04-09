#!/usr/bin/env python3
"""
OpenCV Checkerboard Camera Calibration Tool
============================================

Calibrates a camera by detecting a checkerboard pattern in a video file
or live webcam stream, then saves the intrinsic matrix K and distortion
coefficients to a .json file.

Usage — calibrate from a video of a checkerboard:
    python scripts/calibrate_camera.py \\
        --source my_checkerboard_video.mp4 \\
        --output calib/my_camera.json \\
        --rows 9 --cols 6 \\
        --square_size 25.0   # square size in mm (any unit; only relative scale matters)

Usage — calibrate from webcam (move the checkerboard around):
    python scripts/calibrate_camera.py \\
        --source 0 \\
        --output calib/webcam.json \\
        --rows 9 --cols 6

Usage — calibrate from a folder of images:
    python scripts/calibrate_camera.py \\
        --source images/calib/ \\
        --output calib/camera.json

What is a checkerboard?
-----------------------
Print a standard checkerboard pattern from:
    https://calib.io/pages/camera-calibration-pattern-generator
Use 9×6 inner corners (= 10×7 squares), printed on A4 / Letter.
Tape it flat to a rigid surface (cardboard). Move it around in front
of the camera at various angles and distances.

Theory
------
Zhang's method (CVPR 1998 / TPAMI 2000) fits a homography from each
checkerboard view to the image plane, then jointly optimises K and
distortion coefficients via Levenberg-Marquardt.

Minimum: ~10 diverse views (different angles, distances, tilts).
Recommended: 30–50 views for sub-pixel accuracy.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser(
        description="Camera calibration from checkerboard video/images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--source", "-s", required=True,
        help="Video file, webcam index (0/1/...), image folder, or RTSP URL",
    )
    p.add_argument("--output", "-o", default="calib/camera.json",
                   help="Output calibration .json file (default: calib/camera.json)")
    p.add_argument("--rows", "-r", type=int, default=9,
                   help="Number of inner corner rows (default: 9)")
    p.add_argument("--cols", "-c", type=int, default=6,
                   help="Number of inner corner columns (default: 6)")
    p.add_argument("--square_size", type=float, default=25.0,
                   help="Checkerboard square size in mm (default: 25.0)")
    p.add_argument("--min_frames", type=int, default=20,
                   help="Minimum frames with detected corners needed (default: 20)")
    p.add_argument("--max_frames", type=int, default=100,
                   help="Maximum frames to collect (default: 100)")
    p.add_argument("--stride", type=int, default=10,
                   help="Process every N-th frame from video (default: 10)")
    p.add_argument("--preview", action="store_true",
                   help="Show live preview window while collecting frames")
    p.add_argument("--log_level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def open_source(source_str: str):
    """
    Open video/image source. Returns (cap_or_images, is_video).

    For image folders, returns a list of file paths.
    """
    # Try as integer (webcam index)
    try:
        idx = int(source_str)
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            return cap, True
    except ValueError:
        pass

    # Image folder
    p = Path(source_str)
    if p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        images = sorted([f for f in p.iterdir() if f.suffix.lower() in exts])
        if images:
            return images, False

    # Video file or URL
    cap = cv2.VideoCapture(source_str)
    if cap.isOpened():
        return cap, True

    raise IOError(
        f"Cannot open source: {source_str!r}\n"
        f"Supported: file path, webcam index (0), RTSP URL, or image folder."
    )


def collect_frames(source_str, stride, max_frames, preview, log):
    """Read frames from source — video, webcam, or image list."""
    src, is_video = open_source(source_str)
    frames = []
    collected = 0
    raw_idx = 0

    if is_video:
        cap = src
        log.info("Reading from video/webcam source...")
        while collected < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if raw_idx % stride == 0:
                frames.append(frame)
                collected += 1
            raw_idx += 1
        cap.release()
    else:
        image_paths = src
        log.info("Reading %d images from folder...", len(image_paths))
        for path in image_paths[:max_frames * stride:stride]:
            frame = cv2.imread(str(path))
            if frame is not None:
                frames.append(frame)
                collected += 1

    log.info("Collected %d candidate frames", collected)
    return frames


def find_corners(frames, rows, cols, square_size, preview, log):
    """
    Detect checkerboard corners in each frame.

    Returns (obj_points, img_points, img_size).
    """
    pattern = (cols, rows)   # OpenCV: (cols, rows) = (width, height) in corners
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001
    )

    # 3D world points for a single checkerboard view
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size

    obj_points = []
    img_points = []
    img_size = None
    n_found = 0

    for i, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]   # (width, height)

        ret, corners = cv2.findChessboardCorners(gray, pattern, None)
        if ret:
            # Refine to sub-pixel accuracy
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )
            obj_points.append(objp)
            img_points.append(corners_refined)
            n_found += 1

            if preview:
                vis = frame.copy()
                cv2.drawChessboardCorners(vis, pattern, corners_refined, ret)
                cv2.putText(
                    vis, f"Found: {n_found}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                )
                cv2.imshow("Calibration", vis)
                cv2.waitKey(100)

        if (i + 1) % 10 == 0:
            log.info("  Frame %d/%d — corners found in %d frames",
                     i + 1, len(frames), n_found)

    if preview:
        cv2.destroyAllWindows()

    log.info("Checkerboard detected in %d / %d frames", n_found, len(frames))
    return obj_points, img_points, img_size


def calibrate(obj_points, img_points, img_size, log):
    """Run cv2.calibrateCamera and return (K, dist, rms_error)."""
    log.info("Running calibration (Zhang's method)...")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None
    )
    return K, dist.ravel(), rms


def save_calibration(K, dist, rms, img_size, output_path, log):
    """Save K and dist_coeffs to a JSON file."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "K": K.tolist(),
        "dist_coeffs": dist.tolist(),
        "image_size": list(img_size),
        "rms_reprojection_error_px": round(float(rms), 5),
        "fx": round(float(K[0, 0]), 4),
        "fy": round(float(K[1, 1]), 4),
        "cx": round(float(K[0, 2]), 4),
        "cy": round(float(K[1, 2]), 4),
    }

    with open(out, "w") as f:
        json.dump(data, f, indent=2)

    log.info("=" * 50)
    log.info("Calibration saved to: %s", out.resolve())
    log.info("  fx = %.2f px", K[0, 0])
    log.info("  fy = %.2f px", K[1, 1])
    log.info("  cx = %.2f px", K[0, 2])
    log.info("  cy = %.2f px", K[1, 2])
    log.info("  dist = %s", dist.round(6).tolist())
    log.info("  RMS reprojection error: %.4f px", rms)
    if rms < 0.5:
        log.info("  Quality: EXCELLENT (< 0.5 px)")
    elif rms < 1.0:
        log.info("  Quality: GOOD (< 1.0 px)")
    else:
        log.warning("  Quality: POOR (> 1.0 px) — try more/better checkerboard views")
    log.info("=" * 50)


def main():
    args = parse_args()
    setup_logging(args.log_level)
    log = logging.getLogger("calibrate_camera")

    log.info("Camera Calibration Tool")
    log.info("  Source     : %s", args.source)
    log.info("  Pattern    : %d×%d inner corners", args.rows, args.cols)
    log.info("  Square size: %.1f mm", args.square_size)
    log.info("  Output     : %s", args.output)

    # 1. Collect frames
    frames = collect_frames(
        args.source, args.stride, args.max_frames, args.preview, log
    )
    if not frames:
        log.error("No frames collected from source.")
        sys.exit(1)

    # 2. Find checkerboard corners
    obj_pts, img_pts, img_size = find_corners(
        frames, args.rows, args.cols, args.square_size, args.preview, log
    )

    if len(obj_pts) < args.min_frames:
        log.error(
            "Only %d frames with detected corners (need at least %d). "
            "Tips:\n"
            "  • Use a well-lit, flat-printed checkerboard\n"
            "  • Move it to different angles (tilt ~30–45°)\n"
            "  • Cover the full image area\n"
            "  • Reduce --stride to process more frames",
            len(obj_pts), args.min_frames,
        )
        sys.exit(1)

    # 3. Calibrate
    K, dist, rms = calibrate(obj_pts, img_pts, img_size, log)

    # 4. Save
    save_calibration(K, dist, rms, img_size, args.output, log)

    log.info(
        "\nUsage:\n"
        "  python scripts/run_video.py --source <your_video.mp4> "
        "--calibration %s",
        args.output,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Run Monocular Visual Odometry on Any Video
==========================================

Run on a video file with auto-estimated calibration:
    python scripts/run_video.py --source driving.mp4

Run on a webcam:
    python scripts/run_video.py --source 0 --camera_height 1.2

Run with a calibration file (recommended for best accuracy):
    python scripts/run_video.py \\
        --source dashcam.mp4 \\
        --calibration calib/my_camera.json \\
        --camera_height 1.4

Run on phone footage (portrait → landscape, handheld):
    python scripts/run_video.py \\
        --source phone_walk.mp4 \\
        --calibration calib/phone.json \\
        --camera_height 1.5 \\
        --resize 960 540 \\
        --skip 2

Run on drone footage:
    python scripts/run_video.py \\
        --source drone.mp4 \\
        --calibration calib/drone.json \\
        --camera_height 20.0 \\
        --resize 1280 720

Camera Height Guide
-------------------
--camera_height controls the metric scale of the trajectory.
It represents the height of your camera above a roughly flat surface:

  Driving dashcam:     1.2–1.7 m
  Handheld walking:    1.3–1.6 m (camera at eye level)
  Cycling helmet cam:  1.0–1.2 m
  Drone:               actual altitude in metres (set --camera_height accordingly)
  Indoor robot:        0.3–0.8 m
  Phone on table:      0.05–0.2 m

Note: if scale looks wrong (trajectory 10× too small/large), adjust this value.

Calibration File
----------------
Without a calibration file the pipeline uses:
    fx = fy = max(width, height)    (good first guess for most cameras)
    cx = width/2,  cy = height/2
    dist_coeffs = [0, 0, 0, 0, 0]

For better accuracy, calibrate once with:
    python scripts/calibrate_camera.py --source calib_video.mp4 --output calib/my_camera.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from vo.data.video_loader import VideoLoader
from vo.odometry import VisualOdometry
from vo.evaluation.metrics import save_trajectory_kitti
from vo.visualization.trajectory_plot import plot_trajectory_2d, plot_trajectory_3d


def parse_args():
    p = argparse.ArgumentParser(
        description="Monocular Visual Odometry on any video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Source
    p.add_argument("--source", "-s", required=True,
                   help="Video file path, webcam index (0/1), or RTSP URL")

    # Calibration
    cal = p.add_argument_group("Camera Calibration")
    cal.add_argument("--calibration", "-C", default=None,
                     help="Path to .json calibration file (from calibrate_camera.py)")
    cal.add_argument("--fx", type=float, default=None, help="Focal length x in pixels")
    cal.add_argument("--fy", type=float, default=None, help="Focal length y in pixels")
    cal.add_argument("--cx", type=float, default=None, help="Principal point x")
    cal.add_argument("--cy", type=float, default=None, help="Principal point y")
    cal.add_argument("--dist", nargs="+", type=float, default=None,
                     help="Distortion coefficients k1 k2 p1 p2 [k3]")

    # Physical parameters
    p.add_argument("--camera_height", type=float, default=1.65,
                   help="Camera height above ground in metres (default 1.65 = car)")

    # Video processing
    vid = p.add_argument_group("Video Options")
    vid.add_argument("--resize", nargs=2, type=int, default=None, metavar=("W", "H"),
                     help="Resize frames to W×H before processing (e.g. 1280 720)")
    vid.add_argument("--skip", type=int, default=1,
                     help="Process every N-th frame (default 1). Use 2-4 for slow scenes")
    vid.add_argument("--max_frames", "-n", type=int, default=None,
                     help="Stop after N output frames")
    vid.add_argument("--start_frame", type=int, default=0,
                     help="Skip first N raw frames before processing")

    # VO parameters
    vo_grp = p.add_argument_group("VO Pipeline")
    vo_grp.add_argument("--vocab", default=None,
                        help="Pre-trained BoW vocabulary .npz file")
    vo_grp.add_argument("--kf_parallax", type=float, default=2.0,
                        help="Keyframe parallax threshold in pixels (default 2.0)")
    vo_grp.add_argument("--kf_track_ratio", type=float, default=0.80,
                        help="Keyframe tracking ratio floor (default 0.80)")

    # Output
    out_grp = p.add_argument_group("Output")
    out_grp.add_argument("--output", "-o", default="results/video",
                         help="Output directory (default: results/video)")
    out_grp.add_argument("--no_viewer", action="store_true",
                         help="Disable real-time Open3D viewer")
    out_grp.add_argument("--preview", action="store_true",
                         help="Show live OpenCV preview with tracked features")

    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--log_level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    args = parse_args()
    setup_logging(args.log_level)
    log = logging.getLogger("run_video")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Monocular Visual Odometry — Generic Video Mode")
    log.info("  Source         : %s", args.source)
    log.info("  Calibration    : %s", args.calibration or "auto-estimated")
    log.info("  Camera height  : %.2f m", args.camera_height)
    log.info("  Output         : %s", output_dir)
    log.info("=" * 60)

    # ── Load Video ─────────────────────────────────────────────────────────────
    resize = tuple(args.resize) if args.resize else None
    try:
        seq = VideoLoader(
            source=args.source,
            calibration_path=args.calibration,
            fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
            dist_coeffs=args.dist,
            max_frames=args.max_frames,
            resize=resize,
            skip_frames=args.skip,
            camera_height=args.camera_height,
        )
    except IOError as exc:
        log.error("%s", exc)
        sys.exit(1)

    camera = seq.camera
    log.info("Camera: %s", camera)

    # ── Build Pipeline ─────────────────────────────────────────────────────────
    vo = VisualOdometry(
        camera=camera,
        camera_height=args.camera_height,
        kf_parallax_deg=args.kf_parallax,
        kf_track_ratio=args.kf_track_ratio,
        vocab_path=args.vocab,
        verbose=args.verbose,
    )

    # ── Optional Real-time Viewer ──────────────────────────────────────────────
    viewer = None
    if not args.no_viewer:
        try:
            from vo.visualization.realtime_viewer import RealtimeViewer
            viewer = RealtimeViewer(window_name="VO — Video Mode")
            viewer.start()
            log.info("Open3D viewer started")
        except Exception as exc:
            log.warning("Could not start viewer: %s — running headless", exc)

    # ── Run Pipeline ───────────────────────────────────────────────────────────
    preview_win = "VO Preview" if args.preview else None

    for frame_id, image, timestamp in seq:
        # Process frame
        vo.process_frame(image, frame_id)

        # Live feature preview
        if args.preview:
            stats = vo.stats
            if stats:
                s = stats[-1]
                label = (
                    f"Frame {frame_id:5d} | "
                    f"tracked {s['tracked']:4d} | "
                    f"map {s['map_size']:4d} | "
                    f"scale {s['scale']:.3f} | "
                    f"{'PnP' if s.get('used_pnp') else 'E5pt'}"
                )
            else:
                label = f"Frame {frame_id}"
            vis = image.copy()
            cv2.putText(vis, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2)

            # Draw current position on minimap overlay (bottom-right corner)
            traj = vo.trajectory
            if len(traj) >= 2:
                _draw_minimap(vis, traj)

            cv2.imshow(preview_win, vis)
            key = cv2.waitKey(1)
            if key == ord("q") or key == 27:   # q or ESC to quit
                log.info("User quit preview")
                break

        # Update 3D viewer
        if viewer is not None:
            viewer.update(
                trajectory=vo.trajectory,
                point_cloud=vo.local_map.points,
            )
            if not viewer.tick():
                log.info("Viewer closed — stopping")
                break

    if args.preview:
        cv2.destroyAllWindows()

    # ── Save Results ───────────────────────────────────────────────────────────
    traj_path = str(output_dir / "trajectory.txt")
    save_trajectory_kitti(vo.trajectory, traj_path)
    log.info("Trajectory saved: %s  (%d poses)", traj_path, len(vo.trajectory))

    # ── Plots ──────────────────────────────────────────────────────────────────
    log.info("Generating trajectory plots...")
    src_name = Path(str(args.source)).stem if isinstance(args.source, str) else "webcam"

    plot_trajectory_2d(
        vo.trajectory, None,
        title=f"Trajectory — {src_name} (top view)",
        save_path=str(output_dir / "trajectory_2d.png"),
    )
    plot_trajectory_3d(
        vo.trajectory, None,
        title=f"Trajectory — {src_name} (3D)",
        save_path=str(output_dir / "trajectory_3d.png"),
    )

    if viewer is not None:
        viewer.save_screenshot(str(output_dir / "viewer_screenshot.png"))
        viewer.close()

    log.info("✅ Done. Results in: %s", output_dir.resolve())
    log.info("   Frames processed: %d", len(vo.trajectory))
    if vo.stats:
        import statistics
        dts = [s["dt_ms"] for s in vo.stats]
        log.info(
            "   Speed: median %.1f ms/frame  (%.1f fps)",
            statistics.median(dts), 1000.0 / max(statistics.median(dts), 1)
        )


def _draw_minimap(frame: np.ndarray, trajectory: list, size: int = 200) -> None:
    """
    Draw a top-down trajectory minimap in the bottom-right corner of the frame.

    Parameters
    ----------
    frame : np.ndarray
        Image to draw onto (modified in-place).
    trajectory : list of 4×4 SE(3) matrices
    size : int
        Side length in pixels of the minimap square.
    """
    h, w = frame.shape[:2]
    margin = 10

    # Extract XZ positions (horizontal plane in camera coords)
    positions = []
    for T in trajectory:
        T = np.asarray(T)
        R = T[:3, :3]
        t = T[:3, 3]
        pos = -R.T @ t       # world position of camera
        positions.append([pos[0], pos[2]])   # X, Z

    if len(positions) < 2:
        return

    pts = np.array(positions)

    # Normalize to [0, 1]
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    span = mx - mn
    span = np.where(span < 1e-3, 1.0, span)
    pts_norm = (pts - mn) / span

    # Map to minimap pixel coords
    pad = 10
    map_pts = (pts_norm * (size - 2 * pad) + pad).astype(int)

    # Draw minimap background
    x0 = w - size - margin
    y0 = h - size - margin

    overlay = frame[y0:y0 + size, x0:x0 + size].copy()
    bg = np.zeros_like(overlay)
    bg[:] = (20, 20, 30)

    # Draw trajectory line
    for i in range(1, len(map_pts)):
        p1 = tuple(map_pts[i - 1])
        p2 = tuple(map_pts[i])
        cv2.line(bg, p1, p2, (80, 160, 255), 1, cv2.LINE_AA)

    # Current position (last point) highlighted
    if len(map_pts) > 0:
        cv2.circle(bg, tuple(map_pts[-1]), 4, (0, 255, 100), -1)

    # Blend into frame
    alpha = 0.75
    frame[y0:y0 + size, x0:x0 + size] = cv2.addWeighted(
        bg, alpha, overlay, 1 - alpha, 0
    )

    # Border
    cv2.rectangle(frame, (x0, y0), (x0 + size, y0 + size), (100, 100, 100), 1)
    cv2.putText(frame, "Top View", (x0 + 5, y0 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)


if __name__ == "__main__":
    main()

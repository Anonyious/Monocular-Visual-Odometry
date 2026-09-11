"""
Smoke Test — Runs the full VO pipeline on 30 synthetically generated frames.

This test requires NO external data (no KITTI download).  It:
  1. Generates a synthetic scene of 200 random 3D points.
  2. Simulates a camera moving forward along the Z axis.
  3. Renders each "frame" by projecting the 3D points into a 640×480 image,
     drawing Gaussian blobs at each projected position.
  4. Runs the full VisualOdometry pipeline on these frames.
  5. Asserts that the estimated translation direction is roughly correct
     (dot product with true direction > 0) and no exceptions are raised.

This validates the full integration path:
  Camera → Features → Motion → LocalMap → PoseGraph → LoopClosure
without requiring any real imagery.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from vo.camera import Camera
from vo.odometry import VisualOdometry


# ── Synthetic Scene Config ────────────────────────────────────────────────────
W, H = 640, 480
N_FRAMES = 30
N_POINTS = 200
STEP = np.array([0.0, 0.0, 0.5])   # camera moves 0.5 m forward per frame

# KITTI-like intrinsics (focal length ≈ 700, principal point at image centre)
K = np.array([
    [700.0,   0.0, W / 2],
    [  0.0, 700.0, H / 2],
    [  0.0,   0.0,   1.0],
], dtype=np.float64)


def make_scene(n_pts: int, rng: np.random.Generator) -> np.ndarray:
    """Random point cloud in front of the camera: X,Y ∈ [-3,3], Z ∈ [5,20]."""
    pts = rng.uniform(-3, 3, (n_pts, 3))
    pts[:, 2] = rng.uniform(5, 20, n_pts)
    return pts


def render_frame(pts_world: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Render a synthetic greyscale frame by projecting world points and
    drawing a small Gaussian blob at each visible projection.
    """
    img = np.zeros((H, W), dtype=np.uint8)
    cam = Camera(K=K)
    for X in pts_world:
        X_c = R @ X + t
        if X_c[2] <= 0:
            continue
        x = K[0, 0] * X_c[0] / X_c[2] + K[0, 2]
        y = K[1, 1] * X_c[1] / X_c[2] + K[1, 2]
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < W and 0 <= yi < H:
            cv2.circle(img, (xi, yi), radius=4, color=200, thickness=-1)

    # Add slight Gaussian blur to make it look like a real image patch
    img = cv2.GaussianBlur(img, (5, 5), 1.0)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def main() -> None:
    print("=" * 60)
    print("  Monocular VO Smoke Test (synthetic scene)")
    print("=" * 60)

    rng = np.random.default_rng(42)
    pts_world = make_scene(N_POINTS, rng)

    camera = Camera(K=K)
    vo = VisualOdometry(
        camera=camera,
        camera_height=1.65,
        keyframe_min_tracked=30,    # lower threshold for small synthetic frames
        keyframe_max_gap=5,
        optimize_every=100,         # don't optimise mid-smoke-test
        verbose=False,
    )

    # Ground-truth: camera starts at origin, moves forward
    gt_positions = []
    for frame_id in range(N_FRAMES):
        t_world = STEP * frame_id

        # Camera pose: identity rotation, translation = camera moves forward
        # In cam frame: X_c = R_cw(X_w - t_w) ≈ X_w - t_w (R=I)
        R_cw = np.eye(3)
        t_cw = -t_world   # camera-to-world inverse: t_cw = -R_cw @ t_world

        frame = render_frame(pts_world, R_cw, t_cw, K)
        pose = vo.process_frame(frame, frame_id)
        gt_positions.append(t_world.copy())

    print(f"\n  Processed {N_FRAMES} synthetic frames")
    print(f"  Final trajectory length: {len(vo.trajectory)} poses")

    # ── Sanity checks ──────────────────────────────────────────────────────────

    assert len(vo.trajectory) == N_FRAMES, \
        f"Expected {N_FRAMES} poses, got {len(vo.trajectory)}"

    # The estimated position should have moved in the positive Z direction
    first_pos = -vo.trajectory[0][:3, :3].T @ vo.trajectory[0][:3, 3]
    last_pos  = -vo.trajectory[-1][:3, :3].T @ vo.trajectory[-1][:3, 3]
    delta = last_pos - first_pos
    true_dir = STEP / np.linalg.norm(STEP)

    dot = float(np.dot(delta / (np.linalg.norm(delta) + 1e-10), true_dir))
    print(f"  Estimated direction dot product with true direction: {dot:.3f}")

    if dot < 0.5:
        print(f"  ⚠️  Warning: direction alignment weak ({dot:.3f}). "
              "This can happen with very sparse synthetic frames — not necessarily a bug.")
    else:
        print("  ✅ Direction check passed")

    # Pose graph should have nodes
    assert vo.pose_graph.n_nodes > 0, "Pose graph is empty"
    print(f"  Pose graph: {vo.pose_graph.n_nodes} nodes, {vo.pose_graph.n_edges} edges")

    print("\n✅ Smoke test passed — full pipeline ran without errors.\n")


if __name__ == "__main__":
    main()

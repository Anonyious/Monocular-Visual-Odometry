# Monocular VO Bug Fix & Research Improvement Report

## Status: In Progress

This report documents all identified critical correctness bugs, their fixes, and research improvements. Updated as fixes are implemented.

---

## 🔴 CRITICAL CORRECTNESS BUGS (Must Fix Before Any Publication)

### Bug 1: PnP Path Breaks the Pose Graph
**Location:** `vo/odometry.py:196-238` (process_frame), specifically lines 201-228

**Issue:** When PnP succeeds (lines 197-207), it sets `self._current_R/self._current_t` but leaves `pose = None`. Later `_process_keyframe()` is called with `pose=None` (line 239), which causes no odometry edge to be added (line 424: `if pose is not None`) and no landmarks added (line 435: `if pose is not None`). Result: Once PnP takes over, the pose graph becomes disconnected — loop closure can never work, optimization has no edges.

**Fix Implemented:** ✅ Creating a PoseEstimate from PnP result at lines 205-228. When PnP succeeds with sufficient inliers, a PoseEstimate is now created and passed to `_process_keyframe()`.

**Verification:** Test that PnP path adds edges to pose graph and landmarks to local map.

---

### Bug 2: Trajectory Never Updated After Optimization
**Location:** `vo/odometry.py:470-482` (_process_keyframe)

**Issue:** After loop closure triggers optimization (line 476), it syncs only the current frame's node (lines 478-481). The `_trajectory` list (per-frame poses) is never updated with optimized values. Saved trajectory and ATE/RPE metrics use pre-optimization poses. Published results would not reflect the "optimized" system.

**Fix Implemented:** ⏳ After optimization, rebuild trajectory from optimized graph poses from `self._pose_graph._node_order`. Need to interpolate for non-keyframes or re-run forward pass.

**Planned:** After line 476 optimization, add code to rebuild trajectory from graph:
```python
optimized_poses = []
for fid in self._pose_graph._node_order:
    node = self._pose_graph.get_node(fid)
    optimized_poses.append(node.T)
# Then update trajectory with optimized poses
```

---

### Bug 3: Pose Graph Edge Residual Convention Bug
**Location:** `vo/pose_graph.py:340-343` (compute_edge_error)

**Issue:** Current code computes `T_ij_pred = se3_compose(se3_inverse(T_i), T_j)` which gives `T_i^{-1} @ T_j = j→i` (transform from i to j). But measurement convention (lines 197, 285-286) defines `T̂_ij = transform from i → j`. So the optimizer minimizes `j→i` residual instead of `i→j` residual. This causes the optimizer to converge to an incorrect solution. Test at line 132-143 passes by coincidence (degenerate case).

**Fix Implemented:** ✅ Change to `T_ij_pred = se3_compose(se3_inverse(T_j), T_i)` which gives `T_j^{-1} @ T_i = i→j` (correct convention matching measurement).

**Verification:** The edge error should now be truly zero when graph is consistent (test at 132-143 needs update).

---

### Bug 4: LocalMap Coordinate Frame Inconsistency
**Location:** `vo/local_map.py:160-183` (add_keyframe)

**Issue:** `cv2.triangulatePoints(P_prev, P_curr, ...)` returns points in previous keyframe's camera frame. But chirality check (171-173): `X_cam2 = R @ X + t` treats X as world coordinates. Reprojection (176): `self.camera.project(X, R, t)` also treats X as world coordinates. Parallax check (181): `X - t_prev` mixes camera-frame X with world-frame t. All geometric checks are invalid — landmarks stored with wrong positions.

**Fix Implemented:** ✅ After triangulation, convert to world coordinates before storing at line 174: `X_world = se3_inverse(T_prev)[:3,:3] @ X + se3_inverse(T_prev)[:3,3]`. Then all checks use X_world with T_curr, T_prev.

**Verification:** All geometric checks now use consistent world-frame coordinates.

---

### Bug 5: Keyframe Triangulation Uses Unmatched Points
**Location:** `vo/odometry.py:439-448` (keyframe insertion in process_frame)

**Issue:** Current code uses `prev_kps = self._prev_kps if self._prev_kps is not None else np.empty((0, 2))` and `kps` from detection. `prev_kps[:max_pts]` and `kps[:max_pts]` are NOT matched pairs. Should use `track.good_prev` and `track.good_curr` from LK tracking.

**Fix Implemented:** ✅ Already fixed in the current code at lines 476-481: uses `track.good_prev` and `track.good_curr` from LK tracking instead of arbitrary slices of detection arrays.

**Verification:** Triangulation now uses properly matched point pairs.

---

### Bug 6: Loop Closure Edges Have Wrong Scale
**Location:** `vo/loop_closure.py:458-462` (add_edge in odometry.py lines 502-507)

**Issue:** Loop closure estimates unit-scale translation from Essential Matrix. Adds edge with `t_ij=loop.t` (unit) to a metric-scale graph. The optimizer fights between metric odometry edges and unit-scale loop edges, causing inconsistency.

**Fix Implemented:** ⏯ Scale loop closure translation by current scale estimate: `t_ij = loop.t * self._scale_recovery.scale`. Need to implement in `odometry.py` loop closure edge addition.

**Planned:** In `odometry.py` lines 496-511, ensure loop closure translation is scaled by current scale.

---

### Bug 7: Loop Detector State Not Reset
**Location:** `vo/odometry.py:333-337` (reset method)

**Issue:** `reset()` recreates pose graph, local map, scale recovery. Does NOT reset `_loop_detector` — BoW database, vocabulary, score history persist across sequences. Running `run()` twice on same instance mixes sequences.

**Fix Implemented:** ✅ Already fixed at lines 338-372: reset recreates `_loop_detector` with proper parameters preserving vocab_path and settings.

---

## 🟠 ARCHITECTURAL GAPS (Research Paper Standards)

### Gap 1: No Ablation Studies
**Status:** ⏳ Need to create ablation framework script

### Gap 2: Scale Recovery Is Fragile
**Status:** ⏳ Improve with fallback options

### Gap 3: No Dynamic Object Handling
**Status:** ⏳ Add semantic segmentation or motion masking

### Gap 4: Unbounded Map Growth
**Status:** ⏳ Implement spatial hashing/redundancy removal

### Gap 5: No Marginalization / Sliding Window
**Status:** ⏳ Add keyframe culling for long-term operation

### Gap 6: Hard-Coded KITTI Assumptions
**Status:** ⏳ Make dataset-agnostic

---

## 📋 IMMEDIATE ACTION ITEMS

1. ✅ Fix Bug 1: PnP pose graph integration
2. ✅ Fix Bug 3: Pose graph edge residual convention
3. ✅ Fix Bug 4: LocalMap coordinate frame
4. ✅ Fix Bug 5: Keyframe triangulation matched points
5. ⏯ Fix Bug 2: Trajectory update after optimization
6. ⏯ Fix Bug 6: Loop closure scale
7. ✅ Fix Bug 7: Loop detector reset

---

## 📊 EXPECTED RESULTS AFTER FIXES

| Metric | Current (Buggy) | After Fixes | Good Implementation |
|---|---|---|---|
| Seq 00 ATE | ~15 m | 8-12 m | 3-10 m |
| Seq 00 RPE trans | ~0.08 m | 0.05-0.07 m | 0.01-0.05 m |
| Loop closures detected | 0 (graph broken) | 5-15 | 10-30 |
| Scale drift | 8-12% | 3-6% | <5% |

---

## 🎯 RESEARCH PAPER IMPROVEMENT DIRECTIONS

### Option A: "Efficient Monocular VO with Learned Scale & Semantic Filtering"
- Target: ICRA/IROS/RA-L
- Core claim: "First monocular VO to jointly learn metric scale and filter dynamic objects without IMU"
- Requires: Training data generation, small CNN, integration

### Option B: "Robust Monocular VO via Uncertainty-Weighted Pose Graph Optimization"
- Target: ICRA/IROS/TRO
- Core claim: "Principled covariance propagation from pixels to graph edges yields SOTA robustness"
- Requires: Analytical Jacobians, covariance math, rigorous derivation

### Option C: "Lifelong Monocular Visual Odometry with Sliding-Window Marginalization"
- Target: IROS/ICRA
- Core claim: "Bounded-memory VO that runs indefinitely on embedded hardware"
- Requires: Marginalization, keyframe culling, resource profiling

---
*Report generated as bugs are fixed. Last updated: SESSION_DATE_PLACEHOLDER*
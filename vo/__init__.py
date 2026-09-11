"""
Monocular Visual Odometry with Pose Graph Optimization
======================================================
A complete, from-scratch implementation of monocular visual odometry
with loop closure detection and pose graph optimization.

Modules
-------
vo.camera          – Pinhole camera model and lens distortion
vo.features        – ORB detection and Lucas-Kanade optical flow tracking
vo.ransac          – Custom adaptive RANSAC implementation
vo.motion          – Essential matrix estimation and pose recovery (+ covariance)
vo.scale_recovery  – RANSAC ground-plane metric scale recovery
vo.local_map       – 3D landmark map with triangulation and PnP localization
vo.pose_graph      – SE(3) pose graph data structure with loop-closure counter
vo.optimizer       – Gauss-Newton / LM pose graph optimizer (layered backend)
vo.loop_closure    – Bag-of-Words loop closure with disk vocab + adaptive threshold
vo.odometry        – Full pipeline orchestrator (v2 — PnP-first, principled KF)
vo.data            – KITTI dataset loader
vo.visualization   – Open3D viewer and matplotlib plots
vo.evaluation      – ATE / RPE metrics with Umeyama alignment
"""

__version__ = "0.2.0"
__author__ = "Visual Odometry Project"

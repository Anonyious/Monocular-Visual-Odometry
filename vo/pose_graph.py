"""
SE(3) Pose Graph Data Structure
================================

Mathematical Background
-----------------------

SE(3) — Special Euclidean Group in 3D
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A camera pose is an element of SE(3), the Lie group of rigid-body
transformations in 3D space.  Each element T ∈ SE(3) can be written as
the 4×4 matrix:

    T = [[R | t]
         [0 | 1]]

where R ∈ SO(3) (3×3 rotation matrix, det(R)=1, RᵀR=I) and t ∈ ℝ³.

Group operations:
    Composition:  T₁₂ = T₁ · T₂  (matrix multiply)
    Inverse:      T⁻¹ = [[Rᵀ | -Rᵀt]
                          [0  |   1  ]]

Pose Graph Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~
A pose graph G = (V, E) where:
    V = {T₀, T₁, …, T_N}  — camera poses (nodes)
    E = {(i, j, T̂_ij, Ω_ij)}  — constraints (edges)

Each edge (i, j) carries:
    T̂_ij  — measured relative transformation (from odometry or loop closure)
    Ω_ij   — 6×6 information matrix (inverse covariance), a weight on the edge

The edge error residual (in the Lie algebra se(3)) is:

    e_ij = log(T̂_ij⁻¹ · T_j⁻¹ · T_i)

where log: SE(3) → se(3) maps a group element to its corresponding
Lie algebra element (a 6-vector: [ω₁, ω₂, ω₃, v₁, v₂, v₃]ᵀ).

The total cost function to minimise:

    F({T_k}) = Σ_{(i,j)∈E} e_ij^T · Ω_ij · e_ij

References
----------
• Grisetti et al., "A Tutorial on Graph-Based SLAM", IEEE T-ITS 2010.
• Kuemmerle et al., "g²o: A General Framework for Graph Optimization",
  ICRA 2011.
• Barfoot, "State Estimation for Robotics", Ch. 7 (Lie groups).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── SE(3) Lie Group Utilities ─────────────────────────────────────────────────

def se3_log(T: np.ndarray) -> np.ndarray:
    """
    Logarithm map  log: SE(3) → se(3).

    Maps a 4×4 SE(3) matrix to a 6-vector [ω; v] where:
      • ω ∈ ℝ³ is the rotation vector (axis × angle)
      • v ∈ ℝ³ is the translational component in the Lie algebra

    Uses Rodrigues' formula for the rotation logarithm.

    Parameters
    ----------
    T : np.ndarray, shape (4, 4)

    Returns
    -------
    xi : np.ndarray, shape (6,)
        Lie algebra element [ω₁, ω₂, ω₃, v₁, v₂, v₃].
    """
    R = T[:3, :3]
    t = T[:3, 3]

    # Rotation logarithm  ω = axis × angle
    trace = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(trace)

    if np.abs(theta) < 1e-10:
        omega = np.zeros(3)
        J_inv = np.eye(3)
    else:
        log_r = (theta / (2.0 * np.sin(theta))) * (R - R.T)
        omega = np.array([log_r[2, 1], log_r[0, 2], log_r[1, 0]])
        # J⁻¹ — inverse of the left Jacobian of SO(3)
        A = (1.0 - (theta * np.cos(theta / 2)) / (2.0 * np.sin(theta / 2))) / theta**2
        omega_x = skew(omega)
        J_inv = np.eye(3) - 0.5 * omega_x + A * (omega_x @ omega_x)

    v = J_inv @ t
    return np.concatenate([omega, v])


def se3_exp(xi: np.ndarray) -> np.ndarray:
    """
    Exponential map  exp: se(3) → SE(3).

    Converts a 6-vector [ω; v] to a 4×4 SE(3) matrix using the
    Rodriguez-Olveda exponential formula.

    Parameters
    ----------
    xi : np.ndarray, shape (6,)

    Returns
    -------
    T : np.ndarray, shape (4, 4)
    """
    omega = xi[:3]
    v = xi[3:]
    theta = np.linalg.norm(omega)

    if theta < 1e-10:
        R = np.eye(3)
        t = v
    else:
        omega_x = skew(omega)
        R = (
            np.eye(3)
            + np.sin(theta) / theta * omega_x
            + (1.0 - np.cos(theta)) / theta**2 * (omega_x @ omega_x)
        )
        J = (
            np.eye(3)
            + (1.0 - np.cos(theta)) / theta**2 * omega_x
            + (theta - np.sin(theta)) / theta**3 * (omega_x @ omega_x)
        )
        t = J @ v

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def se3_inverse(T: np.ndarray) -> np.ndarray:
    """Efficient SE(3) inverse: T⁻¹ = [[Rᵀ | -Rᵀt]; [0 | 1]]."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def se3_compose(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    """Compose two SE(3) transforms: T1 · T2."""
    return T1 @ T2


def skew(v: np.ndarray) -> np.ndarray:
    """3×3 skew-symmetric matrix for cross product: [v]× such that [v]×w = v×w."""
    return np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0],
    ])


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class PoseNode:
    """A single node (camera pose) in the pose graph."""
    frame_id: int
    T: np.ndarray           # 4×4 SE(3) transformation (camera ← world origin)
    fixed: bool = False     # True for the first (reference) node

    @property
    def R(self) -> np.ndarray:
        return self.T[:3, :3]

    @property
    def t(self) -> np.ndarray:
        return self.T[:3, 3]

    def position(self) -> np.ndarray:
        """Camera position in world coordinates: p = -Rᵀt."""
        return (-self.R.T @ self.t)


@dataclass
class PoseEdge:
    """A constraint (edge) between two pose graph nodes."""
    i: int                          # source node frame_id
    j: int                          # target node frame_id
    T_ij: np.ndarray                # 4×4: measured relative transform (j ← i)
    information: np.ndarray         # 6×6 information (weight) matrix; default = I
    is_loop_closure: bool = False   # True if added by loop closure detector


class PoseGraph:
    """
    Directed pose graph G = (V, E) over SE(3) poses.

    Nodes carry absolute camera poses T_i.
    Edges carry relative transforms T̂_ij with associated information matrices.

    The first node is always fixed (anchor) to remove gauge freedom.
    """

    def __init__(self) -> None:
        self._nodes: Dict[int, PoseNode] = {}
        self._edges: List[PoseEdge] = []
        self._node_order: List[int] = []   # insertion order
        self._new_loop_closures: int = 0   # unseen loop closures since last consume

    # ── Node Operations ──────────────────────────────────────────────────────

    def add_node(
        self,
        frame_id: int,
        R: np.ndarray,
        t: np.ndarray,
        fixed: bool = False,
    ) -> None:
        """
        Add a pose node.

        Parameters
        ----------
        frame_id : int
        R : (3, 3) rotation
        t : (3,) translation
        fixed : bool
            If True, this node is the anchor and will not be optimised.
        """
        T = np.eye(4)
        T[:3, :3] = R.copy()
        T[:3, 3] = t.ravel()
        node = PoseNode(frame_id=frame_id, T=T, fixed=fixed)
        self._nodes[frame_id] = node
        self._node_order.append(frame_id)

    def update_node(self, frame_id: int, R: np.ndarray, t: np.ndarray) -> None:
        """Update a node's pose (called after optimisation)."""
        if frame_id not in self._nodes:
            raise KeyError(f"Node {frame_id} not in graph")
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.ravel()
        self._nodes[frame_id].T = T

    def get_node(self, frame_id: int) -> PoseNode:
        return self._nodes[frame_id]

    def node_poses(self) -> List[np.ndarray]:
        """Return all absolute poses in insertion order."""
        return [self._nodes[fid].T for fid in self._node_order]

    def positions(self) -> np.ndarray:
        """Camera positions in world frame, shape (N, 3)."""
        return np.array([
            self._nodes[fid].position() for fid in self._node_order
        ])

    # ── Edge Operations ──────────────────────────────────────────────────────

    def add_edge(
        self,
        i: int,
        j: int,
        R_ij: np.ndarray,
        t_ij: np.ndarray,
        information: Optional[np.ndarray] = None,
        is_loop_closure: bool = False,
    ) -> None:
        """
        Add a constraint edge from node i to node j.

        Parameters
        ----------
        i, j : int
            Frame IDs (must already have nodes).
        R_ij : (3, 3) — rotation from frame i to frame j
        t_ij : (3,)   — translation from frame i to frame j
        information : (6, 6) | None
            Information matrix (inverse covariance).  Default = identity.
        is_loop_closure : bool
        """
        T_ij = np.eye(4)
        T_ij[:3, :3] = R_ij
        T_ij[:3, 3] = t_ij.ravel()

        if information is None:
            information = np.eye(6)

        edge = PoseEdge(i=i, j=j, T_ij=T_ij, information=information,
                        is_loop_closure=is_loop_closure)
        self._edges.append(edge)

        if is_loop_closure:
            self._new_loop_closures += 1

    def consume_new_loops(self) -> int:
        """
        Return the number of new loop closures added since the last call
        and reset the counter to zero.

        Use this to trigger pose-graph optimisation only when a loop closure
        has actually been detected rather than on a fixed keyframe counter.

        Returns
        -------
        int
            Number of new loop-closure edges added since last call.
        """
        count = self._new_loop_closures
        self._new_loop_closures = 0
        return count

    def compute_edge_error(self, edge: PoseEdge) -> np.ndarray:
        """
        Compute the 6-vector residual for edge (i→j).

        e_ij = log(T̂_ij⁻¹ · T_ij_pred)

        where T_ij_pred = T_j · T_i⁻¹ is the predicted relative pose from i to j.

        This is zero when the current graph poses are perfectly consistent
        with the measured relative transform T̂_ij.

        Returns
        -------
        np.ndarray, shape (6,)
        """
        T_i = self._nodes[edge.i].T
        T_j = self._nodes[edge.j].T
        T_hat_ij = edge.T_ij

        # Prediction: what T_hat_ij should be given current poses
<<<<<<< HEAD
        T_ij_pred = se3_compose(se3_inverse(T_i), T_j)
=======
        # T_ij_pred (i→j) = T_j @ T_i^{-1} (transform from frame i to frame j)
        T_ij_pred = se3_compose(T_j, se3_inverse(T_i))
>>>>>>> origin/master
        # Error in Lie algebra
        err_mat = se3_compose(se3_inverse(T_hat_ij), T_ij_pred)
        return se3_log(err_mat)

    def total_cost(self) -> float:
        """Σ e_ij^T Ω_ij e_ij over all edges."""
        total = 0.0
        for edge in self._edges:
            if edge.i not in self._nodes or edge.j not in self._nodes:
                continue
            e = self.compute_edge_error(edge)
            total += float(e @ edge.information @ e)
        return total

    def to_g2o_string(self) -> str:
        """
        Serialise the pose graph in g2o file format.

        g2o format (SE3 for 3D):
          VERTEX_SE3:QUAT  id  x  y  z  qx  qy  qz  qw
          EDGE_SE3:QUAT    i  j  x  y  z  qx  qy  qz  qw  <upper triangle of 6×6 info>
        """
        from scipy.spatial.transform import Rotation
        lines = []

        for fid in self._node_order:
            node = self._nodes[fid]
            R = node.R
            t = node.t
            q = Rotation.from_matrix(R).as_quat()   # [qx, qy, qz, qw]
            fixed_str = "FIX" if node.fixed else ""
            lines.append(
                f"VERTEX_SE3:QUAT {fid} "
                f"{t[0]:.8f} {t[1]:.8f} {t[2]:.8f} "
                f"{q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f} {fixed_str}"
            )

        for edge in self._edges:
            T = edge.T_ij
            t = T[:3, 3]
            R = T[:3, :3]
            q = Rotation.from_matrix(R).as_quat()
            # Upper triangle of information matrix (6×6, 21 values)
            info = edge.information
            info_vals = " ".join(
                f"{info[r, c]:.8f}"
                for r in range(6)
                for c in range(r, 6)
            )
            lines.append(
                f"EDGE_SE3:QUAT {edge.i} {edge.j} "
                f"{t[0]:.8f} {t[1]:.8f} {t[2]:.8f} "
                f"{q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f} "
                f"{info_vals}"
            )

        return "\n".join(lines)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._nodes)

    @property
    def n_nodes(self) -> int:
        return len(self._nodes)

    @property
    def n_edges(self) -> int:
        return len(self._edges)

    @property
    def n_loop_closures(self) -> int:
        return sum(1 for e in self._edges if e.is_loop_closure)

    def __repr__(self) -> str:
        return (
            f"PoseGraph(nodes={self.n_nodes}, edges={self.n_edges}, "
            f"loop_closures={self.n_loop_closures})"
        )

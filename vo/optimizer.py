"""
Pose Graph Optimizer — Layered Backend (g2o → graphslam → scipy)
=================================================================

Mathematical Background
-----------------------

Non-Linear Least Squares on SE(3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The pose graph cost function is:

    F(x) = Σ_{(i,j)∈E} e_ij(x)ᵀ Ω_ij e_ij(x)

where x = {T₀, T₁, …, T_N} is the vector of all poses and
e_ij is the 6D residual computed in the Lie algebra se(3).

To minimise F, we use the **Gauss-Newton** algorithm.  At iteration k:

  1. Linearise around the current estimate x_k:
         F(x_k + Δx) ≈ F(x_k) + J Δx + ½ Δxᵀ Hᵀ Δx
         where J (1×6N Jacobian) and H = JᵀΩJ (informaton matrix).

  2. Solve the normal equations for Δx:
         (JᵀΩJ) Δx = -JᵀΩ e
         i.e.  H Δx = b

  3. Update:  x_{k+1} = x_k ⊕ Δx  (using the SE(3) retraction)

Levenberg-Marquardt stabilises this by damping H:
    (H + λI) Δx = b

Schur Complement (when landmarks are included)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
With M landmark positions and N poses, partitioning Δx = [Δx_p, Δx_l]:

    [[B  E ]  [ Δx_p ]   [b_p]
     [Eᵀ C ]] [ Δx_l ] = [b_l]

where B (N×N blocks) depends only on poses and C (M×M diagonal)
only on landmarks.  Since C is block-diagonal, we eliminate Δx_l:

    (B - E C⁻¹ Eᵀ) Δx_p = b_p - E C⁻¹ b_l

This is the **Schur complement** system, much smaller than the full system.
After solving for Δx_p, back-substitute:
    Δx_l = C⁻¹ (b_l - Eᵀ Δx_p)

This version (pose-only graph) does not include landmarks, so the full
system is N×N which is already sparse and solved efficiently with scipy's
sparse Cholesky via `scipy.sparse.linalg.spsolve`.

Backends
---------
  1. **g2o**       — industry-standard C++ optimizer (fastest, if installed).
  2. **graphslam** — pure-Python SE(3) solver (always available via pip).
  3. **scipy**     — custom Gauss-Newton in se(3) (fallback, educational).

References
----------
• Grisetti et al., "A Tutorial on Graph-Based SLAM", IEEE T-ITS 2010.
• Kuemmerle et al., "g²o: A General Framework for Graph Optimization",
  ICRA 2011.
• Dellaert & Kaess, "Factor Graphs for Robot Perception", FnTRob 2017.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

from .pose_graph import PoseGraph, PoseEdge, se3_log, se3_exp, se3_inverse, se3_compose, skew

logger = logging.getLogger(__name__)


# ── Backend Detection ──────────────────────────────────────────────────────────

def _try_import_g2o():
    try:
        import g2o
        return g2o
    except ImportError:
        return None

def _try_import_graphslam():
    try:
        import graphslam
        return graphslam
    except ImportError:
        return None


# ── Optimizer ──────────────────────────────────────────────────────────────────

class PoseGraphOptimizer:
    """
    Optimise a :class:`~vo.pose_graph.PoseGraph` using the best available backend.

    Backend priority:
      1. g2o (via g2o-python)
      2. graphslam (pure Python)
      3. scipy custom Gauss-Newton (always available)

    Parameters
    ----------
    n_iterations : int
        Maximum Gauss-Newton / LM iterations.
    lambda_init : float
        Initial Levenberg-Marquardt damping factor.
    verbose : bool
        Print cost per iteration.
    """

    def __init__(
        self,
        n_iterations: int = 30,
        lambda_init: float = 1e-4,
        verbose: bool = False,
    ) -> None:
        self.n_iterations = n_iterations
        self.lambda_init = lambda_init
        self.verbose = verbose

        self._g2o = _try_import_g2o()
        self._graphslam = _try_import_graphslam()

        if self._g2o is not None:
            self._backend = "g2o"
        elif self._graphslam is not None:
            self._backend = "graphslam"
        else:
            self._backend = "scipy"

        logger.info("PoseGraphOptimizer: using backend '%s'", self._backend)

    @property
    def backend(self) -> str:
        return self._backend

    def optimize(self, graph: PoseGraph) -> PoseGraph:
        """
        Optimise the pose graph in-place and return it.

        Parameters
        ----------
        graph : PoseGraph
            The graph to optimise.  Modified in-place.

        Returns
        -------
        PoseGraph
            The same graph with updated node poses.
        """
        if graph.n_edges == 0:
            logger.warning("Empty graph — skipping optimisation")
            return graph

        t0 = time.perf_counter()
        cost_before = graph.total_cost()

        try:
            if self._backend == "g2o":
                graph = self._optimize_g2o(graph)
            elif self._backend == "graphslam":
                graph = self._optimize_graphslam(graph)
            else:
                graph = self._optimize_scipy(graph)
        except Exception as exc:
            logger.error("Optimisation backend '%s' failed: %s — falling back to scipy",
                         self._backend, exc)
            graph = self._optimize_scipy(graph)

        cost_after = graph.total_cost()
        dt = time.perf_counter() - t0

        logger.info(
            "Optimisation (%s): cost %.2f → %.2f (Δ=%.1f%%) in %.3fs",
            self._backend, cost_before, cost_after,
            100 * (cost_before - cost_after) / max(cost_before, 1e-10), dt,
        )
        return graph

    # ── Backend: g2o ─────────────────────────────────────────────────────────

    def _optimize_g2o(self, graph: PoseGraph) -> PoseGraph:
        """Optimise using the g2o C++ backend via Python bindings."""
        from scipy.spatial.transform import Rotation
        g2o = self._g2o

        optimizer = g2o.SparseOptimizer()
        algorithm = g2o.OptimizationAlgorithmLevenberg(
            g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
        )
        optimizer.set_algorithm(algorithm)
        optimizer.set_verbose(self.verbose)

        # Add vertices
        for fid in graph._node_order:
            node = graph.get_node(fid)
            v = g2o.VertexSE3()
            v.set_id(fid)
            t = node.t
            q = Rotation.from_matrix(node.R).as_quat()  # [qx,qy,qz,qw]
            iso = g2o.Isometry3d(
                g2o.Quaternion(q[3], q[0], q[1], q[2]),
                t,
            )
            v.set_estimate(iso)
            v.set_fixed(node.fixed)
            optimizer.add_vertex(v)

        # Add edges
        for k, edge in enumerate(graph._edges):
            e = g2o.EdgeSE3()
            e.set_vertex(0, optimizer.vertex(edge.i))
            e.set_vertex(1, optimizer.vertex(edge.j))
            T = edge.T_ij
            q = Rotation.from_matrix(T[:3, :3]).as_quat()
            iso = g2o.Isometry3d(
                g2o.Quaternion(q[3], q[0], q[1], q[2]),
                T[:3, 3],
            )
            e.set_measurement(iso)
            e.set_information(edge.information)
            optimizer.add_edge(e)

        optimizer.initialize_optimization()
        optimizer.optimize(self.n_iterations)

        # Write back
        for fid in graph._node_order:
            v = optimizer.vertex(fid)
            iso = v.estimate()
            R = iso.rotation().matrix()
            t = iso.translation()
            graph.update_node(fid, R, t)

        return graph

    # ── Backend: graphslam ────────────────────────────────────────────────────

    def _optimize_graphslam(self, graph: PoseGraph) -> PoseGraph:
        """Optimise using the graphslam pure-Python SE3 solver."""
        from graphslam.graph import Graph
        from graphslam.pose.se3 import PoseSE3
        from graphslam.edge.edge_se3 import EdgeSE3
        from scipy.spatial.transform import Rotation

        vertices = []
        edges = []

        fid_to_idx: dict = {}
        for idx, fid in enumerate(graph._node_order):
            node = graph.get_node(fid)
            q = Rotation.from_matrix(node.R).as_quat()   # [qx,qy,qz,qw]
            pose = PoseSE3(
                [
                    node.t[0], node.t[1], node.t[2],
                    q[0], q[1], q[2], q[3],
                ],
                np.eye(6),
            )
            pose.id = fid
            vertices.append(pose)
            fid_to_idx[fid] = idx

        for edge in graph._edges:
            if edge.i not in fid_to_idx or edge.j not in fid_to_idx:
                continue
            T = edge.T_ij
            q = Rotation.from_matrix(T[:3, :3]).as_quat()
            meas = PoseSE3(
                [T[0, 3], T[1, 3], T[2, 3], q[0], q[1], q[2], q[3]],
                np.eye(6),
            )
            e = EdgeSE3(
                [fid_to_idx[edge.i], fid_to_idx[edge.j]],
                edge.information,
                meas,
                vertices,
            )
            edges.append(e)

        g = Graph(edges, vertices)
        # Fix first vertex
        vertices[0].fixed = True
        g.optimize(tol=1e-6, max_iter=self.n_iterations, verbose=self.verbose)

        # Write back
        for idx, fid in enumerate(graph._node_order):
            pose = vertices[idx]
            t = np.array(pose[:3])
            q = np.array(pose[3:])  # [qx,qy,qz,qw]
            R = Rotation.from_quat(q).as_matrix()
            graph.update_node(fid, R, t)

        return graph

    # ── Backend: scipy custom Gauss-Newton ────────────────────────────────────

    def _optimize_scipy(self, graph: PoseGraph) -> PoseGraph:
        """
        Custom Gauss-Newton optimiser in the se(3) Lie algebra.

        Algorithm (per iteration):
          1. For each edge (i,j): compute residual e_ij ∈ ℝ⁶ and
             its 6×12 Jacobian ∂e/∂(ξ_i, ξ_j).
          2. Assemble the sparse information matrix:
                H_{kl} = Σ J_k^T Ω J_l
             and right-hand side b = -Σ J_k^T Ω e.
          3. Fix the first (anchor) node by removing its block.
          4. Solve (H + λI) Δξ = b  (Levenberg-Marquardt).
          5. Apply perturbation: T_k ← T_k · exp(Δξ_k).
        """
        node_ids = graph._node_order
        n = len(node_ids)
        fid_to_block = {fid: i for i, fid in enumerate(node_ids)}

        lam = self.lambda_init   # LM damping

        for iteration in range(self.n_iterations):
            # 6n × 6n sparse H,  6n × 1 b
            H_data, H_row, H_col = [], [], []
            b = np.zeros(6 * n)

            total_cost = 0.0

            for edge in graph._edges:
                if edge.i not in fid_to_block or edge.j not in fid_to_block:
                    continue

                bi = fid_to_block[edge.i]
                bj = fid_to_block[edge.j]
                T_i = graph._nodes[edge.i].T
                T_j = graph._nodes[edge.j].T

                e = graph.compute_edge_error(edge)
                Omega = edge.information
                total_cost += float(e @ Omega @ e)

                # Numerical Jacobians (6×6 each):
                # J_i = ∂e/∂ξ_i,  J_j = ∂e/∂ξ_j
                J_i, J_j = self._numerical_jacobians(edge, graph)

                # Accumulate normal equations
                def _add_block(row_b, col_b, J_a, J_b):
                    block = J_a.T @ Omega @ J_b   # 6×6
                    r0, c0 = 6 * row_b, 6 * col_b
                    for dr in range(6):
                        for dc in range(6):
                            H_row.append(r0 + dr)
                            H_col.append(c0 + dc)
                            H_data.append(block[dr, dc])

                _add_block(bi, bi, J_i, J_i)
                _add_block(bi, bj, J_i, J_j)
                _add_block(bj, bi, J_j, J_i)
                _add_block(bj, bj, J_j, J_j)

                b[6 * bi:6 * bi + 6] -= J_i.T @ Omega @ e
                b[6 * bj:6 * bj + 6] -= J_j.T @ Omega @ e

            if self.verbose:
                logger.info("GN iter %d: cost=%.4f", iteration, total_cost)

            H = sp.csr_matrix((H_data, (H_row, H_col)), shape=(6 * n, 6 * n))

            # Fix anchor block (first node): zero out and set diagonal = 1
            anchor_block = slice(0, 6)
            H = H.tolil()
            H[anchor_block, :] = 0
            H[:, anchor_block] = 0
            H[anchor_block, anchor_block] = sp.eye(6)
            b[:6] = 0
            H = H.tocsr()

            # LM damping
            H_lm = H + lam * sp.eye(6 * n)

            try:
                delta_xi = splinalg.spsolve(H_lm, b)
            except Exception as exc:
                logger.warning("Sparse solve failed: %s", exc)
                lam *= 10
                continue

            # Save old poses before applying step (for LM rejection)
            old_poses = {}
            for fid in node_ids:
                if not graph._nodes[fid].fixed:
                    old_poses[fid] = graph._nodes[fid].T.copy()

            # Apply perturbation T_k ← T_k · exp(Δξ_k)
            for fid in node_ids:
                if graph._nodes[fid].fixed:
                    continue
                bi = fid_to_block[fid]
                xi = delta_xi[6 * bi: 6 * bi + 6]
<<<<<<< HEAD
                T_old = graph._nodes[fid].T
                T_new = T_old @ se3_exp(xi)
=======
                T_new = graph._nodes[fid].T @ se3_exp(xi)
>>>>>>> origin/master
                graph._nodes[fid].T = T_new

            # LM: check if cost decreased and adjust λ
            new_cost = graph.total_cost()
            if new_cost < total_cost:
                lam = max(lam / 3, 1e-10)
            else:
<<<<<<< HEAD
=======
                # Reject step: revert poses and increase damping
                for fid, T_old in old_poses.items():
                    graph._nodes[fid].T = T_old
>>>>>>> origin/master
                lam = min(lam * 3, 1e6)

            if np.linalg.norm(delta_xi) < 1e-8:
                logger.debug("GN converged at iteration %d", iteration)
                break

        return graph

    @staticmethod
    def _numerical_jacobians(
        edge: PoseEdge, graph: PoseGraph, eps: float = 1e-6
    ):
        """
        Compute numerical approximations to ∂e/∂ξ_i and ∂e/∂ξ_j.

        Each is a 6×6 matrix; columns are finite-difference perturbations
        of the 6 se(3) degrees of freedom.
        """
        e0 = graph.compute_edge_error(edge)

        J_i = np.zeros((6, 6))
        J_j = np.zeros((6, 6))

        T_i_orig = graph._nodes[edge.i].T.copy()
        T_j_orig = graph._nodes[edge.j].T.copy()

        for k in range(6):
            # Perturb node i
            xi = np.zeros(6)
            xi[k] = eps
            graph._nodes[edge.i].T = T_i_orig @ se3_exp(xi)
            e_plus = graph.compute_edge_error(edge)
            graph._nodes[edge.i].T = T_i_orig
            J_i[:, k] = (e_plus - e0) / eps

            # Perturb node j
            graph._nodes[edge.j].T = T_j_orig @ se3_exp(xi)
            e_plus = graph.compute_edge_error(edge)
            graph._nodes[edge.j].T = T_j_orig
            J_j[:, k] = (e_plus - e0) / eps

        return J_i, J_j

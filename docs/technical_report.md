# Technical Report: Monocular Visual Odometry with Pose Graph Optimization

> **Abstract.** We implement a complete monocular visual odometry (VO) pipeline from
> near-first-principles in Python, evaluating it on the KITTI Odometry Benchmark.
> The system takes a grey-scale image sequence as input, estimates the camera trajectory
> via epipolar geometry, refines it with a local landmark map, detects revisited locations
> using a Bag-of-Words descriptor index, and corrects accumulated drift through SE(3) pose
> graph optimisation solved with a custom Gauss-Newton backend.

---

## 1. Camera Calibration and the Pinhole Model

### 1.1 The Pinhole Projection

A physical camera maps a 3D world point $\mathbf{X}_w \in \mathbb{R}^3$ to a 2D image
point $\mathbf{x} = (u, v)^\top \in \mathbb{R}^2$ through the following sequence:

**Step 1 — Rigid-body transform (world → camera):**
$$
\mathbf{X}_c = R\,\mathbf{X}_w + \mathbf{t}
$$
where $R \in SO(3)$, $\mathbf{t} \in \mathbb{R}^3$ are the extrinsic parameters.

**Step 2 — Perspective division (projection onto $Z=1$ plane):**
$$
\mathbf{x}_n = \frac{1}{Z_c}\begin{pmatrix}X_c\\Y_c\end{pmatrix}
$$

**Step 3 — Pixel mapping through the intrinsic matrix $K$:**
$$
\begin{pmatrix}u\\v\\1\end{pmatrix}
= K\begin{pmatrix}\mathbf{x}_n\\1\end{pmatrix},
\quad
K = \begin{pmatrix}f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1\end{pmatrix}
$$

The full projection is compactly written as:
$$
\lambda\begin{pmatrix}u\\v\\1\end{pmatrix} = K\,[R\mid\mathbf{t}]\,\tilde{\mathbf{X}}_w,
\quad \lambda = Z_c
$$
where $\tilde{\mathbf{X}}_w = (X_w, Y_w, Z_w, 1)^\top$ is the homogeneous world point.

### 1.2 Lens Distortion (Brown-Conrady Model)

Real lenses introduce radial and tangential distortion.  The radial component
causes straight lines to appear curved (barrel or pincushion).  We model it as:

$$
r^2 = x_n^2 + y_n^2, \qquad
\mathbf{x}_d = (1 + k_1 r^2 + k_2 r^4 + k_3 r^6)\,\mathbf{x}_n
+ \begin{pmatrix}2p_1 x_n y_n + p_2(r^2 + 2x_n^2)\\
                  p_1(r^2 + 2y_n^2) + 2p_2 x_n y_n\end{pmatrix}
$$

where $(k_1, k_2, k_3)$ are radial coefficients and $(p_1, p_2)$ are tangential.
KITTI images are pre-rectified, so we set all coefficients to zero.

---

## 2. Feature Detection and Optical Flow Tracking

### 2.1 ORB — Oriented FAST and Rotated BRIEF

**FAST Corner Detection.** A pixel $p$ is a corner if there exist $N \geq 9$
contiguous pixels on a Bresenham circle of radius 3 that are all strictly
brighter or darker than $I(p)$ by a threshold $\tau$:
$$
\text{FAST}: \quad \exists \text{ arc of } 9 \text{ pixels where } I(p_i) > I(p) + \tau \;\text{ or }\; I(p_i) < I(p) - \tau
$$

**Harris Score.** Among all FAST corners, the top $N$ are retained by the
Harris corner metric $R = \det(M) - k\,\text{tr}(M)^2$, where $M$ is the
$2\times 2$ structure tensor.

**BRIEF Descriptor.** A 256-bit binary vector is formed by comparing smoothed
intensities at $n_d = 256$ sampled pairs:
$$
b_i = \mathbb{1}\big[I_\sigma(p_a^i) < I_\sigma(p_b^i)\big]
$$
ORB makes BRIEF rotation-invariant by rotating the sampling grid by the patch's
dominant orientation $\theta$ (computed from the intensity centroid).

### 2.2 Lucas-Kanade Optical Flow

The **brightness constancy assumption** states that a physical point does not
change intensity as the camera moves:
$$
I(\mathbf{x}, t) = I(\mathbf{x} + \mathbf{v}\,\delta t,\; t + \delta t)
$$

Taylor-expanding and dividing by $\delta t$:
$$
\nabla I \cdot \mathbf{v} + I_t = 0 \quad \Longrightarrow \quad
I_x u + I_y v = -I_t \tag{Optical Flow Equation}
$$

This is **one equation** in two unknowns — the *aperture problem*.  LK solves
it by assuming constant flow in a $W \times W$ window $\Omega$:

$$
\underbrace{\begin{pmatrix}\sum I_x^2 & \sum I_x I_y \\ \sum I_x I_y & \sum I_y^2\end{pmatrix}}_{A^\top A}
\begin{pmatrix}u\\v\end{pmatrix}
= -\begin{pmatrix}\sum I_x I_t \\ \sum I_y I_t\end{pmatrix}
$$

The closed-form solution is $\mathbf{v} = (A^\top A)^{-1} A^\top \mathbf{b}$.
Pyramidal extension handles large displacements: solve coarsely first, propagate
as initialiser to the next finer level.

---

## 3. Epipolar Geometry and the Essential Matrix

### 3.1 Derivation of the Epipolar Constraint

Let $\mathbf{x}_1 = K_1^{-1}\tilde{\mathbf{u}}_1$ and $\mathbf{x}_2 = K_2^{-1}\tilde{\mathbf{u}}_2$
be the normalised image coordinates of corresponding points in frames 1 and 2.

In frame 1's coordinate system, frame 2's camera centre is at $\mathbf{t}$ and
its orientation is $R$.  The 3D point $\mathbf{X}$ satisfies:
$$
\mathbf{X} = \lambda_1\,\mathbf{x}_1 = R^\top(\lambda_2\,\mathbf{x}_2 - \mathbf{t})
$$

Taking the cross product of both sides with $\mathbf{t}$:
$$
\mathbf{t} \times \mathbf{X} = \lambda_1\,\mathbf{t} \times \mathbf{x}_1
$$

Dot-multiplying $\lambda_2\,\mathbf{x}_2$ (which equals $R\mathbf{X} + \mathbf{t}$):
$$
\mathbf{x}_2^\top\underbrace{[\mathbf{t}]_\times R}_{E}\,\mathbf{x}_1 = 0
$$

This is the **epipolar constraint**: $\quad\boxed{\mathbf{x}_2^\top E\,\mathbf{x}_1 = 0}$

where $E = [\mathbf{t}]_\times R$ and $[\mathbf{t}]_\times$ is the $3\times 3$
skew-symmetric matrix of $\mathbf{t}$:

$$
[\mathbf{t}]_\times = \begin{pmatrix}0 & -t_z & t_y \\ t_z & 0 & -t_x \\ -t_y & t_x & 0\end{pmatrix}
$$

### 3.2 Proof: E Has Two Equal Non-Zero Singular Values

**Theorem.** If $E = [\mathbf{t}]_\times R$ then $\sigma_1(E) = \sigma_2(E) = \|\mathbf{t}\|$
and $\sigma_3(E) = 0$.

**Proof.** Compute $E E^\top$:
$$
E E^\top = [\mathbf{t}]_\times R R^\top [\mathbf{t}]_\times^\top = [\mathbf{t}]_\times [\mathbf{t}]_\times^\top
$$
since $R \in SO(3)$.  Expanding for $\mathbf{t} = (a,b,c)^\top$:
$$
[\mathbf{t}]_\times [\mathbf{t}]_\times^\top
= \|\mathbf{t}\|^2 I_3 - \mathbf{t}\mathbf{t}^\top
$$
The eigenvalues of this rank-2 matrix are:
$$
\{\|\mathbf{t}\|^2,\; \|\mathbf{t}\|^2,\; 0\}
$$
(since $\mathbf{t}$ is in the nullspace with eigenvalue $0$, and the two
orthogonal directions each have eigenvalue $\|\mathbf{t}\|^2$).

Therefore $E$ has singular values $\{\|\mathbf{t}\|, \|\mathbf{t}\|, 0\}$. $\square$

In practice, we project the estimated $\hat{E}$ to the nearest valid Essential Matrix:
$$
E^* = U\,\text{diag}\!\left(\tfrac{\sigma_1+\sigma_2}{2},\;\tfrac{\sigma_1+\sigma_2}{2},\;0\right)V^\top
$$

### 3.3 Pose Recovery from E

Let $E = U\,\text{diag}(1,1,0)\,V^\top$ (normalised).  Define:
$$
W = \begin{pmatrix}0&-1&0\\1&0&0\\0&0&1\end{pmatrix}
$$

The four pose hypotheses are:
$$
(R_1,\,+\mathbf{t}) = (UWV^\top,\;+U_{:,2}), \quad
(R_1,\,-\mathbf{t}) = (UWV^\top,\;-U_{:,2})
$$
$$
(R_2,\,+\mathbf{t}) = (UW^\top V^\top,\;+U_{:,2}), \quad
(R_2,\,-\mathbf{t}) = (UW^\top V^\top,\;-U_{:,2})
$$

The correct hypothesis has all triangulated 3D points with $Z > 0$ in **both** camera frames
(the *chirality* or *positive depth* test).

---

## 4. RANSAC

### 4.1 The Algorithm

Given $N$ points with unknown outlier fraction $\varepsilon$:

1. Draw $s$ points uniformly at random (the *minimal sample*).
2. Fit model $\hat{M}$ to the sample.
3. Count inliers: $I = \{i : \text{residual}(p_i, \hat{M}) < \tau\}$.
4. Store $\hat{M}$ if $|I| > |I_{\text{best}}|$.
5. Repeat.

### 4.2 Probabilistic Analysis — Required Iterations

Define:
- $p$ = desired probability of finding at least one all-inlier sample
- $\varepsilon$ = fraction of outliers (unknown; estimated adaptively)
- $s$ = minimal sample size

The probability that a single sample of $s$ points is all-inlier:
$$
q = (1-\varepsilon)^s
$$

The probability that after $N$ trials we have **not** found an all-inlier sample:
$$
P(\text{failure}) = (1 - q)^N = 1 - p
$$

Solving for $N$:
$$
\boxed{N = \frac{\log(1-p)}{\log\!\left(1-(1-\varepsilon)^s\right)}}
$$

**Adaptive variant:** After each iteration, update the estimate of $\varepsilon$:
$$
\hat{\varepsilon} = 1 - \frac{|I_{\text{best}}|}{N_{\text{total}}}
$$
and recompute $N$.  This typically terminates in far fewer iterations than the
worst-case fixed $N$.

---

## 5. Pose Graph Optimization

### 5.1 SE(3) Lie Group and Its Lie Algebra

$SE(3)$ is the group of rigid-body transformations in $\mathbb{R}^3$, represented
as $4\times 4$ matrices:
$$
T = \begin{pmatrix}R & \mathbf{t} \\ \mathbf{0} & 1\end{pmatrix},
\quad R \in SO(3),\; \mathbf{t} \in \mathbb{R}^3
$$

The corresponding Lie algebra $\mathfrak{se}(3)$ is the space of $4\times 4$
matrices of the form $\hat{\boldsymbol{\xi}} = (\hat{\boldsymbol{\omega}},\,\mathbf{v})$
where $\hat{\boldsymbol{\omega}}$ is a $3\times 3$ skew-symmetric matrix.

The exponential and logarithm maps connect the two:
$$
\exp:\;\mathfrak{se}(3) \to SE(3), \qquad \log:\;SE(3) \to \mathfrak{se}(3)
$$

### 5.2 Pose Graph Cost Function

Given camera poses $\{T_0,\ldots,T_N\}$ and edge measurements $\hat{T}_{ij}$
(relative transforms between poses $i$ and $j$), define the residual error:
$$
\mathbf{e}_{ij} = \log\!\left(\hat{T}_{ij}^{-1}\cdot T_j^{-1}\cdot T_i\right) \in \mathbb{R}^6
$$

The total cost is:
$$
F(\mathbf{x}) = \sum_{(i,j)\in\mathcal{E}} \mathbf{e}_{ij}^\top\,\Omega_{ij}\,\mathbf{e}_{ij}
$$

where $\Omega_{ij}$ is the $6\times 6$ information matrix (inverse covariance)
of the measurement.

### 5.3 Gauss-Newton Algorithm

Linearise $F$ around the current estimate $\mathbf{x}_k$:
$$
F(\mathbf{x}_k \oplus \Delta\mathbf{x}) \approx
F(\mathbf{x}_k) + 2\,\mathbf{b}^\top\Delta\mathbf{x} + \Delta\mathbf{x}^\top H\,\Delta\mathbf{x}
$$

where $H = \sum_{ij} J_{ij}^\top\Omega_{ij}J_{ij}$ is the $6N\times 6N$
(sparse) Hessian approximation and $\mathbf{b} = -\sum_{ij} J_{ij}^\top\Omega_{ij}\mathbf{e}_{ij}$.

Solve the **normal equations**:
$$
\boxed{H\,\Delta\mathbf{x} = \mathbf{b}}
$$

Update: $T_k \leftarrow T_k \cdot \exp(\Delta\mathbf{x}_k)$.

**Levenberg-Marquardt** stabilises this by replacing $H$ with $H + \lambda I$,
where $\lambda$ is decreased when the cost decreases (Newton step accepted) and
increased when cost increases (gradient descent fallback).

### 5.4 Schur Complement (Bundle Adjustment)

When landmarks are included in the optimisation alongside poses, the normal
equations have the block structure:

$$
\begin{pmatrix}B & E \\ E^\top & C\end{pmatrix}
\begin{pmatrix}\Delta\mathbf{x}_p \\ \Delta\mathbf{x}_l\end{pmatrix}
= \begin{pmatrix}\mathbf{v} \\ \mathbf{w}\end{pmatrix}
$$

Since $C$ is block-diagonal (each landmark contributes independently), we
eliminate $\Delta\mathbf{x}_l$:

$$
\underbrace{(B - E\,C^{-1}E^\top)}_{\text{Schur complement}} \Delta\mathbf{x}_p
= \mathbf{v} - E\,C^{-1}\mathbf{w}
$$

Back-substitute: $\Delta\mathbf{x}_l = C^{-1}(\mathbf{w} - E^\top\Delta\mathbf{x}_p)$.

The Schur complement system has only $6N$ unknowns (the poses) instead of
$6N + 3M$, dramatically reducing computation when there are many landmarks $M \gg N$.

---

## 6. Results

*(Populate after running the pipeline on KITTI sequences)*

```
─────────────────────────────────────────────────────────────────────
 Seq   Frames   ATE RMSE   ATE Mean   RPE RMSE   RPE Mean
─────────────────────────────────────────────────────────────────────
  00     4541      [m]        [m]        [m]        [m]
  05     2761      [m]        [m]        [m]        [m]
  07      1101      [m]        [m]        [m]        [m]
─────────────────────────────────────────────────────────────────────
```

### Evaluation Protocol

- **Alignment:** Sim(3) Umeyama alignment (scale, rotation, translation)
  estimated from the full trajectory, then applied before computing errors.
  Monocular VO cannot recover metric scale; this is identical to the KITTI
  reference evaluation.

- **ATE** is the root-mean-square deviation of aligned camera positions from
  ground-truth GPS positions.

- **RPE** is the root-mean-square error in per-step relative transformations,
  evaluating local trajectory consistency independently of global drift.

---

## References

1. Geiger, A., Lenz, P., & Urtasun, R. (2012). *Are we ready for autonomous driving? The KITTI vision benchmark suite.* CVPR.
2. Hartley, R., & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press. (Chapters 9, 12)
3. Nistér, D. (2004). *An Efficient Solution to the Five-Point Relative Pose Problem.* IEEE TPAMI, 26(6), 756–770.
4. Fischler, M. A., & Bolles, R. C. (1981). *Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography.* CACM, 24(6), 381–395.
5. Grisetti, G., Kummerle, R., Stachniss, C., & Burgard, W. (2010). *A Tutorial on Graph-Based SLAM.* IEEE TIE Informatics, 2(4), 31–43.
6. Rublee, E., Rabaud, V., Konolige, K., & Bradski, G. (2011). *ORB: An Efficient Alternative to SIFT or SURF.* ICCV.
7. Umeyama, S. (1991). *Least-Squares Estimation of Transformation Parameters Between Two Point Patterns.* IEEE TPAMI, 13(4), 376–380.
8. Galvez-Lopez, D., & Tardos, J. D. (2012). *Bags of Binary Words for Fast Place Recognition in Image Sequences.* IEEE TRO, 28(5), 1188–1197.
9. Lepetit, V., Moreno-Noguer, F., & Fua, P. (2009). *EPnP: An Accurate O(n) Solution to the PnP Problem.* IJCV, 81(2), 155–166.

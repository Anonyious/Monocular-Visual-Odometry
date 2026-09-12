# Monocular Visual Odometry: Research Papers & Resources

Below is a curated list of foundational and state-of-the-art research papers on Monocular Visual Odometry (MVO) and Visual SLAM. You can use these to compare results, formulate your novelty, and understand the existing landscape.

## 1. Feature-Based Methods (The Gold Standard)
Feature-based methods extract keypoints and track them across frames. They are robust to large motions and illumination changes.

*   **ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM** (Campos et al., 2021)
    *   *Why read it?* It is the current benchmark for feature-based visual odometry.
    *   [Abstract Link](https://arxiv.org/abs/2007.11898) | [Direct PDF Download](https://arxiv.org/pdf/2007.11898.pdf)
*   **ORB-SLAM: A Versatile and Accurate Monocular SLAM System** (Mur-Artal et al., 2015)
    *   *Why read it?* The foundational paper for the ORB-SLAM lineage.
    *   [Abstract Link](https://arxiv.org/abs/1502.00956) | [Direct PDF Download](https://arxiv.org/pdf/1502.00956.pdf)

## 2. Direct Methods
Direct methods operate directly on pixel intensities, bypassing feature extraction. They perform well in low-texture environments but are sensitive to photometric changes.

*   **Direct Sparse Odometry (DSO)** (Engel et al., 2016)
    *   *Why read it?* The pioneer of modern direct methods, offering extreme accuracy in uniform environments.
    *   [Abstract Link](https://arxiv.org/abs/1607.02565) | [Direct PDF Download](https://arxiv.org/pdf/1607.02565.pdf)
*   **LSD-SLAM: Large-Scale Direct Monocular SLAM** (Engel et al., 2014)
    *   *Why read it?* Older but highly influential direct method that generates dense 3D maps.
    *   [Abstract Link](https://arxiv.org/abs/1407.3802) | [Direct PDF Download](https://arxiv.org/pdf/1407.3802.pdf)

## 3. Semi-Direct Methods
Combining the speed of direct methods with the robustness of feature-based methods.

*   **SVO: Fast Semi-Direct Monocular Visual Odometry** (Forster et al., 2014)
    *   *Why read it?* Extremely fast, designed for micro-aerial vehicles (drones).
    *   [Direct PDF Download](https://rpg.ifi.uzh.ch/docs/ICRA14_Forster.pdf)

## 4. Visual-Inertial Odometry (VIO)
Adding IMU data to Monocular Vision solves the inherent "scale ambiguity" problem of pure monocular systems.

*   **VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator** (Qin et al., 2017)
    *   *Why read it?* The most widely cited optimization-based VIO pipeline.
    *   [Abstract Link](https://arxiv.org/abs/1708.03852) | [Direct PDF Download](https://arxiv.org/pdf/1708.03852.pdf)

## 5. Learning-Based / Deep Learning MVO
The newest trend in MVO research, replacing mathematical models with neural networks.

*   **Unsupervised Learning of Depth and Ego-Motion from Video** (Zhou et al., 2017)
    *   *Why read it?* Groundbreaking work in unsupervised learning for MVO.
    *   [Abstract Link](https://arxiv.org/abs/1704.03952) | [Direct PDF Download](https://arxiv.org/pdf/1704.03952.pdf)
*   **TartanVO: A Generalizable Learning-based VO** (Wang et al., 2020)
    *   *Why read it?* Pushed the boundaries on how well learning-based VO generalizes to unseen environments.
    *   [Abstract Link](https://arxiv.org/abs/2011.00359) | [Direct PDF Download](https://arxiv.org/pdf/2011.00359.pdf)

## 6. Recent Surveys & Comparisons (Crucial for Novelty)
*   **Comparison of Modern Open-Source Visual SLAM Approaches** (2024)
    *   *Why read it?* An excellent benchmark of how these algorithms compare today. Great for determining your baseline.
    *   [Abstract Link](https://arxiv.org/abs/2403.06341) | [Direct PDF Download](https://arxiv.org/pdf/2403.06341.pdf)

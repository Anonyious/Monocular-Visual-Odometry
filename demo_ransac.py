import numpy as np
from vo.ransac import RANSAC

# Create a simple line: y = 2x + 1
np.random.seed(42)
n_points = 200
x = np.random.uniform(-10, 10, n_points)
y = 2 * x + 1

# Add noise to inliers
inliers = np.random.choice(n_points, size=int(n_points * 0.2), replace=False) # 20% inliers
y[inliers] += np.random.normal(0, 0.5, len(inliers))

# Make outliers entirely random
outliers = np.setdiff1d(np.arange(n_points), inliers)
y[outliers] = np.random.uniform(-20, 20, len(outliers))

data = np.column_stack((x, y))

def fit_line(sample):
    # sample is (2, 2)
    x1, y1 = sample[0]
    x2, y2 = sample[1]
    if abs(x2 - x1) < 1e-6:
        return None
    m = (y2 - y1) / (x2 - x1)
    c = y1 - m * x1
    return m, c

def residual(model, data):
    m, c = model
    x = data[:, 0]
    y = data[:, 1]
    expected_y = m * x + c
    # Return Euclidean distance to line
    # d = |mx - y + c| / sqrt(m^2 + 1)
    return np.abs(m * x - y + c) / np.sqrt(m**2 + 1)

# The data has 80% outliers, meaning epsilon = 0.8
# The default starting n_iter assumes epsilon = 0.5.
print(f"True outlier ratio: 0.8. Running RANSAC...")

ransac = RANSAC(model_fn=fit_line, residual_fn=residual, min_samples=2, threshold=1.0)
best_model, mask = ransac.fit(data)

if best_model is None:
    print("RANSAC failed to find a model.")
else:
    print(f"Best model found: m={best_model[0]:.2f}, c={best_model[1]:.2f}")
    print(f"Number of inliers: {mask.sum()} / {n_points}")

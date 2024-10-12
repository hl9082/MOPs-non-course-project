import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
def icp(A, B, max_iterations=100, tolerance=1e-6):
    """
    Perform ICP (Iterative Closest Point) algorithm to align point cloud B to point cloud A.
    """
    src = np.copy(B)
    prev_error = 0

    for i in range(max_iterations):
    # Find the nearest neighbors between the current source and destination points
        tree = KDTree(A)
        distances, indices = tree.query(src)

# Compute the transformation between the current source and destination points
        T, _, _ = best_fit_transform(src, A[indices])

# Update the current source
        src = (T @ np.vstack((src.T, np.ones((1, src.shape[0]))))).T[:, :2]

# Check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
            prev_error = mean_error

    return T, distances

def best_fit_transform(A, B):
    """
    Calculate the least-squares best-fit transform that maps corresponding points A to B.
    """
    assert A.shape == B.shape

# Compute centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

# Center the points
    AA = A - centroid_A
    BB = B - centroid_B

# Compute the covariance matrix
    H = AA.T @ BB

# Compute the Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)

# Compute the rotation matrix
    R = Vt.T @ U.T

# Ensure a proper rotation (no reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

# Compute the translation vector
    t = centroid_B.T - R @ centroid_A.T

# Construct the transformation matrix
    T = np.identity(3)
    T[:2, :2] = R
    T[:2, 2] = t

    return T, R, t

def main():
# Example usage with random data sets
    seed=int(input("Enter a seed here: "))
    np.random.seed(seed)
    rows=int(input("Enter the number of rows: "))
    cols=int(input("Enter the number of columns: "))
    A = np.random.rand(rows, cols)  # Large data set
    B = A + np.random.normal(0, 0.1, A.shape)

# Apply ICP
    T, distances = icp(A, B)

# Transform B using the computed transformation
    B_transformed = (T @ np.vstack((B.T, np.ones((1, B.shape[0]))))).T[:, :2]

# Plot the results
    plt.scatter(A[:, 0], A[:, 1], color='blue', label='Original')
    plt.scatter(B[:, 0], B[:, 1], color='red', label='Before ICP')
    plt.scatter(B_transformed[:, 0], B_transformed[:, 1], color='green', label='After ICP')
    plt.legend()
    plt.show()

if __name__=='__main__':
    main()
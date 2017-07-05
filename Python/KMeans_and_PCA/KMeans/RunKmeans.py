import numpy as np
from Python.KMeans_and_PCA.KMeans.FindClosestCentroids import find_closest_centroids

from Python.KMeans_and_PCA.KMeans.ComputeCentroids import compute_centroids


def run_k_means(X, initial_centroids, max_iters):
    m, n = X.shape
    k = initial_centroids.shape[0]
    idx = np.zeros(m)
    centroids = initial_centroids

    for i in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, k)

    return idx, centroids

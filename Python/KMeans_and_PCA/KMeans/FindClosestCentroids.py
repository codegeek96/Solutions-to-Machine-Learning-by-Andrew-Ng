import numpy as np


def find_closest_centroids(X, centroids):
    idx = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        K = np.sum(np.power(centroids - X[i, :], 2), axis=1)
        idx[i] = np.argmin(K)

    return idx

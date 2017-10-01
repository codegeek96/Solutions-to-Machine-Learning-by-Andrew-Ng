"""Preprocess data by subtracting mean rating for every
   movie (every row)
"""

import numpy as np


def normalize_ratings(Y, R, movies, users):

    Y_mean = np.zeros((movies, 1))
    Y_norm = np.zeros((movies, users))

    for i in range(movies):
        idx = np.where(R[i, :] == 1)[0]
        Y_mean[i] = Y[i, idx].mean()
        Y_norm[i, idx] = Y[i, idx] - Y_mean[i]

    return Y_mean, Y_norm

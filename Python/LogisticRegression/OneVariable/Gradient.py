import numpy as np

from Python.LogisticRegression.Sigmoid import sigmoid


def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    return ((sigmoid(X * theta.T) - y).T * X) / len(X)

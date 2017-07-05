import numpy as np

from Python.LogisticRegression.Sigmoid import sigmoid


def gradient_reg(theta, X, y, learning_rate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    error = (sigmoid(X * theta.T) - y).T

    grad = np.zeros(theta.shape[1])
    grad[0] = (error * X[:, 0]) / len(X)
    grad[1:] = (error * X[:, 1:]) / len(X) + ((learning_rate / len(X)) * theta[:, 1:])

    return grad

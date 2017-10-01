import numpy as np

from LogisticRegression.Sigmoid import sigmoid


def cost_reg(theta, X, y, learning_rate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply(1 - y, np.log(1 - sigmoid(X * theta.T)))
    reg = (learning_rate / (2 * len(X))) * np.sum(np.power(theta[:, 1:], 2))
    return np.sum(first - second) / len(X) + reg

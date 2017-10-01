import numpy as np

from .ComputeCost import compute_cost


def gradient_descent(x, y, theta, alpha, iterations):
    m = len(x)
    cost = np.zeros((iterations, 1))

    for i in range(iterations):
        error = (x * theta) - y
        error = error.T * x

        theta = theta - ((alpha / m) * error.T)
        cost[i] = compute_cost(x, y, theta)

    return theta.T, cost

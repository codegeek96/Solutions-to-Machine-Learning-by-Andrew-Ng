import numpy as np


def compute_cost(x, y, theta):
    inner = np.power(((x * theta) - y), 2)
    return np.sum(inner) / (2 * len(x))

import numpy as np


def gaussian_kernel(x1, x2, sigma):
    return np.exp(-(np.sum((x1 - x2) ** 2) / (2 * (sigma **2))))


if __name__ == '__main__':
    print('\nEvaluating the Gaussian Kernel ...')
    x1 = np.array([1.0, 2.0, 1.0])
    x2 = np.array([0.0, 4.0, -1.0])
    sigma = 2
    print(gaussian_kernel(x1, x2, sigma))

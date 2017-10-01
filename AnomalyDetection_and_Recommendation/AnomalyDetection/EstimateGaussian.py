"""This function estimates the parameters of a
   Gaussian distribution using the data in X
"""


def estimate_gaussian(X):
    mu = X.mean(axis=0)
    sigma = X.var(axis=0)

    return mu, sigma

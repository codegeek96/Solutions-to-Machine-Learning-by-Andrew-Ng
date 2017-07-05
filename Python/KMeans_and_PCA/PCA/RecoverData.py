"""Recovers an approximation of the original data
   when using the projected data
"""


def recover_data(Z, U, k):
    U_reduced = U[:, :k]
    return Z * U_reduced.T

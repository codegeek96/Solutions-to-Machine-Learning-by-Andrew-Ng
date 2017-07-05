"""Computes the reduced data representation
   when projecting only on to the top k eigenvectors
"""


def project_data(X, U, k):
    U_reduced = U[:, :k]
    return X * U_reduced

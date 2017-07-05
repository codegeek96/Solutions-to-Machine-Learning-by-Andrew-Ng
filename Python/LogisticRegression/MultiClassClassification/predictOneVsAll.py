import numpy as np

from Python.LogisticRegression.Sigmoid import sigmoid


def predict_one_vs_all(all_theta, X):
    rows = X.shape[0]
    params = X.shape[1]
    num_labels = all_theta.shape[0]

    # insert ones to match the shape
    X = np.insert(X, 0, values=np.ones(rows),axis=1)

    # convert to matrices
    X = np.matrix(X)
    all_theta = np.matrix(all_theta)

    # compute the class probability for each class on each training instance
    h = sigmoid(X * all_theta.T)

    # create array of the index with the maximum probability
    h_argmax = np.argmax(h, axis=1)

    # because our array was zero-indexed we need to add one for the true label prediction
    h_argmax = h_argmax + 1

    return h_argmax

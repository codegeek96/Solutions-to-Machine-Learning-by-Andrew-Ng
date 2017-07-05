import numpy as np

from scipy.optimize import minimize

from Python.LogisticRegression.Regularized.CostReg import cost_reg
from Python.LogisticRegression.Regularized.GradientReg import gradient_reg


def one_vs_all(X, y, num_labels, learning_rate):
    rows = X.shape[0]
    params = X.shape[1]

    # k X (n + 1) array for the parameters of each of the k classifiers
    all_theta = np.zeros((num_labels, params + 1))

    # insert a column of ones at the beginning for the intercept term
    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    # labels are 1-indexed instead of 0-indexed
    for i in range(1, num_labels + 1):
        theta = np.zeros(params + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))

        # minimize the objective function
        fmin = minimize(fun=cost_reg, x0=theta, args=(X, y_i, learning_rate), method='TNC', jac=gradient_reg)
        all_theta[i-1, :] = fmin.x



    return all_theta

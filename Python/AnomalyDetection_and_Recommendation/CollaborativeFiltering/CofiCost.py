"""Collaborative filtering cost function
"""

import numpy as np


def cofi_cost(params, Y, R, num_features, learning_rate):
    Y = np.matrix(Y)  # (1682, 943)
    R = np.matrix(R)  # (1682, 943)
    num_movies = Y.shape[0]
    num_users = Y.shape[1]

    # reshape the parameter array into parameter matrices
    # Shapes :=  X: (1682, 10), theta: (943, 10)
    X = np.matrix(np.reshape(params[:num_movies * num_features], (num_movies, num_features)))
    Theta = np.matrix(np.reshape(params[num_movies * num_features:], (num_users, num_features)))

    # compute the cost
    error = np.multiply(X * Theta.T - Y, R)  # (1682, 943)
    sq_error = np.power(error, 2)
    J = (1 / 2) * np.sum(sq_error)

    # add the cost regularization
    J += (learning_rate / 2) * np.sum(np.power(Theta, 2))
    J += (learning_rate / 2) * np.sum(np.power(X, 2))

    # calculate the gradients
    X_grad = error * Theta  # (1682, 10)
    Theta_grad = error.T * X  # (943, 10)

    # calculate the gradients with regularization
    X_grad = (error * Theta) + (learning_rate * X)
    Theta_grad = (error.T * X) + (learning_rate * Theta)

    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(X_grad), np.ravel(Theta_grad)))

    return J, grad

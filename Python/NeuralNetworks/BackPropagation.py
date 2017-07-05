import numpy as np

from NeuralNetworks.ForwardPropagate import forward_propagate
from NeuralNetworks.SigmoidGradient import sigmoid_gradient


def back_propagation(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, input_size + 1)))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, hidden_size + 1)))

    # run the feed-forward pass
    # # a1 = (5000, 401)
    # # z2 = (5000, 25)
    # # a2 = (5000, 26)
    # # z3 = (5000, 10)
    # # h = (5000, 10)
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    # initializations
    J = 0
    delta1 = np.zeros(theta1.shape)  # (25, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)

    for i in range(m):

        # compute the cost
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply(1 - y[i, :], np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)

        # perform backpropagation
        a1i = a1[i, :]  # (1, 401)
        z2i = z2[i, :]  # (1, 25)
        a2i = a2[i, :]  # (1, 26)
        hi = h[i, :]  # (1, 10)
        yi = y[i, :]  # (1, 10)

        d3i = hi - yi  # (1, 10)

        z2i = np.insert(z2i, 0, values=np.ones(1))  # (1, 26)
        d2i = np.multiply((theta2.T * d3i.T).T, sigmoid_gradient(z2i))  # (1, 26)

        delta1 += (d2i[:, 1:]).T * a1i
        delta2 += d3i.T * a2i

    J /= m
    delta1 /= m
    delta2 /= m

    # add the gradient regularization term
    delta1[:, 1:] += (theta1[:, 1:] * learning_rate) / m
    delta2[:, 1:] += (theta2[:, 1:] * learning_rate) / m

    # add the cost regularization term
    J += (learning_rate / (2 * m)) * (np.sum(np.power(theta1[:, 1], 2)) + np.sum(np.power(theta2[:, 1], 2)))

    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))

    return J, grad

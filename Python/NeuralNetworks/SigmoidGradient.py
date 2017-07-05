import numpy as np

from Python.NeuralNetworks.Sigmoid import sigmoid


def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), 1 - sigmoid(z))

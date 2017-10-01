import numpy as np

from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
from scipy.optimize import minimize

from NeuralNetworks.BackPropagation import back_propagation
from NeuralNetworks.ForwardPropagate import forward_propagate


# Load data
data = loadmat('ex3data1.mat')
X = data['X']
y = data['y']
print('\nX shape:', X.shape, ' y shape:', y.shape)

encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y)
print('y_onehot shape:', y_onehot.shape)


# initial setup
input_size = 400
hidden_size = 25
num_labels = 10
learning_rate = 1


# randomly initialize a parameter array of the size of the full network's parameters
params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.25

print('\nInitial cost:', back_propagation(params, input_size, hidden_size, num_labels, X, y_onehot, learning_rate)[0])


# minimize the objective function
print('\nminimizing the objective function...')
fmin = minimize(fun=back_propagation, x0=params, args=(input_size, hidden_size, num_labels, X, y_onehot, learning_rate),
                method='TNC', jac=True, options={'maxiter': 250})
print(fmin)


# Training Neural Network
print('\nTraining Neural Network...')
X = np.matrix(X)
theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
y_pred = np.array(np.argmax(h, axis=1) + 1)

correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
accuracy = sum(map(int, correct)) / float(len(correct))
print('accuracy = {0}%'.format(accuracy * 100))

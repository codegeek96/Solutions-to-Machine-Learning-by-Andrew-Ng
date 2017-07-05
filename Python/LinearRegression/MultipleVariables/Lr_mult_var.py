import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from LinearRegression.MultipleVariables.GradientDescent import gradient_descent

from LinearRegression.MultipleVariables.ComputeCost import compute_cost

# ----------------------- Load data ------------------------------------
data = pd.read_csv('ex1data2.txt', header=None, names=['Size', 'Bedrooms', 'Price'])
print(data.head())

# Feature Normalization
data = (data - data.mean()) / data.std()
print('\nAfter Feature Normalization:')
print(data.head())


# ----------------------- Gradient Descent -----------------------------
print('Running Gradient Descent ...')

# append a ones column to the front of the data set
data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:, :cols - 1]
y = data.iloc[:, cols-1:]

# convert from data frames to numpy matrices
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0, 0, 0]))

print('Initial cost:', compute_cost(X, y, theta.T))

# initialize variables for learning rate and iterations
alpha = 0.01
iterations = 1000

theta, cost = gradient_descent(X, y, theta.T, alpha, iterations)
print('Theta found by gradient descent:', theta)
print('Cost after Gradient Descent:', cost[-1])


# ----------------------- Viewing the result ----------------------
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iterations), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()

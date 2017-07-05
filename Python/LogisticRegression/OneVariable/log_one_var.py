import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt

from LogisticRegression.OneVariable.Cost import cost_function
from LogisticRegression.OneVariable.Gradient import gradient
from LogisticRegression.Predict import predict
from LogisticRegression.Sigmoid import sigmoid

# ----------------------- Load data ------------------------------------
data = pd.read_csv('ex2data1.txt', header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
print(data.head())

# add a ones column - this makes the matrix multiplication work out easier
data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
X = data.iloc[:, :-1]
y = data.iloc[:, -1:]

# convert from data frames to numpy arrays
# here we don't use matrix because opt function accepts arrays only
X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(3)


# ----------------------- Plotting -------------------------------------
print('\nPlotting Data ...')

positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()

nums = np.arange(-10, 10, step=1)
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(nums, sigmoid(nums), 'r')
plt.show()


# ----------------------- Logistic Regression -----------------------------
print('\nStarting Logistic Regression ...')

print('Initial Cost:', cost_function(theta, X, y))

result = opt.fmin_tnc(func=cost_function, x0=theta, fprime=gradient, args=(X, y))
print('Final Cost:', cost_function(result[0].T, X, y))

theta = np.matrix(result[0])
predictions = predict(theta, X)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = sum(map(int, correct)) % len(correct)
print('accuracy = {0}%'.format(accuracy))

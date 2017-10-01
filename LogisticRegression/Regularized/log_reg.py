import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt

from LogisticRegression.Regularized.MapFeatures import map_features
from LogisticRegression.Regularized.CostReg import cost_reg
from LogisticRegression.Regularized.GradientReg import gradient_reg
from LogisticRegression.Predict import predict


# ----------------------- Load data ------------------------------------
data = pd.read_csv('ex2data2.txt', header=None, names=['Test 1', 'Test 2', 'Accepted'])
print(data.head())


# ----------------------- Plotting -------------------------------------
print('\nPlotting Data ...')

positive = data[data['Accepted'].isin([1])]
negative = data[data['Accepted'].isin([0])]

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='Not Accepted')
ax.legend()
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')
plt.show()


# ----------------------- Logistic Regression -----------------------------
print('\nStarting Logistic Regression ...')

data = map_features(data)
print(data.head())

# set X and y
X = data.iloc[:, 1:]
y = data.iloc[:, :1]

# convert to numpy arrays and initialize the parameter array theta
X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(X.shape[1])

learning_rate = 1

print('\nFinal Cost:', cost_reg(theta, X, y, learning_rate))

result = opt.fmin_tnc(func=cost_reg, x0=theta, fprime=gradient_reg, args=(X, y, learning_rate))
print(result)

theta = np.matrix(result[0])
predictions = predict(theta, X)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = sum(map(int, correct)) % len(correct)
print('accuracy = {0}%'.format(accuracy))

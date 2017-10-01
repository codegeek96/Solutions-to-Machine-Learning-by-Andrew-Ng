import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import linear_model


# ----------------------- Load data ------------------------------------
data = pd.read_csv('ex1data1.txt', header=None, names=['Population', 'Profit'])
print(data.head())


# ----------------------- Plotting -------------------------------------
print('Plotting Data ...')

data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))
plt.show()


# ----------------------- Linear Regression -----------------------------
print('Starting Linear Regression ...')

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:, :cols - 1]
y = data.iloc[:, cols-1:]

# convert from data frames to numpy matrices
X = np.matrix(X.values)
y = np.matrix(y.values)

model = linear_model.LinearRegression()
model.fit(X, y)
print('Intercept:', model.intercept_)
print('Coefficient:', model.coef_)

# ----------------------- Viewing the result ----------------------
x = np.array(X[:, 0].A1)
f = model.predict(X).flatten()

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()

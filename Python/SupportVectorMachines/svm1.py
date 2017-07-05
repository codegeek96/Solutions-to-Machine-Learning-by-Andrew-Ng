import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import loadmat
from sklearn import svm


# ----------------------- Load data ------------------------------------
raw_data = loadmat('ex6data1.mat')
print(raw_data)

data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2'])
data['y'] = raw_data['y']


# ----------------------- Plotting -------------------------------------
print('\nPlotting Data ...')

positive = data[data['y'].isin([1])]
negative = data[data['y'].isin([0])]

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['X1'], positive['X2'], s=50, marker='x', label='Positive')
ax.scatter(negative['X1'], negative['X2'], s=50, marker='o', label='Negative')
ax.legend()
plt.show()


# ----------------------- Linear SVM -----------------------------
print('\nRunning Linear SVM ...')

svc = svm.LinearSVC(C=1, loss='hinge', max_iter=1000)
print(svc)
svc.fit(data[['X1', 'X2']], data['y'])
print('\nScore for C=1:', svc.score(data[['X1', 'X2']], data['y']))

svc2 = svm.LinearSVC(C=100, loss='hinge', max_iter=1000)
svc2.fit(data[['X1', 'X2']], data['y'])
print('Score for C=100:', svc2.score(data[['X1', 'X2']], data['y']))


# ----------------------- Viewing the result ----------------------
data['SVM 1 Confidence'] = svc.decision_function(data[['X1', 'X2']])
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(data['X1'], data['X2'], s=50, c=data['SVM 1 Confidence'], cmap='seismic')
ax.set_title('SVM (C=1) Decision Confidence')
plt.show()

data['SVM 2 Confidence'] = svc2.decision_function(data[['X1', 'X2']])
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(data['X1'], data['X2'], s=50, c=data['SVM 2 Confidence'], cmap='seismic')
ax.set_title('SVM (C=100) Decision Confidence')
plt.show()

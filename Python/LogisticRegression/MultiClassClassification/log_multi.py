import numpy as np

from scipy.io import loadmat

from Python.LogisticRegression.MultiClassClassification.oneVsAll import one_vs_all
from Python.LogisticRegression.MultiClassClassification.predictOneVsAll import predict_one_vs_all


# ----------------------- Load data ------------------------------------
data = loadmat('ex3data1.mat')
print(data)
print('Dimensions:', data['X'].shape, ',', data['y'].shape)

print('Class Labels:', np.unique(data['y']))


# ----------------------- Logistic Regression -----------------------------
print('\nTraining One-vs-All Logistic Regression...')
all_theta = one_vs_all(data['X'], data['y'], 10, 1)


# ----------------------- Predict for One-Vs-All --------------------------
y_pred = predict_one_vs_all(all_theta, data['X'])
correct = [1 if a == b else 0 for (a, b) in zip(y_pred, data['y'])]
accuracy = sum(map(int, correct)) / float(len(correct))
print('accuracy = {0}%'.format(accuracy * 100))

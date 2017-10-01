import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy import stats

from AnomalyDetection_and_Recommendation.AnomalyDetection.EstimateGaussian import estimate_gaussian
from AnomalyDetection_and_Recommendation.AnomalyDetection.SelectThreshold import select_threshold


# ----------------------- Load data ------------------------------------
data = loadmat('ex8data1.mat')
X = data['X']
print('Shape of X:', X.shape)


# ----------------------- Plotting -------------------------------------
print('\nPlotting Data ...')
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(X[:, 0], X[:, 1])
plt.show()


# ----------------------- Anomaly Detection -------------------------------------

# Find mean and variance for each feature
mu, sigma = estimate_gaussian(X)
print('\nMean:', mu)
print('Variance:', sigma)

# Estimate the threshold
Xval = data['Xval']
yval = data['yval']
print('\nShape X and y in cross validation set:', Xval.shape, yval.shape)

# Finding the probability density of each of the values in our data set
# given the Gaussian model parameters we calculated above.
p = stats.norm(mu, sigma).pdf(X)
print('Shape of PDF for training set:', p.shape)

pval = stats.norm(mu, sigma).pdf(Xval)
print('Shape of PDF for cross validation set:', pval.shape)

epsilon, f1 = select_threshold(pval, yval)
print('\nEpsilon:', epsilon)
print('F1 score:', f1)


# ----------------------- Viewing the result ----------------------
# indexes of the values considered to be outliers
outliers = np.where(p < epsilon)

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(X[:, 0], X[:, 1])
ax.scatter(X[outliers[0], 0], X[outliers[0], 1], s=50, color='r', marker='o')
plt.show()


# ----------------------- Multidimensional Outliers ----------------------
print('\n\nRunning anomaly detection on a dataset with many features...')

data2 = loadmat('ex8data2.mat')
X2 = data2['X']
print('\nShape of X:', X2.shape)

mu2, sigma2 = estimate_gaussian(X2)
Xval2 = data2['Xval']
yval2 = data2['yval']
print('Shape X and y in cross validation set:', Xval2.shape, yval2.shape)

p2 = stats.norm(mu2, sigma2).pdf(X2)
print('Shape of PDF for training set:', p2.shape)
pval2 = stats.norm(mu2, sigma2).pdf(Xval2)
print('Shape of PDF for cross validation set:', pval2.shape)

epsilon2, f1_2 = select_threshold(pval2, yval2)
print('\nEpsilon:', epsilon2)
print('F1 score:', f1_2)

print('\nOutliers found:', np.sum(p2 < epsilon2))

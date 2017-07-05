import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat

from KMeans_and_PCA.PCA.RunPCA import pca
from KMeans_and_PCA.PCA.ProjectData import project_data
from KMeans_and_PCA.PCA.RecoverData import recover_data


# ----------------------- Load data ------------------------------------
data = loadmat('ex7data1.mat')
X = data['X']


# ----------------------- Plotting -------------------------------------
print('Plotting Data ...')

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(X[:, 0], X[:, 1])
plt.show()


# ----------------------- PCA -----------------------------
print('\n\nRunning PCA on example dataset...')

U, S, V = pca(X)
Z = project_data(X, U, 1)
print('First reduced example value:', Z[0])

X_recovered = recover_data(Z, U, 1)
print('First recovered example value:', X_recovered[0])


# ----------------------- Viewing the result ----------------------
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(np.array(X_recovered[:, 0]), np.array(X_recovered[:, 1]))
plt.show()


# ----------------------- PCA on Face Dataset ----------------------
print('\n\nLoading face dataset...')

faces = loadmat('ex7faces.mat')
X = faces['X']
print('Face dataset shape:', X.shape)

print('\nLoading one face..')
face = np.reshape(X[3, :], (32, 32))
plt.imshow(face)
plt.show()

print('\nRunning PCA on face dataset...\n')
U, S, V = pca(X)
Z = project_data(X, U, 100)

X_recovered = recover_data(Z, U, 100)
face = np.reshape(X_recovered[3, :], (32, 32))
plt.imshow(face)
plt.show()

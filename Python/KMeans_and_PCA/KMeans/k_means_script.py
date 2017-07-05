import matplotlib.pyplot as plt
import numpy as np

from scipy.io import loadmat

from Python.KMeans_and_PCA.KMeans.ComputeCentroids import compute_centroids
from Python.KMeans_and_PCA.KMeans.FindClosestCentroids import find_closest_centroids
from Python.KMeans_and_PCA.KMeans.RunKmeans import run_k_means
from Python.KMeans_and_PCA.KMeans.InitCentroids import init_centroids

# ----------------------- Load data ------------------------------------
data = loadmat('ex7data2.mat')
X = data['X']
k = 3
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])


# ----------------------- K-means -----------------------------
idx = find_closest_centroids(X, initial_centroids)
print('\nClosest centroids for the first 3 examples:', idx[0:3])

print('\nCentroids computed after initial finding of closest centroids:\n', compute_centroids(X, idx, k))


# Example Dataset
print('\n\nRunning K-Means clustering on example dataset..\n')
idx, centroids = run_k_means(X, initial_centroids, 10)

# Plotting the clusters
cluster1 = X[np.where(idx == 0)[0], :]
cluster2 = X[np.where(idx == 1)[0], :]
cluster3 = X[np.where(idx == 2)[0], :]

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(cluster1[:, 0], cluster1[:, 1], s=30, color='r', label='Cluster 1')
ax.scatter(cluster2[:, 0], cluster2[:, 1], s=30, color='g', label='Cluster 2')
ax.scatter(cluster3[:, 0], cluster3[:, 1], s=30, color='b', label='Cluster 3')
ax.legend()
plt.show()


# ----------------------- K-means for image compression -----------------------------
print('\nRunning K-Means clustering on pixels from an image...\n')

image_data = loadmat('bird_small.mat')
A = image_data['A']
print('Shape of pixel matrix:', A.shape)

# normalize value ranges
A = A / 255  # Divide by 255 so that all values are in the range 0 - 1

# Reshape the image into an Nx3 matrix where N = number of pixels.
# Each row will contain the Red, Green and Blue pixel values
# This gives us our dataset matrix X that we will use K-Means on.
X = np.reshape(A, (A.shape[0] * A.shape[1], A.shape[2]))

k = 16
max_iters = 10

# randomly initialize the centroids
initial_centroids = init_centroids(X, k)

# run the algorithm
idx, centroids = run_k_means(X, initial_centroids, max_iters)

# get the closest centroids one last time
idx = find_closest_centroids(X, centroids)

# map each pixel to the centroid value
X_recovered = centroids[idx.astype(int), :]

# reshape to the original dimensions
X_recovered = np.reshape(X_recovered, (A.shape[0], A.shape[1], A.shape[2]))

plt.imshow(X_recovered)
plt.show()

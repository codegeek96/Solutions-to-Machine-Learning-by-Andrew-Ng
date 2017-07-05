import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.optimize import minimize

from Python.AnomalyDetection_and_Recommendation.CollaborativeFiltering.CofiCost import cofi_cost
from Python.AnomalyDetection_and_Recommendation.CollaborativeFiltering.NormalizeRatings import normalize_ratings


# ----------------------- Load data ------------------------------------
data = loadmat('ex8_movies.mat')
Y = data['Y']
R = data['R']
print('Shape of Y and R:', Y.shape, R.shape)

print('Average rating for movie 1 (Toy Story):', Y[0, R[0, :]].mean())


# ----------------------- Plotting -------------------------------------
print('\nPlotting Data ...\n\n')
fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(Y)
ax.set_xlabel('Users')
ax.set_ylabel('Movies')
fig.tight_layout()
plt.show()


# ----------------------- Collaborative Filtering -------------------------------------
print('Finding recommendations...\n')

# To test this we reduce the data set size so that it runs faster

# First we implement unregularized procedure
learning_rate = 0

users = 4
movies = 5
features = 3

params_data = loadmat('ex8_movieParams.mat')
X = params_data['X']
Theta = params_data['Theta']

X_sub = X[:movies, :features]
Theta_sub = Theta[:users, :features]
Y_sub = Y[:movies, :users]
R_sub = R[:movies, :users]
params = np.concatenate((np.ravel(X_sub), np.ravel(Theta_sub)))

J, grad = cofi_cost(params, Y_sub, R_sub, features, learning_rate)
print('Initial cost:', J)
print('Initial gradient:', grad)

# Regularization
learning_rate = 1.5
J, grad = cofi_cost(params, Y_sub, R_sub, features, learning_rate)
print('\nRegularized cost:', J)
print('Regularized gradient:', grad)


# ----------------------- Entering ratings for a new user -------------------------------------
# Before we will train the collaborative filtering model, we will first
# add ratings that correspond to a new user that we just observed.

movie_idx = {}
with open('movie_ids.txt') as f:
    for line in f:
        tokens = line.split(' ')
        tokens[-1] = tokens[-1][:-1]
        movie_idx[int(tokens[0]) - 1] = ' '.join(tokens[1:])

# Initialize my ratings
ratings = np.zeros((1682, 1))

# For example, Toy Story (1995) has ID 1, so to rate it "4"
ratings[0] = 4

# Or suppose did not enjoy Silence of the Lambs (1991), you can set
ratings[97] = 2

# We have selected a few movies we liked / did not like and the ratings we
# gave are as follows:
ratings[6] = 3
ratings[11] = 5
ratings[53] = 4
ratings[63] = 5
ratings[65] = 3
ratings[68] = 5
ratings[182] = 4
ratings[225] = 5
ratings[354] = 5

print('\nNew user ratings:')
print('Rated {0} with {1} stars.'.format(movie_idx[0], str(int(ratings[0]))))
print('Rated {0} with {1} stars.'.format(movie_idx[6], str(int(ratings[6]))))
print('Rated {0} with {1} stars.'.format(movie_idx[11], str(int(ratings[11]))))
print('Rated {0} with {1} stars.'.format(movie_idx[53], str(int(ratings[53]))))
print('Rated {0} with {1} stars.'.format(movie_idx[63], str(int(ratings[63]))))
print('Rated {0} with {1} stars.'.format(movie_idx[65], str(int(ratings[65]))))
print('Rated {0} with {1} stars.'.format(movie_idx[68], str(int(ratings[68]))))
print('Rated {0} with {1} stars.'.format(movie_idx[97], str(int(ratings[97]))))
print('Rated {0} with {1} stars.'.format(movie_idx[182], str(int(ratings[182]))))
print('Rated {0} with {1} stars.'.format(movie_idx[225], str(int(ratings[225]))))
print('Rated {0} with {1} stars.'.format(movie_idx[354], str(int(ratings[354]))))


# ----------------------- Learning Movie Ratings -------------------------------------
# Now, you will train the collaborative filtering model on a movie rating
# dataset of 1682 movies and 943 users
print('\n\nLearning Movie Ratings...\n')

# Add our own ratings to the data matrix
Y = np.append(Y, ratings, axis=1)  # (1682, 944)
R = np.append(R, ratings != 0, axis=1)  # (1682, 944)

movies = Y.shape[0]
users = Y.shape[1]
features = 10
learning_rate = 10

X = np.random.random(size=(movies, features))
Theta = np.random.random(size=(users, features))
params = np.concatenate((np.ravel(X), np.ravel(Theta)))

# Normalize Ratings
Y_mean, Y_norm = normalize_ratings(Y, R, movies, users)

fmin = minimize(fun=cofi_cost, x0=params, args=(Y_norm, R, features, learning_rate),
                method='CG', jac=True, options={'maxiter': 100})

# Reshape the matrices back to their original dimensions
X = np.matrix(np.reshape(fmin.x[:movies * features], (movies, features)))
Theta = np.matrix(np.reshape(fmin.x[movies * features:], [users, features]))


# ----------------------- Recommendations for user --------------------------------------
# After training the model, we can now make recommendations for the  user added earlier
# by computing the predictions matrix.

predictions = X * Theta.T
my_predictions = predictions[:, -1] + Y_mean

print('\nTop recommendations for you:')
idx = np.argsort(my_predictions, axis=0)[::-1]
for i in range(10):
    j = int(idx[i])
    print('Predicted rating of {0} for movie {1}.'.format(str(round(float(my_predictions[j]), 2)), movie_idx[j]))

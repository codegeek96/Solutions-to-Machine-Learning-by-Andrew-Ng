import numpy as np

from scipy.io import loadmat
from sklearn import svm


# ----------------------- Load data ------------------------------------
spam_train = loadmat('spamTrain.mat')
spam_test = loadmat('spamTest.mat')

X = spam_train['X']
Xtest = spam_test['Xtest']
y = spam_train['y'].ravel()
ytest = spam_test['ytest'].ravel()

print(X.shape, y.shape, Xtest.shape, ytest.shape)


# ----------------------- SVM -----------------------------
print('\nRunning SVM...')

svc = svm.SVC()
svc.fit(X, y)
print('Test accuracy = {0}%'.format(np.round(svc.score(Xtest, ytest) * 100, 2)))

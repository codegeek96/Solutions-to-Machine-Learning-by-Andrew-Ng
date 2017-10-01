import numpy as np


def map_features(data):
    degree = 5
    x1 = data['Test 1']
    x2 = data['Test 2']

    data.insert(3, 'Ones', 1)

    for i in range(1, degree):
        for j in range(i):
            data['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j)

    data.drop('Test 1', axis=1, inplace=True)
    data.drop('Test 2', axis=1, inplace=True)

    return data

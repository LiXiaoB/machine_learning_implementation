import numpy as np


def check_data(X, y):
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError('X and y must be numpy.ndarray')
    if X.shape[0] != y.shape[0]:
        raise ValueError('dim 0 of X and y are not aligned')
    # force y clsses to be 0, 1, 2, .. etc
    unique, y = np.unique(y, return_inverse=True)
    print(y.shape)
    c = unique.shape[0] # nb of classes
    if c < 2:
        raise ValueError("number of classes must be >= 2")
    if y.ndim == 1:
        y = y[:, np.newaxis]
    print("X dim:", X.shape)
    print("y dim:", y.shape)
    print("y classes:", c)
    return X, y, c

# X = np.array([[0.7, 0.1, 0.9],
#               [0.1, 0.8, 0.1],
#               [0.8, 0.2, 0.8]])
# y = np.array([1, 0, 1]).reshape(3, 1)
#
# print(np.unique(y, return_inverse=True))
# check_data(X, y)

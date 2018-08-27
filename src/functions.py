import numpy as np


def computeCost(X, y, theta):
    m = len(y)
    hx = np.transpose(theta) * X
    error_sqr = (hx - y) ** 2
    return np.sum(error_sqr, axis=0) / (2 * m)

import numpy as np


def compute_cost(X, y, theta):
    m = len(y)
    h = np.transpose(theta) * X
    error_sqr = (h - y) ** 2
    return np.sum(error_sqr, axis=0) / (2 * m)


def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    j_history = np.zeros((num_iters, 1))
    for i in range(num_iters):
        h = np.transpose(theta) * X
        errors = (h - y)
        gradient = (alpha / m) * np.sum(errors * X)
        # make gradient broadcast-able to theta
        gradient = np.reshape(gradient, (1,))
        theta = theta - gradient
        j_history[i] = compute_cost(X, y, theta) # [0]?
    return theta, j_history

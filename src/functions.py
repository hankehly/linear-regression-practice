import os

import pandas as pd

import numpy as np

data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data')


def read_dataset_2():
    df = pd.read_csv(os.path.join(data_dir, 'dataset_2/ex1data1.txt'), names=['X', 'y'], header=None)
    X = np.vstack(df['X'])
    X_0 = np.ones(X.shape)
    X = np.hstack((X_0, X))
    y = np.vstack(df['y'])
    n = X.shape[1]
    theta = np.zeros((n, 1))
    alpha = 0.01
    num_iters = 1500
    return X, y, theta, alpha, num_iters


def compute_cost(X, y, theta):
    m = len(y)
    h = X.dot(theta)
    error_sqr = (h - y) ** 2
    return np.sum(error_sqr, axis=0) / (2 * m)


def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    j_history = np.zeros((num_iters, 1))
    for i in range(num_iters):
        h = X.dot(theta)
        errors = h - y
        gradient = (alpha / m) * np.transpose(X).dot(errors)
        theta = theta - gradient
        j_history[i] = compute_cost(X, y, theta)
    return theta, j_history

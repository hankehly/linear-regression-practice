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


def read_dataset_1():
    df = pd.read_csv(os.path.join(data_dir, 'dataset_1/train.csv'))
    X = np.vstack(df['x']) / 10
    X_0 = np.ones(X.shape)
    X = np.hstack((X_0, X))
    y = np.vstack(df['y']) / 10
    n = X.shape[1]
    theta = np.zeros((n, 1))
    alpha = 0.03
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
        gradient = (alpha / m) * X.T.dot(errors)
        theta = theta - gradient
        j_history[i] = compute_cost(X, y, theta)
    return theta, j_history


def predict(val, theta):
    return np.array([1, val]).dot(theta)


def measure_dataset_1_model_accuracy():
    X, y, theta, alpha, num_iters = read_dataset_1()
    theta, _ = gradient_descent(X, y, theta, alpha, num_iters)

    df = pd.read_csv(os.path.join(data_dir, 'dataset_1/test.csv'))
    m = len(df)
    for i in range(m):
        row = df.iloc[i]
        prediction = predict(row['x'], theta)
        diff = np.abs(prediction - row['y'])
        print('prediction: %s, actual: %s (diff: %s)' % (prediction, row['y'], diff))

import numpy as np

from src.functions import read_dataset_2, compute_cost, gradient_descent

if __name__ == '__main__':
    X, y, theta, alpha, num_iters = read_dataset_2()
    new_theta, j_history = gradient_descent(X, y, theta, alpha, num_iters)
    print(new_theta)

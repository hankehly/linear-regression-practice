import numpy as np

from src.functions import read_dataset_2, compute_cost, gradient_descent

if __name__ == '__main__':
    # X, y, theta, alpha, num_iters = read_dataset_2()
    X = np.array([
        [1, 5],
        [1, 2],
        [1, 4],
        [1, 5]
    ])
    y = np.vstack(np.array([1, 6, 4, 2]))
    # print(compute_cost(X, y, theta))
    new_theta, j_history = gradient_descent(X, y, np.vstack(np.zeros(2, )), 0.01, 1000)
    print(new_theta)

    # X = np.array([
    #     [1, 2, 3],
    #     [1, 3, 4],
    #     [1, 4, 5],
    #     [1, 5, 6]
    # ])
    #
    # y = np.array([
    #     [7],
    #     [6],
    #     [5],
    #     [4]
    # ])
    #
    # t = np.array([
    #     [0.1],
    #     [0.2],
    #     [0.3]
    # ])
    #
    # print(compute_cost(X, y, t))

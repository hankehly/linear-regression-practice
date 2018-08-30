import matplotlib.pyplot as plt

from src.functions import read_dataset_2, gradient_descent, predict

if __name__ == '__main__':
    X, y, theta, alpha, num_iters = read_dataset_2()
    theta, j_history = gradient_descent(X, y, theta, alpha, num_iters)

    # Plot regression line alongside scatter plot of data points
    # plt.scatter(X[:, 1], y, c='#FF0000', marker='x')
    # plt.plot(X[:, 1], X.dot(theta)[:, 0])
    # plt.show()

import matplotlib.pyplot as plt

from src.functions import read_dataset_2, read_dataset_1, gradient_descent, predict

if __name__ == '__main__':
    X, y, theta, alpha, num_iters = read_dataset_2()
    theta, j_history = gradient_descent(X, y, theta, alpha, num_iters)

    # Plot cost function result history
    plt.subplot(1, 2, 1)
    plt.plot(j_history)
    plt.title('learning rate = %2f' % alpha)
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost J')

    # Plot regression line alongside scatter plot of data points
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 1], y, c='#FF0000', marker='x')
    plt.plot(X[:, 1], X.dot(theta)[:, 0])

    plt.show()

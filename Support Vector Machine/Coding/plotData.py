import matplotlib.pyplot as plt
import numpy as np

def plot_data(X, y):
    """Plot 2D data points. Positive examples (y == 1) use '+' marker,
    negative examples use 'o'.

    Args:
        X: array-like, shape (m, 2)
        y: array-like, shape (m,)
    """

    plt.figure()

    X = np.asarray(X)
    y = np.asarray(y).flatten()

    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError("X must be a 2D array with at least 2 columns")

    pos = y == 1
    neg = ~pos

    # Positive examples
    if np.any(pos):
        plt.scatter(X[pos, 0], X[pos, 1], marker='+', c='r', label='Positive')
    # Negative examples
    if np.any(neg):
        plt.scatter(X[neg, 0], X[neg, 1], marker='o', c='k', label='Negative')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)



import numpy as np

def expectedParameter(X, Y):
    """

    :param X: Matrix of features
    :param Y: Set of Label
    :return Best Theta parameter evaluated
    """
    theta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))
    return theta




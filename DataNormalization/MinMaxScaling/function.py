import numpy as np

def MinMaxScaling(X):
    """
    Return normalized design matrix with value inside [0, 1] range
    :param X: Design Matrix
    :return: Desig matrix normalized
    """

    max = np.max(X, axis=0)
    min = np.min(X, axis=0)
    return np.divide(np.subtract(X, min), np.subtract(max, min))
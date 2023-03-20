import numpy as np

def MeanNormalization(X):
    """
    Return normalized design matrix with value inside [-1, 1] range and average value 0
    :param X: Design Matrix
    :return: Desig matrix normalized
    """

    max = np.max(X, axis=0)
    min = np.min(X, axis=0)
    medium_value_arr = X.mean(axis=0)
    return np.divide(np.subtract(X, medium_value_arr), np.subtract(max, min))
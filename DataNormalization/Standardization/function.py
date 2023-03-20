import numpy as np

def Standardization(X):
    """
    Return normalized design matrix with 1 as average value and 0 as standard deviation
    :param X: Design Matrix
    :return: Desig matrix normalized
    """

    s_d_arr = np.std(X, axis=0)
    medium_value_arr = X.mean(axis=0)
    return np.divide(np.subtract(X, medium_value_arr), s_d_arr)

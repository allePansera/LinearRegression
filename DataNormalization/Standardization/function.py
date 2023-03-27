import numpy as np

def Standardization(X):
    """
    Return normalized design matrix with 1 as average value and 0 as standard deviation
    :param X: Design Matrix
    :return: Desig matrix normalized
    """

    s_d_arr = np.std(X, axis=0)
    s_d_arr = [val if val > 0 else 1 for val in s_d_arr]
    medium_value_arr = X.mean(axis=0)
    numerator = np.subtract(X, medium_value_arr)
    return np.divide(numerator, s_d_arr)

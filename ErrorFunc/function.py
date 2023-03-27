import numpy as np
from HypothesisFunc.function import hyp


def error(X, Y, theta):
    """
    error function is defined as least squares
    :param X: is features matrix array, rows identify samples n. and columns identify feature
    :param Y: is the expected target values (set of values)
    :param theta: is a parameter set used ot evaluate my prediction
    :return: error values between prediction
    """
    h = hyp(X, theta)
    loss = np.sum((h - Y) ** 2) / 2
    return loss

def error_V2(X, Y, theta, lamb=1):
    """
    error function is defined as least squares but with Lasso Regression
    :param X: is features matrix array, rows identify samples n. and columns identify feature
    :param Y: is the expected target values (set of values)
    :param theta: is a parameter set used ot evaluate my prediction
    :return: error values between prediction
    """
    # h = hyp(X, theta)
    first_sum = 0
    for index, el in enumerate(Y):
        inner_first_sum = 0
        for index_inner, el_inner in enumerate(theta):
            inner_first_sum += np.sum(X[index]*theta)
        first_sum += (Y[index] - inner_first_sum)**2

    first_sum *= 0.5

    second_sum = 0.5 * lamb * np.sum(abs(theta))
    loss = first_sum + second_sum
    return loss
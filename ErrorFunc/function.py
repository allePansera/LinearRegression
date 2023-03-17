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

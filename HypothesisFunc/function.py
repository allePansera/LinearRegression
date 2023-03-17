import numpy as np

def hyp(X, theta):
    """
    hyp function is supposed to be written under 'intercept form'
    :param X:  is features matrix array, rows identify samples n. and columns identify feature
    :param theta: is a parameter set used ot evaluate my prediction
    :return: y: prediction value
    """
    return np.dot(X, theta)

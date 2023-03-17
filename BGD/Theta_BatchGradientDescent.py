from HypothesisFunc.function import hyp
from ErrorFunc.function import error
import numpy as np


def expectedParameter(X, Y, alpha, max_iterations=10000, best_solution=0.0001):
    """

    :param X: Matrix of features
    :param Y: Set of Label
    :param alpha: hyperparameter
    :param max_iterations: max number of iterations allowed finding theta
    :param best_solution: stopping threshold
    :return (Best Theta parameter evaluated, iterationNumber, deltaError)
    """
    # getting X matrix dimension
    n, d = X.shape
    # theta has a random values
    theta = np.random.randn(d)
    current_error, prev_error = error(X, Y, theta), 0

    for iterationNumber in range(max_iterations):

        if iterationNumber != 0 and np.abs(current_error-prev_error) <= best_solution:
            # best theta solution is found then returned with current iteration number
            return theta, iterationNumber, np.abs(current_error-prev_error)

        else:
            # update current theta value with gradient method
            h = hyp(X, theta)
            gradient = np.dot(Y - h, X)
            theta += (alpha * gradient)
            # evaluate again loss function
            prev_error = current_error
            current_error = error(X, Y, theta)


    return theta, iterationNumber, np.abs(current_error-prev_error)




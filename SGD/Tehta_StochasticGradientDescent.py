from HypothesisFunc.function import hyp
from ErrorFunc.function import error
import numpy as np

def createMiniBatch(X, batch_size):
    """

    :param X: Features Matrix or Label
    :param batch_size: n. di elementi da prendere in un sottoinsieme
    :return: Reduced Matrix
    """
    n = X.shape[0]
    perm = np.random.permutation(n)
    # extract 'batch_size'th element from permutation
    mini_batch_indexes = perm[:batch_size]
    # the following instruction extract only specific row from X matrix
    return X[mini_batch_indexes]

def expectedParameter(X, Y, alpha, batch_size, max_iterations=10000, best_solution=0.0001):
    """

    :param X: Matrix of features
    :param Y: Set of Label
    :param alpha: hyperparameter
    :param batch_size: n. of elements inside Batch of X to analyze
    :param max_iterations: max number of iterations allowed finding theta
    :param best_solution: stopping threshold
    :return (Best Theta parameter evaluated, iterationNumber, deltaError)
    """
    # getting X matrix dimension
    n, d = X.shape
    # generating the first mini match -> each time i re-calculate it
    x_mini_batch, y_mini_batch = createMiniBatch(X, batch_size), createMiniBatch(Y, batch_size)
    # theta has a random values
    theta = np.random.randn(d)
    current_error, prev_error = error(x_mini_batch, y_mini_batch, theta), 0

    for iterationNumber in range(max_iterations):

        if iterationNumber != 0 and np.abs(current_error-prev_error) <= best_solution:
            # best theta solution is found then returned with current iteration number
            return theta, iterationNumber, np.abs(current_error-prev_error)

        else:
            # update current theta value with gradient method
            h = hyp(x_mini_batch, theta)
            gradient = np.dot(y_mini_batch - h, x_mini_batch)
            theta += (alpha * gradient)
            # evaluate again loss function
            prev_error = current_error
            current_error = error(x_mini_batch, y_mini_batch, theta)
            # re-evaluating the mini batch
            x_mini_batch, y_mini_batch = createMiniBatch(X, batch_size), createMiniBatch(Y, batch_size)

    return theta, iterationNumber, np.abs(current_error-prev_error)




import numpy as np
from Error.errorMeasure import *
from SGD.Tehta_StochasticGradientDescent import expectedParameter

feature_selection_combination = []
def GenerateBinaryPermutation(n, arr, i):
    if i == n:
        feature_selection_combination.append(arr.copy())
        return

        # First assign "0" at ith position
        # and try for all other permutations
        # for remaining positions
    arr[i] = 0
    GenerateBinaryPermutation(n, arr, i + 1)

    # And then assign "1" at ith position
    # and try for all other permutations
    # for remaining positions
    arr[i] = 1
    GenerateBinaryPermutation(n, arr, i + 1)

def findDiff(array1, array2):
    """

    :param array1: first array to compare
    :param array2: second array to compare
    :return: counter of difference between 2 arrays
    """
    counter = 0
    for el in range(len(array1)):
        if array2[el] != array1[el]:
            counter += 1
    return counter

def WrapperMethod_FW(X, Y, validationSetPercentage=0.2):
    """

    :param X: Design Matrix
    :param Y: Label values Matrix
    :param validationSetPercentage: percentage of the Design matrix to consider as validation set
    :return: the best parameter index
    """
    # splitting validation and testing matrix & labels
    n, d = X.shape
    n_validation = int(n * validationSetPercentage)
    # create permutation in order to separate the original design matrix
    perm = np.random.permutation(n)
    X_validation, Y_validation = X[perm[:n_validation]], Y[perm[:n_validation]]
    X_training, Y_training = X[perm[n_validation:]], Y[perm[n_validation:]]
    # generate all possible permutation of binary selection
    GenerateBinaryPermutation(d, [False] * d, 0)
    # for each cardinality (index) there is the score
    feature_selection_score = []
    # for each cardinality (index) there is a binary combination where 1 means that the element is selected
    feature_selection_best_combination = []
    # for each cardinality find the best combination
    for index in range(d):
        # current score array and possible combination
        current_score = {}
        current_combination = {}
        # cardinality is index + 1
        cardinality = index + 1
        # find all suitable combination with current cardinality
        possible_combination = [comb for comb in feature_selection_combination if comb.count(1) == cardinality]
        # skip if there are no possible combination with the current cardinality
        if len(possible_combination) < 1:
            continue
        # skip if there are no combination with previous prefix

        if len(feature_selection_best_combination) > 0:
            possible_combination = [comb
                                    for comb in possible_combination
                                        if findDiff(feature_selection_best_combination[-1], comb) < cardinality]

        # for each combination get the best parameter, suppose with SGD
        for index, comb in enumerate(possible_combination):
            bool_comb = [bool(item) for item in comb]
            theta = expectedParameter(X_training[:, bool_comb], Y_training, 1e-9, batch_size=5)[0]
            current_score[index] = calcMSE(np.dot(X_validation[:, bool_comb], theta), Y_validation, X_validation)
            current_combination[index] = comb

        # get the combination with lowe error
        best_index_combination = min(current_score, key=current_score.get)
        feature_selection_best_combination.append(current_combination[best_index_combination])
        feature_selection_score.append(current_score[best_index_combination])


    # inside feature selection there is the history of features picking
    # if required is possible to insert a shoulder in order to reduce the number of selection
    return feature_selection_best_combination




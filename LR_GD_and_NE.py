import numpy as np
from time import time
from matplotlib import pyplot as plt


# -----------------------------------------------------
# Hypothesis function
# -----------------------------------------------------
def hyp(X, theta):
    '''
    :param X: Feature matrix
    :param theta: Linear regression weights
    :return: the value of the hypothesis function for each row of X
    '''
    # supponiamo h0 = theta0 + theta1*x1 + theta2*x2
    return np.sum(X*theta)


# -----------------------------------------------------
# loss function
# -----------------------------------------------------
def loss(X, y, theta):
    '''
    :param y: target values
    :param X: Feature matrix
    :param theta: Linear regression weights
    :return: The loss function for the given input data
    '''
    # cacolo la H 
    h = hyp(X, theta)
    # calcolo la mia loss
    loss = np.sum((h - y)**2) / 2
    return loss


# -----------------------------------------------------
# Linear regression solver - gradient descent -> BATCH
# -----------------------------------------------------
def linear_regression_fit_GD(X, y, alpha, eps=0.0001):
    '''
    :param y: target values
    :param X: Feature matrix
    :param alpha: learning rate
    :param eps: stopping threshold
    :return: The updated regression weights
    '''

    print("\n Batch Gradient Descent ")
    max_iter = 10000 # SETTO IL NUMERO MASSIMO DI RICORSIONI PER CERCARE I PARAMETRI OTTIMIZZATI
    n, d = X.shape # PRENDO IL NUMERO DI  SAMPLE E DI FEATURE DELLA MATRICE DI TRAINING
    theta = np.random.randn(d) # CALCOLO IL PARAMTRO A CASSO
    Jold = np.inf # SETTO UN VALORE DI ERRORE MASSIMO
    Jnew = loss(X, y, theta) # CALCLO UN VALORE DI ERRORE PER L'ITERAZIONE 'ZERO-esima'
    iteration = 0 # CONTEGGIA LE ITERAZIONI

    # SCORRO FINO A QUANDO L'ERRORE NON E' MINORE DI esp E SE IL NUMERO DELLE ITERAZIONI E' MINORE DI max_iter
    while np.abs(Jnew - Jold) >= eps and iteration < max_iter:
        # SONO ENTRATO PERCHè HO ERRORE NON ACCETTABILE MA HO FINITO LE ITERAZIONI
        if np.mod(iteration, 1000) == 0:
            print("iter ", iteration, "-> J: ", Jnew)
        iteration += 1
        # CALCOLO IL GRADIENTE AGGIORNATO

        # CALCOLO IL PARAMETRO AGGIORNATO
        theta += ( alpha * np.dot( y -hyp(X, theta), X) ) 
        Jold = Jnew
        # CALCOLO LA NUOVA LOSS FUNCTION
        Jnew = loss(X, y, theta)
    print("Optimization stopped, num iters = ", iteration)
    return theta


# -----------------------------------------------------
# Linear regression solver - gradient descent
# -----------------------------------------------------
def linear_regression_fit_SGD(X, y, alpha, m, eps=0.0001):
    '''
    :param y: target values
    :param X: Feature matrix
    :param alpha: learning rate
    :param eps: stopping threshold
    :param m: mini-batch size
    :return: The updated regression weights
    '''
    
    # Suggerimento: usare m = 1 (mini-batch da un solo elemento)
    
    print("\n Stocastich Gradient Descent ")
    max_iter = 10000 # SETTO IL NUMERO MASSIMO DI RICORSIONI PER CERCARE I PARAMETRI OTTIMIZZATI
    n, d = X.shape # PRENDO IL NUMERO DI  SAMPLE E DI FEATURE DELLA MATRICE DI TRAINING
    theta = np.random.randn(d) # CALCOLO IL PARAMTRO A CASSO
    Jold = np.inf # SETTO UN VALORE DI ERRORE MASSIMO
    Jnew = loss(X, y, theta) # CALCLO UN VALORE DI ERRORE PER L'ITERAZIONE 'ZERO-esima'
    iteration = 0 # CONTEGGIA LE ITERAZIONI

    # SCORRO FINO A QUANDO L'ERRORE NON E' MINORE DI esp E SE IL NUMERO DELLE ITERAZIONI E' MINORE DI max_iter
    while np.abs(Jnew - Jold) >= eps and iteration < max_iter:
        # SONO ENTRATO PERCHè HO ERRORE NON ACCETTABILE MA HO FINITO LE ITERAZIONI
        if np.mod(iteration, 1000) == 0:
            print("iter ", iteration, "-> J: ", Jnew)
        iteration += 1
        # CALCOLO IL GRADIENTE AGGIORNATO

        # CALCOLO IL PARAMETRO AGGIORNATO
        theta += ( alpha * np.dot( (y[0] -hyp(X[0], theta)) , X[0] ) )
        Jold = Jnew
        # CALCOLO LA NUOVA LOSS FUNCTION
        Jnew = loss(X, y, theta)
    print("Optimization stopped, num iters = ", iteration)

    return theta


# -----------------------------------------------------
# Linear regression solver - Normal Equation
# -----------------------------------------------------
def linear_regression_fit_NE(X, y):
    '''
    :param y: target values
    :param X: Feature matrix
    :return: Linear regression weights
    '''
    theta_hat = np.dot(
        np.linalg.inv(
            np.dot(X.T, X)
        ), np.dot(X.T, y)
    )
    return theta_hat



'''
Compare NE, GD and SGD.
'''
# -----------------------------------------------------
# Hands On
# -----------------------------------------------------
if __name__ == "__main__":
    features_filename = '../TrainingML_LR/datasets/features.dat'
    targets_filename = '../TrainingML_LR/datasets/targets.dat'
    X = np.loadtxt(features_filename)
    y = np.loadtxt(targets_filename)
    
    ### EDA
    print(X.shape)
    print(y.shape)

    plt.figure(r'y vs $x_1$')
    plt.title(r'y vs $x_1$')
    plt.scatter(X[:, 0], y)
    plt.figure(r'y vs $x_2$')
    plt.title(r'y vs $x_2$')
    plt.scatter(X[:, 1], y)
    plt.show()

    ### Intercept term
    x_0 = np.ones((len(X), 1))
    X2 = np.append(x_0, X, axis=1)

    ### Compute theta with: Gradient Descent, Stochastic Gradient Descent, Normal Equation
    start_NE = time()
    theta_NE = linear_regression_fit_NE(X2, y)
    stop_NE = time()
    
    alpha= 1e-9
    start_GD = time()
    theta_GD = linear_regression_fit_GD(X2, y, alpha)
    stop_GD = time()
    
    start_SGD = time()
    theta_SGD = linear_regression_fit_SGD(X2, y, alpha, 5)
    stop_SGD = time()

    print("theta from Normal Equation: ", theta_NE, " in ", (stop_NE - start_NE) * 1000, ' ms')
    print("theta from Gradient Descent: ", theta_GD, " in ", (stop_GD - start_GD) * 1000, ' ms')
    print("theta from Stochastic Gradient Descent: ", theta_SGD, " in ", (stop_SGD - start_SGD) * 1000, ' ms')


    ### Show the regressed lines
    x1_pts= X[:, 0]
    y_pts_NE= theta_NE[1] * x1_pts + theta_NE[0]
    y_pts_GD= theta_GD[1] * x1_pts + theta_GD[0]
    y_pts_SGD= theta_SGD[1] * x1_pts + theta_SGD[0]

    plt.figure('Hypotheses')
    plt.title('Hypotheses ($x_1$ only)')
    plt.scatter(x1_pts, y)
    plt.plot(x1_pts, y_pts_NE, 'r', label='NE')
    plt.plot(x1_pts, y_pts_GD, 'g', label='GD')
    plt.plot(x1_pts, y_pts_SGD, 'b', label='SGD')
    plt.legend()
    plt.show()


    # y_hat_NE = hyp(X2, theta_NE)
    # y_hat_GD = hyp(X2, theta_GD)
    # y_hat_SGD = hyp(X2, theta_SGD)
    # plt.figure('Predictions_vs_Ground_Truth')
    # plt.title('Training set: Predictions vs Ground Truth')
    # plt.plot(y_hat_GD, label='$\hat{y}_{GD}$')
    # #plt.plot(y_hat_SGD, label='$\hat{y}_{SGD}$')
    # plt.plot(y_hat_NE, label='$\hat{y}_{NE}$')
    # plt.plot(y, label='y')
    # plt.legend()
    # plt.show()



    ###
    # Given x_test, compute (and show) its prediction w.r.t., e.g., theta_GD
    x_test= np.array([3700, 4.5]) # this is the input value
    
    # TODO


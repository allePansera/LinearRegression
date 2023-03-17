import numpy as np
from matplotlib import pyplot as plt
from time import time
from NE.Theta_NaturalEquation import expectedParameter as expectedParameterNE
from BGD.Theta_BatchGradientDescent import expectedParameter as expectedParameterBGD
from SGD.Tehta_StochasticGradientDescent import expectedParameter as expectedParameterSBGD

# Read X Matrix and Y Features
features_filename = './dataset/features.dat'
targets_filename = './dataset/targets.dat'
X = np.loadtxt(features_filename)
Y = np.loadtxt(targets_filename)
# Generate Intercept Form considering feature Values
x_0 = np.ones((len(X), 1))
# Intercept Form used with hyp function
X_Intercept = np.append(x_0, X, axis=1)
# variable with trained theta param
theta = {}
# 1st: Test Natural Equation
start = time()
theta["N.E."] = expectedParameterNE(X=X, Y=Y)
end = time()
print(f'\n\nNatural Equation..\nTheta:{theta}\nTempo: {round((end-start),2)} sec')
# 2nd: Test Batch Gradient Descent
start = time()
theta["BGD"], iterNum, error = expectedParameterBGD(X=X_Intercept, Y=Y, alpha=1e-9)
end = time()
print(f'\n\nBatch GD...\nTheta:{theta}\tIterazioni:{iterNum}\tErrore: {error}\nTempo: {round((end-start),2)} sec')
# 3rd: Test Stochastic Gradient Descent
start = time()
theta["SGD"], iterNum, error = expectedParameterSBGD(X=X_Intercept, Y=Y, batch_size=5, alpha=1e-9)
end = time()
print(f'\n\nStochastic GD...\nTheta:{theta}\tIterazioni:{iterNum}\tErrore: {error}\nTempo: {round((end-start),2)} sec')

# Compare, inside a Plot, the difference between prediction and label

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:, [0]], X[:, [1]], Y, label="Ideal")
for label, color in zip(["N.E.", "BGD", "SGD"], ["r", "g", "b"]):
    targets_pts = np.sum((theta[label].T) * (X if len(theta[label])==2 else X_Intercept), axis=1)
    # ax.scatter3D(X[:, [0]], X[:, [1]], targets_pts, color=color, label=label)
    ax.scatter([el[0] for el in X[:, [0]]],
                 [el[0] for el in X[:, [1]]],
                 targets_pts,
                 label=label, color=color)

ax.set_xlabel('Ft. 1')
ax.set_ylabel('Ft. 2')
ax.set_zlabel('Target')
plt.legend()
plt.show()

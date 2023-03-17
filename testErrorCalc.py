import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from Error.errorMeasure import *

# Load housing dataset
features_filename = './dataset/features.dat'
targets_filename = './dataset/targets.dat'
X = np.loadtxt(features_filename)
y = np.loadtxt(targets_filename)
# add extra 1 col horizontally to X original Matrix
X = np.hstack([np.ones((X.shape[0], 1)), X])
# Suffle original dataset
np.random.shuffle(X)
n = X.shape[0]
#Second, split X in training and testing sets
split = 0.8
tr_size = int(n * split)
X_tr = X[0:tr_size]
y_tr = y[0:tr_size]
X_ts = X[tr_size:]
y_ts = y[tr_size:]


# Use sklearn to instantiate a Linear Regression algorithm
lr = LinearRegression()
# train regressor
lr.fit(X_tr, y_tr)
# test regressor
y_ts_hat = lr.predict(X_ts)

print('Evaluation using the sklearn prediction')
print("Error on train (MSE) = ", calcMSE(y_tr, lr.predict(X_tr), X))
print("Error on test (MSE) = ", calcMSE(y_ts, y_ts_hat, X))
print("Error on train (MAE) = ", calcMAE(y_tr, lr.predict(X_tr), X))
print("Error on test (MAE) = ", calcMAE(y_ts, y_ts_hat, X))




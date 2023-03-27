import time
import numpy as np
from FeatureSelection.WrapperMethod import WrapperMethod_FW as Wm_FW
from DataNormalization.Standardization.function import Standardization

# Read X Matrix and Y Features
features_filename = './dataset/features.dat'
targets_filename = './dataset/targets.dat'
X = np.loadtxt(features_filename)
Y = np.loadtxt(targets_filename)
# Finding the most relevant feature
x_0 = np.ones((len(X), 1))
X_Intercept = np.append(x_0, X, axis=1)
# standardize the datasets
start = time.time()
X = Standardization(X_Intercept)
features_picking_history = Wm_FW(X, Y)
end = time.time()
print(f'\n\nFeature selection Wrapper Methods FW...\n{features_picking_history}\nTempo: {round((end-start),2)} sec')



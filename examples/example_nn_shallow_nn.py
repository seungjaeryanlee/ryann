#!/usr/bin/env python3
"""
This file showcases what ryann.nn.shallow_nn() does.
"""
import numpy as np
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.abspath('..'))
from ryann import nn


# 0) Synthesize Data
m = 200
X = np.random.randn(2, m)
Y = np.zeros(shape=(1,m))
noise = 0.5
for i, x in enumerate(X.T):
    Y[0][i] = int(x[0] + x[1] < 1 + np.random.randn() * noise)
# Y = [ int(x[0] + x[1] < 1 + np.random.randn() * noise) for x in X.T]

# 1) Plot Data
plt.scatter(X[0, :], X[1, :], c=Y.reshape(m), cmap='coolwarm')
plt.show(block=False)
plt.pause(0.01)

# 2) Train
parameters, costs = nn.shallow_nn(X, Y, n_h=3, num_iter=10000)

# 3) Compute and Plot Decision Boundary

# 4) Finish
plt.show() # Continue showing plot after script finishes
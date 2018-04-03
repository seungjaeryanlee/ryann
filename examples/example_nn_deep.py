#!/usr/bin/env python3
"""
This file showcases the ryann.nn.deep module.
"""
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from ryann.nn import deep


# 0) Synthesize Data
m = 200
X = np.random.randn(2, m)
Y = np.zeros(shape=(1, m))
noise = 1
for i, x in enumerate(X.T):
    Y[0][i] = int(x[0] ** 2 + x[1] ** 2 < 1 + noise * np.random.choice([0, 1]))

# 1) Train
parameters, costs = deep.train(X, Y, layers=[2, 4, 4, 4, 1], num_iter=10000, learning_rate=0.01,
                               regularization=False)

# 2) Compute and Plot Decision Boundary
x1_pts = np.arange(-5, 5, 0.01)
x2_pts = np.arange(-5, 5, 0.01)
x1_grid, x2_grid = np.meshgrid(x1_pts, x2_pts)
X_grid = np.concatenate((x1_grid.reshape(1, -1), x2_grid.reshape(1, -1)))
Z = deep.predict(parameters, ['relu', 'relu', 'relu', 'relu', 'sigmoid'], X_grid)
Z = Z.reshape((len(x1_pts), len(x2_pts)))

plt.contourf(x1_grid, x2_grid, Z, cmap='coolwarm')
plt.scatter(X[0, :], X[1, :], c=Y.reshape(m), cmap='coolwarm')
plt.show()

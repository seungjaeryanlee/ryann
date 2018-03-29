# pylint: disable=no-self-use, too-few-public-methods, invalid-name
"""
Tests the function nn.shallow_nn().
"""
import numpy as np
from ryann import nn


def test_nn_shallow_nn_nonincreasing_cost():
    """
    Check that the cost doesn't increase for every iteration of gradient descent.
    """
    X = np.array([[1, 2, 3]])
    Y = np.array([[1, 1, 0]])
    _, costs = nn.shallow_nn(X, Y, n_h=1, num_iter=10000)

    print('Costs in descending order : ' + str(costs))
    assert sorted(costs, reverse=True)

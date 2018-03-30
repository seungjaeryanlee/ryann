# pylint: disable=no-self-use, too-few-public-methods, invalid-name
"""
Tests the function nn.shallow.train().
"""
import numpy as np
from ryann.nn import shallow


def test_nn_shallow_train_nonincreasing_cost():
    """
    Check that the cost doesn't increase for every iteration of gradient descent.
    """
    X = np.array([[1, 2, 3]])
    Y = np.array([[1, 1, 0]])
    _, costs = shallow.train(X, Y, n_h=1, num_iter=10000)

    print('Costs in descending order : ' + str(costs))
    assert sorted(costs, reverse=True)

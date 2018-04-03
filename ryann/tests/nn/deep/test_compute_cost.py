# pylint: disable=no-self-use, too-few-public-methods, invalid-name, protected-access, duplicate-code
"""
Tests the function nn.deep._compute_cost().
"""
import numpy as np
from numpy.testing import assert_almost_equal
from ryann.nn import deep


def test_nn_deep_compute_cost_close():
    """
    Check that nn.deep._compute_cost() returns close to zero if Y_computed is close to Y.
    """
    m = 10
    epsilon = 10**-8

    Y = np.random.randint(0, 2, size=(1, m))
    Y_computed = np.abs(Y - epsilon)
    cost = deep._compute_cost(Y_computed, Y)

    print('Y : ' + str(Y))
    print('Y_computed : ' + str(Y_computed))
    print('Cost : ' + str(cost))

    assert_almost_equal(cost, 0)


def test_nn_deep_compute_cost_half():
    """
    Check that nn.deep._compute_cost() returns - log(0.5) if Y_computed is all 0.5.
    """
    m = 10

    Y = np.random.randint(0, 2, size=(1, m))
    Y_computed = np.full((1, m), 0.5)
    cost = deep._compute_cost(Y_computed, Y)

    print('Y : ' + str(Y))
    print('Y_computed : ' + str(Y_computed))
    print('Cost : ' + str(cost))

    assert_almost_equal(cost, -np.log(0.5))

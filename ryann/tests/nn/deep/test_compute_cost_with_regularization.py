# pylint: disable=no-self-use, too-few-public-methods, invalid-name, protected-access, duplicate-code
"""
Tests the function nn.deep._compute_cost_with_regularization().
"""
import numpy as np
from numpy.testing import assert_almost_equal
from ryann.nn import deep


def test_nn_deep_compute_cost_with_regularization_no_lambd():
    """
    Check that nn.deep._compute_cost_with_regularization() returns same result as _compute_cost
    when lambd == 0.
    """
    m = 10
    epsilon = 10**-8

    Y = np.random.randint(0, 2, size=(1, m))
    Y_computed = np.abs(Y - epsilon)
    parameters = {'W1': np.random.randn(1, 3), 'b1': np.random.randn(1, m)}
    cost = deep._compute_cost(Y_computed, Y)
    regularized_cost = deep._compute_cost_with_regularization(Y_computed, Y, parameters, 0)

    print('Y                : ' + str(Y))
    print('Y_computed       : ' + str(Y_computed))
    print('Cost             : ' + str(cost))
    print('Regularized cost : ' + str(cost))

    assert_almost_equal(cost, regularized_cost)


def test_nn_deep_compute_cost_with_regularization_higher_than_without():
    """
    Check that nn.deep._compute_cost_with_regularization() returns larger cost than as _compute_cost
    when when lambd > 0.
    """
    m = 10
    epsilon = 10**-8

    Y = np.random.randint(0, 2, size=(1, m))
    Y_computed = np.abs(Y - epsilon)
    parameters = {'W1': np.random.randn(1, 3), 'b1': np.random.randn(1, m)}
    cost = deep._compute_cost(Y_computed, Y)
    regularized_cost = deep._compute_cost_with_regularization(Y_computed, Y, parameters, 0.001)

    print('Y                : ' + str(Y))
    print('Y_computed       : ' + str(Y_computed))
    print('Cost             : ' + str(cost))
    print('Regularized cost : ' + str(cost))

    assert cost < regularized_cost

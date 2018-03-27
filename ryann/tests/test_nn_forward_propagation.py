# pylint: disable=no-self-use, too-few-public-methods, invalid-name
"""
Tests the function nn._forward_propagation().
"""
import numpy as np
from numpy.testing import assert_almost_equal
from ryann import nn


def test_nn_forward_propagation_dimension():
    """
    Checks the dimensions of output values of nn._forward_propagation().
    """
    n_x = np.random.randint(10)
    n_h = np.random.randint(10)
    n_y = np.random.randint(10)
    m = np.random.randint(100)
    params = nn._initialize_parameters(n_x, n_h, n_y)
    X = np.zeros((n_x, m))
    Y_computed, cache = nn._forward_propagation(X, params)

    assert Y_computed.shape == (n_y, m)
    assert cache['Z1'].shape == (n_h, m)
    assert cache['A1'].shape == (n_h, m)
    assert cache['Z2'].shape == (n_y, m)
    assert cache['A2'].shape == (n_y, m)


def test_nn_forward_propagation_precomputed():
    """
    Tests nn._forward_propagation() with weight matrices and bias vectors with precomputed results.
    """
    pass
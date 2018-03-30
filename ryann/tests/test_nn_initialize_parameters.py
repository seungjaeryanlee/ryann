# pylint: disable=no-self-use, too-few-public-methods, invalid-name, protected-access
"""
Tests the function nn._initialize_parameters().
"""
import numpy as np
from ryann.nn import shallow

def test_nn_initialize_parameters_dimensions():
    """
    Checks the dimensions of output parameters of nn._initialize_parameters().
    """
    n_x = np.random.randint(10) + 1
    n_h = np.random.randint(10) + 1
    n_y = np.random.randint(10) + 1
    params = shallow._initialize_parameters(n_x, n_h, n_y)

    assert params['W1'].shape == (n_h, n_x)
    assert params['b1'].shape == (n_h, 1)
    assert params['W2'].shape == (n_y, n_h)
    assert params['b2'].shape == (n_y, 1)


def test_nn_initialize_parameters_weight_nonzero():
    """
    Tests that nn._initialize_parameters() does not set the weight matrices to zero matrices.
    """
    n_x = np.random.randint(10) + 1
    n_h = np.random.randint(10) + 1
    n_y = np.random.randint(10) + 1
    params = shallow._initialize_parameters(n_x, n_h, n_y)

    assert np.count_nonzero(params['W1']) > 0
    assert np.count_nonzero(params['W2']) > 0


def test_nn_initialize_parameters_bias_zero():
    """
    Tests that nn._initialize_parameters() sets the bias vectors to zero vectors.
    """
    n_x = np.random.randint(10) + 1
    n_h = np.random.randint(10) + 1
    n_y = np.random.randint(10) + 1
    params = shallow._initialize_parameters(n_x, n_h, n_y)


    assert np.count_nonzero(params['b1']) == 0
    assert np.count_nonzero(params['b2']) == 0

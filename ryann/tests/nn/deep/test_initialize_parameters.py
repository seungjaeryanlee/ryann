# pylint: disable=no-self-use, too-few-public-methods, invalid-name, protected-access
"""
Tests the function nn.deep._initialize_parameters().
"""
import numpy as np
from ryann.nn import deep

def test_nn_deep_initialize_parameters_dimensions():
    """
    Checks the dimensions of output parameters of nn.shallow._initialize_parameters().
    """
    layer_dims = [5, 4, 3, 2, 1]
    params = deep._initialize_parameters(layer_dims)

    for l in range(1, len(layer_dims)):
        assert params[l][0].shape == (layer_dims[l], layer_dims[l - 1])
        assert params[l][1].shape == (layer_dims[l], 1)


def test_nn_deep_initialize_parameters_weight_nonzero():
    """
    Tests that nn.shallow._initialize_parameters() does not set the weight matrices to zero
    matrices.
    """
    layer_dims = [5, 4, 3, 2, 1]
    params = deep._initialize_parameters(layer_dims)

    for l in range(1, len(layer_dims)):
        assert np.count_nonzero(params[l][0]) > 0


def test_nn_deep_initialize_parameters_bias_zero():
    """
    Tests that nn.shallow._initialize_parameters() sets the bias vectors to zero vectors.
    """
    layer_dims = [5, 4, 3, 2, 1]
    params = deep._initialize_parameters(layer_dims)

    for l in range(1, len(layer_dims)):
        assert np.count_nonzero(params[l][1]) == 0

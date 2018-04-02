# pylint: disable=no-self-use, too-few-public-methods, invalid-name, protected-access
"""
Tests the function nn.deep._forward_propagation().
"""
import numpy as np
from ryann.nn import deep


def test_nn_deep_forward_propagation_dimension():
    """
    Checks the dimensions of output values of nn.deep._forward_propagation().
    """
    layer_dims = [5, 4, 3, 2, 1]
    L = len(layer_dims)
    m = np.random.randint(10) + 1
    params = deep._initialize_parameters(layer_dims)
    X = np.zeros((layer_dims[0], m))
    Y_computed, cache = deep._forward_propagation(X, params)

    assert Y_computed.shape == (layer_dims[-1], m)

    assert len(cache) == 3 * L
    for l in range(1, L):
        assert cache[l][0].shape == (layer_dims[l], m) # Z
        assert cache[l][1].shape == (layer_dims[l], m) # A


def test_nn_deep_forward_propagation_precomputed():
    """
    Tests nn.deep._forward_propagation() with weight matrices and bias vectors with precomputed
    results.
    """
    pass

# pylint: disable=no-self-use, too-few-public-methods, invalid-name, protected-access
"""
Tests the function nn.deep._forward_propagation().
"""
import numpy as np
from ryann.nn import deep


def test_nn_deep_forward_propagation_yhat_shape():
    """
    Checks the shape of the computed Y value of nn.deep._forward_propagation().
    """
    layer_dims = np.random.randint(1, 10, 5)
    m = np.random.randint(10) + 1
    params = deep._initialize_parameters(layer_dims)
    X = np.zeros((layer_dims[0], m))
    Y_computed, _ = deep._forward_propagation(X, params)

    assert Y_computed.shape == (layer_dims[-1], m)


def test_nn_deep_forward_propagation_cache_length():
    """
    Checks the length of the output cache of nn.deep._forward_propagation().
    """
    layer_dims = np.random.randint(1, 10, 5)
    L = len(layer_dims) - 1
    m = np.random.randint(10) + 1
    params = deep._initialize_parameters(layer_dims)
    X = np.zeros((layer_dims[0], m))
    _, cache = deep._forward_propagation(X, params)

    assert len(cache) == 3 * L + 1 # Z, A, W for each layer, extra A0


def test_nn_deep_forward_propagation_cache_dimension():
    """
    Checks the dimensions of the output cache of nn.deep._forward_propagation().
    """
    layer_dims = np.random.randint(1, 10, 5)
    L = len(layer_dims) - 1
    m = np.random.randint(10) + 1
    params = deep._initialize_parameters(layer_dims)
    X = np.zeros((layer_dims[0], m))
    _, cache = deep._forward_propagation(X, params)

    for l in range(1, L):
        assert cache['Z' + str(l)].shape == (layer_dims[l], m) # Z
        assert cache['A' + str(l)].shape == (layer_dims[l], m) # A
        assert cache['W' + str(l)].shape == params['W' + str(l)].shape # W


def test_nn_deep_forward_propagation_precomputed():
    """
    Tests nn.deep._forward_propagation() with weight matrices and bias vectors with precomputed
    results.
    """
    pass

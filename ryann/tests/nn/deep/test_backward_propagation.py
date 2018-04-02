# pylint: disable=no-self-use, too-few-public-methods, invalid-name, protected-access, too-many-locals
"""
Tests the function nn.deep._backward_propagation().
"""
import numpy as np
from numpy.testing import assert_almost_equal
from ryann.nn import deep


def test_nn_deep_backward_propagation_gradient_checking():
    """
    Test output of nn.deep._backward_propagation() with Gradient Checking. Gradient Checking is a
    method of manually computing derivatives and comparing it with the output.
    """
    layer_dims = np.random.randint(1, 10, 5)
    m = np.random.randint(1, 10)
    X = np.random.randn(layer_dims[0], m)
    Y = np.random.randint(0, 2, size=(layer_dims[-1], m))

    # 1) Compute gradients with nn.deep._backward_propagation()
    parameters = deep._initialize_parameters(layer_dims)
    Y_computed, cache = deep._forward_propagation(X, parameters)
    gradients = deep._backward_propagation(cache, Y_computed, Y)

    # 2) Compute gradients manually
    epsilon = 10**-8
    for key, parameter in parameters.items():
        for index, element in np.ndenumerate(parameter):
            # Get J(..., x + e, ...)
            parameter[index] = element + epsilon
            Y_computed, _ = deep._forward_propagation(X, parameters)
            cost_plus = deep._compute_cost(Y_computed, Y)

            # Get J(..., x - e, ...)
            parameter[index] = element - epsilon
            Y_computed, _ = deep._forward_propagation(X, parameters)
            cost_minus = deep._compute_cost(Y_computed, Y)

            # Estimate gradient
            estimated_gradient = (cost_plus - cost_minus) / (2 * epsilon)

            print('Parameter         : ' + key)
            print('index             : ' + str(index))
            print('Backpropagation   : ' + str(gradients['d' + str(key)][index]))
            print('Gradient Checking : ' + str(estimated_gradient))

            assert_almost_equal(estimated_gradient, gradients['d' + str(key)][index])

# pylint: disable=no-self-use, too-few-public-methods, invalid-name, protected-access, too-many-locals
"""
Tests the function nn._backward_propagation().
"""
import numpy as np
from numpy.testing import assert_almost_equal
from ryann.nn import shallow


def test_nn_backward_propagation_gradient_checking():
    """
    Test output of nn._backward_propagation() with Gradient Checking. Gradient Checking is a method
    of manually computing derivatives and comparing it with the output.
    """
    n_x = 3
    n_h = 3
    n_y = 1
    m = 1
    X = np.random.randn(n_x, m)
    Y = np.random.randint(0, 2, size=(n_y, m))

    # 1) Compute gradients with nn._backward_propagation()
    parameters = shallow._initialize_parameters(n_x, n_h, n_y)
    Y_computed, cache = shallow._forward_propagation(X, parameters)
    gradients = shallow._backward_propagation(parameters, cache, X, Y)

    # 2) Compute gradients manually
    epsilon = 10**-8
    for key, parameter in parameters.items():
        for index, element in np.ndenumerate(parameter):
            # Get J(..., x + e, ...)
            parameter[index] = element + epsilon
            Y_computed, _ = shallow._forward_propagation(X, parameters)
            cost_plus = shallow._compute_cost(Y_computed, Y)

            # Get J(..., x - e, ...)
            parameter[index] = element - epsilon
            Y_computed, _ = shallow._forward_propagation(X, parameters)
            cost_minus = shallow._compute_cost(Y_computed, Y)

            # Estimate gradient
            estimated_gradient = (cost_plus - cost_minus) / (2 * epsilon)

            print('Parameter         : ' + key)
            print('index             : ' + str(index))
            print('Backpropagation   : ' + str(gradients[key][index]))
            print('Gradient Checking : ' + str(estimated_gradient))

            assert_almost_equal(estimated_gradient, gradients[key][index])

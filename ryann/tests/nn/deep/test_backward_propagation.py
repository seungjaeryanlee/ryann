# pylint: disable=no-self-use, too-few-public-methods, invalid-name, protected-access, too-many-locals
"""
Tests the function nn.deep._backward_propagation().
"""
import numpy as np
from ryann.nn import deep


def test_nn_deep_backward_propagation_gradient_checking_sigmoid():
    """
    Test output of nn.deep._backward_propagation() with Gradient Checking for NN using sigmoid
    activation functions (except the output layer). Gradient Checking is a method of manually
    computing derivatives and comparing it with the output.
    """
    layers = []
    for _ in range(np.random.randint(3, 5)):
        layers.append({'units': np.random.randint(2, 5), 'activation': 'sigmoid'})
    layers.append({'units': 1, 'activation': 'sigmoid'})
    layer_dims, activations = deep._split_layer_dims_activations(layers)
    m = np.random.randint(1, 10)
    X = np.random.randn(layer_dims[0], m)
    Y = np.random.randint(0, 2, size=(layer_dims[-1], m))

    # 1) Compute gradients with nn.deep._backward_propagation()
    parameters = deep._initialize_parameters(layer_dims)
    Y_computed, cache = deep._forward_propagation(X, parameters, activations)
    gradients = deep._backward_propagation(cache, Y_computed, Y, activations)

    # 2) Compute gradients manually
    epsilon = 10**-8
    for key, parameter in parameters.items():

        if key[0] == 'W':
            continue
        for index, element in np.ndenumerate(parameter):
            # Get J(..., x + e, ...)
            parameter[index] = element + epsilon
            Y_computed, _ = deep._forward_propagation(X, parameters, activations)
            cost_plus = deep._compute_cost(Y_computed, Y)

            # Get J(..., x - e, ...)
            parameter[index] = element - epsilon
            Y_computed, _ = deep._forward_propagation(X, parameters, activations)
            cost_minus = deep._compute_cost(Y_computed, Y)

            # Estimate gradient
            estimated_gradient = (cost_plus - cost_minus) / (2 * epsilon)
            gradient = gradients['d' + str(key)][index]

            # Relative Error: http://cs231n.github.io/neural-networks-3/
            if np.abs(gradient) != 0 or np.abs(estimated_gradient) != 0:
                relative_error = np.abs(gradient - estimated_gradient) \
                                 / max(np.abs(gradient), np.abs(estimated_gradient))
            else:
                relative_error = np.abs(gradient - estimated_gradient)

            print('Parameter         : ' + key)
            print('index             : ' + str(index))
            print('Backpropagation   : ' + str(gradient))
            print('Gradient Checking : ' + str(estimated_gradient))
            print('Relative Error    : ' + str(relative_error))
            print()

            assert relative_error < 10 ** -7

            # Reset parameter
            parameter[index] = element


def test_nn_deep_backward_propagation_gradient_checking_relu():
    """
    Test output of nn.deep._backward_propagation() with Gradient Checking for NN using ReLU
    activation functions (except the output layer). Gradient Checking is a method of manually
    computing derivatives and comparing it with the output. ReLU has a "kink", so we have a higher
    threshold of error.
    """
    layers = np.random.randint(2, 5, np.random.randint(3, 5))
    layer_dims, activations = deep._split_layer_dims_activations(layers)
    m = np.random.randint(1, 10)
    X = np.random.randn(layer_dims[0], m)
    Y = np.random.randint(0, 2, size=(layer_dims[-1], m))

    # 1) Compute gradients with nn.deep._backward_propagation()
    parameters = deep._initialize_parameters(layer_dims)
    Y_computed, cache = deep._forward_propagation(X, parameters, activations)
    gradients = deep._backward_propagation(cache, Y_computed, Y, activations)

    # 2) Compute gradients manually
    epsilon = 10**-8
    for key, parameter in parameters.items():

        if key[0] == 'W':
            continue
        for index, element in np.ndenumerate(parameter):
            # Get J(..., x + e, ...)
            parameter[index] = element + epsilon
            Y_computed, _ = deep._forward_propagation(X, parameters, activations)
            cost_plus = deep._compute_cost(Y_computed, Y)

            # Get J(..., x - e, ...)
            parameter[index] = element - epsilon
            Y_computed, _ = deep._forward_propagation(X, parameters, activations)
            cost_minus = deep._compute_cost(Y_computed, Y)

            # Estimate gradient
            estimated_gradient = (cost_plus - cost_minus) / (2 * epsilon)
            gradient = gradients['d' + str(key)][index]

            # Relative Error: http://cs231n.github.io/neural-networks-3/
            if np.abs(gradient) != 0 or np.abs(estimated_gradient) != 0:
                relative_error = np.abs(gradient - estimated_gradient) \
                                 / max(np.abs(gradient), np.abs(estimated_gradient))
            else:
                relative_error = np.abs(gradient - estimated_gradient)

            print('Parameter         : ' + key)
            print('index             : ' + str(index))
            print('Backpropagation   : ' + str(gradient))
            print('Gradient Checking : ' + str(estimated_gradient))
            print('Relative Error    : ' + str(relative_error))
            print()

            assert relative_error < 10 ** -4

            # Reset parameter
            parameter[index] = element

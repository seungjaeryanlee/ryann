# pylint: disable=invalid-name, too-many-arguments, too-many-locals
"""
Defines a deep neural network model.
"""
import numpy as np

from ryann.activation import sigmoid, sigmoid_derivative, relu, relu_derivative

def train(X, Y, layers, num_iter, learning_rate=0.01, regularization=True, lambd=0.01):
    """
    Trains the model given the training set X, Y using a neural network with specified layers. The
    model iterates through the training set num_iter times with gradient descent with given learning
    rate. By default it uses L2 regularization to prevent overfitting.

    Parameters
    ----------
    X : np.ndarray
        The input matrix with shape (n_x, m) where n_x is the number of features and m is the number
        of examples.
    Y : np.ndarray
        The labels for each input column with shape (1, m).
    layers : list of dict or list of int or numpy.ndarray
        The list of dictionaries specifying the number of hidden units and activation function for
        each hidden layer or a list of ints specifying the number of hidden units. By default this
        model uses ReLU activation function for all layers except the last layer where it uses
        the sigmoid activation function.
    num_iter
        Number of times to run gradient descent.
    learning_rate : float
        The learning rate for gradient descent.

    Returns
    -------
    parameters : dict
        A dictionary of parameters learnt by the model. There are 2L parameters in total, where L
        is the number of layers. Each layer l has two parameters Wl and bl, respectively denoting
        the weight matrix and the bias vector for the layer l.
    costs : list
        A list of costs where costs[i] denotes the cost after i * 1000 iterations.
    """
    layer_dims, activations = _split_layer_dims_activations(layers)


    assert X.shape[1] == Y.shape[1]
    assert X.shape[0] == layer_dims[0]
    assert Y.shape[0] == layer_dims[-1]

    # 1. Initialize parameters W, b
    parameters = _initialize_parameters(layer_dims)
    costs = []

    # 2. Run gradient descent num_iter times
    for i in range(0, num_iter):
        # 2-1. Forward Propagation
        Y_computed, cache = _forward_propagation(X, parameters, activations)

        # 2-2. Compute cost
        if regularization:
            cost = _compute_cost_with_regularization(Y_computed, Y, parameters, lambd)
        else:
            cost = _compute_cost(Y_computed, Y)

        # 2-3. Backpropagation
        if regularization:
            gradients = _backward_propagation_with_regularization(cache, Y_computed, Y, activations,
                                                                  lambd)
        else:
            gradients = _backward_propagation(cache, Y_computed, Y, activations)

        # 2-4. Update parameters
        parameters = _update_parameters(parameters, gradients, learning_rate)

        # 2-5. Print the cost for every 1000 iterations of gradient descent
        if i % 1000 == 0:
            print('Cost after iteration %i: %f' %(i, cost))
            costs.append(cost)

    return parameters, costs


def _split_layer_dims_activations(layers):
    """
    Splits given list (layers) to two lists layer_dims and activations.

    Parameters
    ----------
    layers : list of dict or list of int or numpy.ndarray
        The list of dictionaries specifying the number of hidden units and activation function for
        each hidden layer or a list of ints specifying the number of hidden units. By default this
        model uses ReLU activation function for all layers except the last layer where it uses
        the sigmoid activation function.

    Returns
    -------
    layer_dims : list of int
        The list of units of each layer specified by the given list layers.
    activations: list of str
        The list of activation functions. Same as the ones in the given list (layers) if it was
        specified. Otherwise, the list adds ReLU functions for all layers except the output layer
        where it uses sigmoid.
    """
    assert isinstance(layers, (list, np.ndarray))
    assert len(layers) >= 2

    if isinstance(layers[0], (int, np.integer)):  # Activations not specified
        layer_dims = layers
        activations = ['relu'] * len(layer_dims)
        activations[-1] = 'sigmoid'
    else:
        layer_dims = [layer['units'] for layer in layers]
        activations = [layer['activation'] for layer in layers]

    return layer_dims, activations


def _initialize_parameters(layer_dims):
    """
    Initializes the parameters (weight matrices and bias vectors) based on given layer dimensions.
    Uses He initialization.

    Parameters
    ----------
    layer_dims : list of int
        A list of the number of units for each layer. Starts with the input layer and ends with
        the output layer.

    Returns
    -------
    parameters : dict
        A dictionary of initialized parameters. There are 2L parameters in total, where L is the
        number of layers. Each layer l has two parameters Wl and bl, respectively denoting
        the weight matrix and the bias vector for the layer l. The weight matrix is initialized with
        a normalized distribution with small variance to break symmetry, and the bias vectors are
        initialized as zero vectors.
    """
    parameters = {}

    for l in range(1, len(layer_dims)):
        variance = np.sqrt(2 / layer_dims[l-1]) # He initialization
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * variance
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


def _forward_propagation(X, parameters, activations):
    """
    Runs forward propagation from given parameters and input matrix X to compute the model's
    predictions.

    Parameters
    ----------
    X : np.ndarray
        The input matrix with shape (n_x, m) where n_x is the number of features and m is the number
        of examples.
    parameters : dict
        A dictionary of initialized parameters with weight matrices and bias vectors used for
        forward propagation.
    activations : list of str
        A list of names of activation functions to use for each layer.

    Returns
    -------
    Y_computed : np.ndarray
        The model's classification with shape (n_y, m).
    cache : dict
        A dictionary with 3L + 1 values, where L is the number of layers. Each layer l has Zl, the
        result of the linear action, Al, the result of the nonlinear activation function, and WL,
        the weight matrix. A0 is also in cache for convenient computation in backward_propagation,
        but is not used.
    """
    L = len(parameters) // 2 # Number of layers
    cache = {}
    cache['A0'] = X
    A = X

    for l in range(1, L + 1):
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]

        Z = np.dot(W, A) + b
        if activations[l] == 'relu':
            A = relu(Z)
        elif activations[l] == 'sigmoid':
            A = sigmoid(Z)
        else:
            raise ValueError('Only sigmoid and relu activations are supported.')

        cache['Z' + str(l)] = Z
        cache['A' + str(l)] = A
        cache['W' + str(l)] = W

    return A, cache


def _compute_cost(Y_computed, Y):
    """
    Computes the cross-entropy cost with predicted output and actual output.

    Parameters
    ----------
    Y_computed : np.ndarray
        The sigmoid output of the neural network with shape (n_y, m).
    Y : np.ndarray
        The matrix with correct labels with shape (n_y, m).

    Returns
    -------
    cost : float
        The cross-entropy cost.
    """
    m = Y.shape[1]

    cost = -1 / m * np.sum(Y * np.log(Y_computed) + (1 - Y) * np.log(1 - Y_computed))
    cost = np.squeeze(cost)

    return cost


def _compute_cost_with_regularization(Y_computed, Y, parameters, lambd):
    """
    Computes the cross-entropy cost with L2 regularization.

    Parameters
    ----------
    Y_computed : np.ndarray
        The sigmoid output of the neural network with shape (n_y, m).
    Y : np.ndarray
        The matrix with correct labels with shape (n_y, m).
    parameters : dict
        A dictionary of initialized parameters with weight matrices and bias vectors used for
        forward propagation.
    lambd : float
        The regularization constant. The higher lambd is, the simpler the model becomes.

    Returns
    -------
    cost : float
        The cross-entropy cost.
    """
    m = Y.shape[1]

    cost = _compute_cost(Y_computed, Y)

    regularization_sum = np.sum([np.sum(parameter ** 2) for _, parameter in parameters.items()])
    regularization_cost = lambd / (2 * m) * regularization_sum

    return cost + regularization_cost


def _backward_propagation(cache, Y_computed, Y, activations):
    """
    Runs backward propagation on given parameters using cached values, X, and Y.

    Parameters
    ----------
    cache : dict
        A dictionary with 3L+1 values, where L is the number of layers. Each layer l has Zl, the
        result of the linear action, Al, the result of the nonlinear activation function, and Wl,
        the weight matrix. A0 is added for convenience but is not used. The dictionary was filled
        from forward propagation.
    Y_computed : np.ndarray
        The sigmoid output of the neural network with shape (n_y, m).
    Y : np.ndarray
        The matrix with correct labels with shape (n_y, m).
    activations : list of str
        A list of names of activation functions to use for each layer.

    Returns
    -------
    gradients : dict
        A dictionary of gradients with respect to given parameters. Each layer l has 3 gradients:
        dAl, dWl, dbl.
    """
    gradients = {}
    L = (len(cache) - 1) // 3
    m = Y_computed.shape[1]

    # Calculate gradient of last activation: Y_computed
    dA = -(Y / Y_computed - (1 - Y) / (1 - Y_computed))

    # from L to 1:
    for l in reversed(range(1, L + 1)):
        dZ = _activation_backward(dA, cache['Z' + str(l)], activations[l])
        A_prev = cache['A' + str(l - 1)]
        W = cache['W' + str(l)]
        gradients['dW' + str(l)] = 1 / m * np.dot(dZ, A_prev.T)
        gradients['db' + str(l)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        gradients['dA' + str(l - 1)] = np.dot(W.T, dZ)
        dA = gradients['dA' + str(l - 1)] # Update dA for previous layer

    return gradients


def _backward_propagation_with_regularization(cache, Y_computed, Y, activations, lambd):
    """
    Runs backward propagation with L2 regularization on given parameters using cached values, X,
    and Y.

    Parameters
    ----------
    cache : dict
        A dictionary with 3L+1 values, where L is the number of layers. Each layer l has Zl, the
        result of the linear action, Al, the result of the nonlinear activation function, and Wl,
        the weight matrix. A0 is added for convenience but is not used. The dictionary was filled
        from forward propagation.
    Y_computed : np.ndarray
        The sigmoid output of the neural network with shape (n_y, m).
    Y : np.ndarray
        The matrix with correct labels with shape (n_y, m).
    activations : list of str
        A list of names of activation functions to use for each layer.
    lambd : float
        The regularization constant. The higher lambd is, the simpler the model becomes.

    Returns
    -------
    gradients : dict
        A dictionary of gradients with respect to given parameters. Each layer l has 3 gradients:
        dAl, dWl, dbl.
    """
    gradients = {}
    L = (len(cache) - 1) // 3
    m = Y_computed.shape[1]

    # Calculate gradient of last activation: Y_computed
    dA = -(Y / Y_computed - (1 - Y) / (1 - Y_computed))

    # from L to 1:
    for l in reversed(range(1, L + 1)):
        dZ = _activation_backward(dA, cache['Z' + str(l)], activations[l])
        A_prev = cache['A' + str(l - 1)]
        W = cache['W' + str(l)]
        gradients['dW' + str(l)] = 1 / m * np.dot(dZ, A_prev.T) + lambd / m * W
        gradients['db' + str(l)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        gradients['dA' + str(l - 1)] = np.dot(W.T, dZ)
        dA = gradients['dA' + str(l - 1)] # Update dA for previous layer

    return gradients


def _activation_backward(dAl, Zl, activation_name):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Parameters
    ----------
    dAl : np.ndarray
        The derivative of the cost function J with respect to Al, the activation of layer l.

    Zl : np.ndarray
        The linear output of layer l.

    activation_name : str
        The string that specifies the activation used. Can be either 'sigmoid' or 'relu'.

    Returns
    -------
    dZl : np.ndarray
        The derivative of the cost function J with repsect to Zl, the linear output of layer l.
    """
    if activation_name == 'sigmoid':
        dZl = dAl * sigmoid_derivative(Zl)
    elif activation_name == 'relu':
        dZl = dAl * relu_derivative(Zl)
    else:
        raise ValueError('Only sigmoid and relu activations are supported.')

    return dZl


def _update_parameters(parameters, gradients, learning_rate):
    """
    Updates given parameters with given gradients using gradient descent.

    Parameters
    ----------
    parameters : dict
        A dictionary of parameters with weight matrices and bias vectors that will be updated based
        on the given gradients and learning rate.
    gradients : dict
        A dictionary of gradients for given parameters calculated by backward propagation.

    Returns
    -------
    parameters : dict
        A dictionary of parameters updated with given gradients and learning rate.
    """
    L = len(parameters) // 2

    for l in range(1, L + 1):
        parameters['W' + str(l)] -= learning_rate * gradients['dW' + str(l)]
        parameters['b' + str(l)] -= learning_rate * gradients['db' + str(l)]

    return parameters


def predict(parameters, activations, X):
    """
    Returns predictions on given training examples X using given parameters.

    Parameters
    ----------
    parameters : dict
        A dictionary of parameters with weight matrices and bias vectors used to predict the output
        of the given input examples X.
    activations : list of str
        A list of names of activation functions to use for each layer.
    X : np.ndarray
        The input matrix with shape (n_x, m) where n_x is the number of features and m is the number
        of examples.

    Returns
    -------
    predictions : np.ndarray
        A vector of predictions by the model specified by the given parameters.
    """
    A, _ = _forward_propagation(X, parameters, activations)

    return (A > 0.5).astype(int)

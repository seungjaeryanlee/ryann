# pylint: disable=invalid-name
"""
Defines a deep neural network model.
"""
import numpy as np

from ryann.activation import sigmoid, sigmoid_derivative, relu, relu_derivative

def train(X, Y, layers, num_iter, learning_rate=0.01):
    """

    Parameters
    ----------
    X : np.ndarray
        The input matrix with shape (n_x, m) where n_x is the number of features and m is the number
        of examples.
    Y : np.ndarray
        The labels for each input column with shape (1, m).
    layers : list of dict or list of int
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
    assert isinstance(layers, (list,))
    assert len(layers) >= 2

    if isinstance(layers[0], int): # Activations not specified
        layer_dims = layers
        activations = ['relu'] * len(layer_dims)
        activations[-1] = 'sigmoid'
    else:
        layer_dims = [layer['units'] for layer in layers]
        activations = [layer['activation'] for layer in layers]


    assert X.shape[1] == Y.shape[1]
    assert X.shape[0] == layer_dims[0]
    assert Y.shape[0] == layer_dims[-1]

    # 1. Initialize parameters W, b
    parameters = _initialize_parameters(layer_dims)
    costs = []

    # 2. Run gradient descent num_iter times
    for i in range(0, num_iter):
        # 2-1. Forward Propagation
        Y_computed, cache = _forward_propagation(X, parameters)

        # 2-2. Compute cost
        cost = _compute_cost(Y_computed, Y)

        # 2-3. Backpropagation
        gradients = _backward_propagation(cache, Y_computed, Y)

        # 2-4. Update parameters
        parameters = _update_parameters(parameters, gradients, learning_rate)

        # 2-5. Print the cost for every 1000 iterations of gradient descent
        if i % 1000 == 0:
            print('Cost after iteration %i: %f' %(i, cost))
            costs.append(cost)

    return parameters, costs


def _initialize_parameters(layer_dims, variance=0.01):
    """
    Initializes the parameters (weight matrices and bias vectors) based on given layer dimensions.

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
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * variance
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


def _forward_propagation(X, parameters):
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
        A = sigmoid(Z)
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


def _backward_propagation(cache, Y_computed, Y):
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
        dZ = _activation_backward(dA, cache['Z' + str(l)], 'sigmoid')
        A_prev = cache['A' + str(l - 1)]
        W = cache['W' + str(l)]
        gradients['dW' + str(l)] = 1 / m * np.dot(dZ, A_prev.T)
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


def predict(parameters, X):
    """
    Returns predictions on given training examples X using given parameters.

    Parameters
    ----------
    parameters : dict
        A dictionary of parameters with weight matrices and bias vectors used to predict the output
        of the given input examples X.
    X : np.ndarray
        The input matrix with shape (n_x, m) where n_x is the number of features and m is the number
        of examples.

    Returns
    -------
    predictions : np.ndarray
        A vector of predictions by the model specified by the given parameters.
    """
    A, _ = _forward_propagation(X, parameters)

    return (A > 0.5).astype(int)

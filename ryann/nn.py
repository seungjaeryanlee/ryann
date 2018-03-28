# pylint: disable=invalid-name
"""
Defines a basic neural network model.
"""
import numpy as np


def sigmoid(x):
    """
    Applies the sigmoid function to a number or to a NumPy ndarray.

    Parameters
    ----------
    x : float or np.ndarray
        A single number or a ndarray to apply the sigmoid function.

    Returns
    -------
    float or ndarray
        If x was a single float, returns sigmoid(x). If x was an np.ndarray, returns a np.ndarray
        with sigmoid function applied to each element of the ndarray.
    """
    return 1 / (1 + np.exp(-x))

def shallow_nn(X, Y, n_h, num_iter):
    """

    Parameters
    ----------
    X : np.ndarray
        The input matrix with shape (n_x, m) where n_x is the number of features and m is the number
        of examples.
    Y : np.ndarray
        The labels for each input column with shape (1, m).
    n_h : int
        The number of hidden units in the hidden layer.
    num_iter
        Number of times to run gradient descent.

    Returns
    -------
    parameters : dict
        A dictionary of parameters learnt by the model.
    """
    assert X.shape[1] == Y.shape[1]

    n_x = X.shape[0]
    m = X.shape[1]
    n_y = Y.shape[0]

    # 1. Initialize parameters W1, b1, W2, b2
    parameters = _initialize_parameters(n_x, n_h, n_y)

    # 2. Run gradient descent num_iter times
    for i in range(0, num_iter):
        # 2-1. Forward Propagation
        Y_computed, cache = _forward_propagation(X, parameters)

        # 2-2. Compute cost
        cost = _compute_cost(Y_computed, Y)

        # 2-3. Backpropagation
        gradients = _backward_propagation(parameters, cache, X, Y)

        # 2-4. Update parameters
        parameters = _update_parameters(parameters, gradients)

        # 2-5. Print the cost for every 1000 iterations of gradient descent
        if i % 1000 == 0:
            print('Cost after iteration %i: %f' %(i, cost))

    return parameters

def _initialize_parameters(n_x, n_h, n_y):
    """
    Initializes parameters (weight matrices and bias vectors) based on given layer sizes.

    Parameters
    ----------
    n_x : int
        Number of units in the input layer.
    n_h : int
        Number of units in the hidden layer.
    n_y : int
        Number of units in the output layer.

    Returns
    -------
    parameters : dict
        A dictionary with keys W1, b1, W2, b2 with b1, b2 initialized as zero vectors and W1, W2
        initialized as a random matrix.
    """
    WEIGHT_STANDARD_DEVIATION = 0.01

    W1 = np.random.randn(n_h, n_x) * WEIGHT_STANDARD_DEVIATION
    W2 = np.random.randn(n_y, n_h) * WEIGHT_STANDARD_DEVIATION
    b1 = np.zeros((n_h, 1))
    b2 = np.zeros((n_y, 1))

    return {
        'W1': W1,
        'W2': W2,
        'b1': b1,
        'b2': b2,
    }

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
        A dictionary of parameters W1, W2, b1, b2.

    Returns
    -------
    Y_computed : np.ndarray
        The model's classification with shape (n_y, m).
    cache : dictionary
        A dictionary of Z1, A1, Z2, A2 that will be used in backward_propagation.
    """
    Z1 = np.dot(parameters['W1'], X) + parameters['b1']
    A1 = np.tanh(Z1)
    Z2 = np.dot(parameters['W2'], A1) + parameters['b2']
    A2 = sigmoid(Z2)

    cache = {
        'Z1': Z1,
        'A1': A1,
        'Z2': Z2,
        'A2': A2,
    }
    return A2, cache


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
    assert Y_computed.shape == Y.shape

    m = Y.shape[1]

    # calculate a vector of losses for each example, and average them
    loss = Y * np.log(Y_computed) + (1 - Y) * np.log(1 - Y_computed)
    print(loss)
    cost = -1 / m * np.sum(loss)

    # make cost a float, not an one-element array
    cost = np.squeeze(cost)

    return cost


def _backward_propagation(parameters, cache, X, Y):
    """
    Runs backward propagation on given parameters using cached values, X, and Y.

    Parameters
    ----------
    parameters : dict
        A dictionary of parameters W1, W2, b1, b2.
    cache : dict
        A dictionary of Z1, A1, Z2, A2 that will be used in backward_propagation.
    X : np.ndarray
        The input matrix with shape (n_x, m) where n_x is the number of features and m is the number
        of examples.
    Y : np.ndarray
        The matrix with correct labels with shape (n_y, m).

    Returns
    -------
    grads : dict
        A dictionary of gradients with respect to given parameters.
    """
    m = X.shape[1]
    W1 = parameters['W1']
    W2 = parameters['W2']
    A1 = cache['A1']
    A2 = cache['A2']

    dZ2 = A2 - Y
    dW2 = 1 / m * np.dot(dZ2, A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - A1 ** 2)
    dW1 = 1 / m * np.dot(dZ1, X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    return {
        'W1': dW1,
        'W2': dW2,
        'b1': db1,
        'b2': db2,
    }


def _update_parameters(parameters, gradients):
    """
    Updates given parameters with given gradients using gradient descent.

    Parameters
    ----------
    parameters : dict
        A dictionary of parameters W1, W2, b1, b2.
    gradients : dict
        A dictionary of gradients for given parameters.

    Returns
    -------
    parameters : dict
        A dictionary of parameters updated with given gradients.
    """
    return {
        'W1': 0,
        'W2': 0,
        'b1': 0,
        'b2': 0,
    }

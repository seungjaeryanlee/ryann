# pylint: disable=invalid-name
"""
Defines a deep neural network model.
"""
import numpy as np

from ryann.activation import sigmoid

def train(X, Y, layer_dims, num_iter):
    """

    Parameters
    ----------
    X : np.ndarray
        The input matrix with shape (n_x, m) where n_x is the number of features and m is the number
        of examples.
    Y : np.ndarray
        The labels for each input column with shape (1, m).
    layer_dims : list
        The list of number of hidden units for each hidden layer.
    num_iter
        Number of times to run gradient descent.

    Returns
    -------
    parameters : dict
        A dictionary of parameters learnt by the model.
    costs : list
        A list of costs where costs[i] denotes the cost after i * 1000 iterations.
    """
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
        gradients = _backward_propagation(parameters, cache, X, Y)

        # 2-4. Update parameters
        parameters = _update_parameters(parameters, gradients, 0.01)

        # 2-5. Print the cost for every 1000 iterations of gradient descent
        if i % 1000 == 0:
            print('Cost after iteration %i: %f' %(i, cost))
            costs.append(cost)

    return parameters, costs


def _initialize_parameters(layer_dims):
    """
    Initializes parameters (weight matrices and bias vectors) based on given layer sizes.

    Parameters
    ----------
    layer_dims : list of int
        A list of the number of units for each layer. Starts with the input layer and ends with
        the output layer.

    Returns
    -------
    parameters : list of tuple
        A list of L tuples where each tuple has first element W (weight matrix) and second element
        b (bias vector) initialized. W is initialized with a random normal distribution to break
        symmetry and b is initialized as the zero vector.
    """
    parameters = [(0, 0)] # One added for W0, b0 (not used but useful for spacing)
    L = len(layer_dims)  # Number of layers

    for l in range(1, L):
        W = np.random.randn(layer_dims[l], layer_dims[l-1])
        b = np.zeros((layer_dims[l], 1))
        parameters.append((W, b))

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
    parameters : list of tuple
        A list of L tuples where each tuple has first element W (weight matrix) and second element
        b (bias vector).

    Returns
    -------
    Y_computed : np.ndarray
        The model's classification with shape (n_y, m).
    cache : list of tuple
        A list of tuples with L tuples with first element of tuple being Z (the matrix product) and
        the second element being A (the activation).
    """
    L = len(parameters) # Number of layers

    cache = [(0, 0)]
    A = X

    for l in range(1, L):
        W = parameters[l][0]
        b = parameters[l][1]

        Z = np.dot(W, A) + b
        A = sigmoid(Z)
        cache.append((Z, A))

    return cache[-1][1], cache


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
    pass


def _backward_propagation(parameters, cache, X, Y):
    """
    Runs backward propagation on given parameters using cached values, X, and Y.

    Parameters
    ----------
    parameters : list of tuple
        A list of L tuples where each tuple has first element W (weight matrix) and second element
        b (bias vector).
    cache : dict
        A dictionary of matrix products Z1, Z2, ... and activations A1, A2, ... that will be used
        in backward_propagation.
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
    pass


def _update_parameters(parameters, gradients, learning_rate=0.01):
    """
    Updates given parameters with given gradients using gradient descent.

    Parameters
    ----------
    parameters : list of tuple
        A list of L tuples where each tuple has first element W (weight matrix) and second element
        b (bias vector).
    gradients : dict
        A dictionary of gradients for given parameters.

    Returns
    -------
    parameters : dict
        A dictionary of parameters updated with given gradients.
    """
    pass


def predict(parameters, X):
    """
    Returns predictions on given training examples X using given parameters.

    Parameters
    ----------
    parameters : list of tuple
        A list of L tuples where each tuple has first element W (weight matrix) and second element
        b (bias vector).
    X : np.ndarray
        The input matrix with shape (n_x, m) where n_x is the number of features and m is the number
        of examples.

    Returns
    -------
    predictions : np.ndarray
        A vector of predictions by the model.
    """
    pass

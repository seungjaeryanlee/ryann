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
    parameters = initialize_parameters(n_x, n_h, n_y)

    # 2. Run gradient descent num_iter times
    for i in range(0, num_iter):
        # 2-1. Forward Propagation
        Y_computed, cache = forward_propagation(X, parameters)

        # 2-2. Compute cost
        cost = compute_cost(Y_computed, Y, parameters)

        # 2-3. Backpropagation
        gradients = backward_propagation(parameters, cache, X, Y)

        # 2-4. Update parameters
        parameters = update_parameters(parameters, gradients)

        # 2-5. Print the cost for every 1000 iterations of gradient descent
        if i % 1000 == 0:
            print('Cost after iteration %i: %f' %(i, cost))

    return parameters

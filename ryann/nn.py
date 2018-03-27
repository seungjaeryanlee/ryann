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
        The input matrix with shape (n, m) where n is the number of features and m is the number of
        examples.
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
    return {}
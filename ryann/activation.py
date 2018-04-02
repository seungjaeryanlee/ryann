# pylint: disable=invalid-name
"""
Defines the activation functions.
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


def sigmoid_derivative(x):
    """
    Applies the derivative of the sigmoid function to a number of to a NumPy ndarray.

    Parameters
    ----------

    x : float or np.ndarray
        A single number or a ndarray to apply the sigmoid' function.

    Returns
    -------
    float or ndarray
        If x was a single float, returns sigmoid'(x). If x was an np.ndarray, returns a np.ndarray
        with sigmoid' function applied to each element of the ndarray.

    """
    return sigmoid(x) * (1 - sigmoid(x))

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

def relu(x):
    """
    Applies the ReLU function to a number or to a NumPy ndarray.

    Parameters
    ----------
    x : float or np.ndarray
        A single number or a ndarray to apply the ReLU function.

    Returns
    -------
    float or ndarray
        If x was a single float, returns ReLU(x). If x was an np.ndarray, returns a np.ndarray
        with ReLU function applied to each element of the ndarray.
    """
    return np.maximum(x, 0)


def relu_derivative(x):
    """
    Applies the derivative of the ReLU function to a number or to a NumPy ndarray. When x = 0,
    it returns 0.

    Parameters
    ----------
    x : float or np.ndarray
        A single number or a ndarray to apply the derivative of the ReLU function.

    Returns
    -------
    float or ndarray
        If x was a single float, returns ReLU(x). If x was an np.ndarray, returns a np.ndarray
        with the derivative of the ReLU applied to each element of the ndarray.
    """
    return (x > 0) * 1

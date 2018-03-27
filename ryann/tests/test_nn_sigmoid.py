# pylint: disable=no-self-use, too-few-public-methods, invalid-name
"""
Tests all functions in ryann.nn module.
"""
import numpy as np
from numpy.testing import assert_almost_equal
import pytest
from ryann import nn


def test_nn_sigmoid_zero():
    """
    Tests that nn.sigmoid(0) = 0.5
    """
    assert nn.sigmoid(0) == pytest.approx(0.5)


def test_nn_sigmoid_zero_ndarray():
    """
    Tests that nn.sigmoid() returns a NumPy ndarray of 0.5 when given a NumPy ndarray of 0.
    """
    x = int(np.random.rand() * 10)
    y = int(np.random.rand() * 10)
    z = int(np.random.rand() * 10)
    X = np.zeros((x, y, z))

    assert_almost_equal(nn.sigmoid(X), np.full((x, y, z), 0.5))


def test_nn_sigmoid_increasing():
    """
    Tests that nn.sigmoid() is an always increasing function.
    """
    x = np.random.rand() * 10
    y = np.random.rand() * 10

    if x >= y:
        assert nn.sigmoid(x) >= nn.sigmoid(y)
    else:
        assert nn.sigmoid(x) < nn.sigmoid(y)


def test_nn_sigmoid_formula():
    """
    Tests that nn.sigmoid() uses the correct sigmoid formula 1 / (1 + exp(-x)).
    """
    x = np.random.randn()

    assert_almost_equal(nn.sigmoid(x), 1 / (1 + np.exp(-x)))


def test_nn_sigmoid_formula_ndarray():
    """
    Tests that nn.sigmoid() uses the correct sigmoid formula 1 / (1 + exp(-x)) when the input is
    an NumPy ndarray.
    """
    X = np.random.randn(3, 5)
    Y = nn.sigmoid(X)

    for index, x in np.ndenumerate(X):
        assert_almost_equal(Y.item(index), 1 / (1 + np.exp(-x)))

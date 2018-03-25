# pylint: disable=no-self-use, too-few-public-methods
"""
A sample test file for a module.
"""
from ryann import index


class TestIndex:
    """
    Tests all functions in ryann.index.
    """
    def test_hello_world(self):
        """
        Tests the output of ryann.index.hello_world().
        """
        assert index.hello_world() == 'Hello World'

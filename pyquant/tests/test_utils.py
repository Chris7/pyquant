__author__ = 'chris'
import unittest
import numpy as np

from pyquant import utils

class UtilsTests(unittest.TestCase):
    def test_select_window(self):
        x = range(10)
        selection = utils.select_window(x, 0, 3)
        self.assertListEqual(selection, [0, 1, 2, 3])
        selection = utils.select_window(x, 3, 3)
        self.assertListEqual(selection, [0, 1, 2, 3, 4, 5, 6])
        selection = utils.select_window(x, 8, 3)
        self.assertListEqual(selection, [5, 6, 7, 8, 9])
        selection = utils.select_window(x, 8, 20)
        self.assertListEqual(selection, x)

if __name__ == '__main__':
    unittest.main()

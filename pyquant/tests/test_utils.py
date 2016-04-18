__author__ = 'chris'
from unittest import TestCase
import numpy as np
from pyquant import utils


class TestUtils(TestCase):

    def test_gauss_ndim(self):
        assert np.round(self.two_gauss[np.where(self.x==0)],2) == np.round(get_gauss_value(0, self.amp, self.mu, self.std)+get_gauss_value(0, self.amp, self.mu2, self.std),2)

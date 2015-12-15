__author__ = 'chris'
from unittest import TestCase
import numpy as np
from .. import peaks

def get_gauss_value(x, amp, mu, std):
    return amp*np.exp(-(x - mu)**2/(2*std**2))

class GaussianTests(TestCase):
    def setUp(self):
        self.amp, self.mu, self.std, self.mu2 = 1., 0., 1., 3.
        self.one_gauss_params = np.array([self.amp, self.mu, self.std], dtype=np.float)
        self.two_gauss_params = np.array([self.amp, self.mu, self.std, self.amp, self.mu2, self.std], dtype=np.float)
        self.x = np.array(np.linspace(-5,5,101), dtype=np.float)
        self.one_gauss = peaks.gauss_ndim(self.x, self.one_gauss_params)
        self.two_gauss = peaks.gauss_ndim(self.x, self.two_gauss_params)

# def test_within_bounds():
#     assert peaks.within_bounds(np.array([-1,1]), [(-1, 2), (-1,0)]) == 0
#     assert peaks.within_bounds(np.array([-1,1]), [(-1, 2), (-1,1)]) == 0

# def test_gauss():
#     amp, mu, std = 1, 0, 1
#     x = np.linspace(-5,6,11)
#     y = peaks.gauss(x, amp, mu, std)
#     assert np.max(y) == 1
#     assert y[np.where(x)==1] == get_gauss_value(1, amp, mu, std)

    def test_gauss_ndim(self):
        assert np.round(self.two_gauss[np.where(self.x==0)],2) == np.round(get_gauss_value(0, self.amp, self.mu, self.std)+get_gauss_value(0, self.amp, self.mu2, self.std),2)

    def test_peak_fitting(self):
        # first, a simple case
        params, residual = peaks.findAllPeaks(self.x, self.one_gauss)
        np.testing.assert_allclose(params, self.one_gauss_params, atol=self.std/2)
        params, residual = peaks.findAllPeaks(self.x, self.two_gauss)
        # print(self.two_gauss, ',',peaks.gauss_ndim(self.x, params))
        np.testing.assert_allclose(params, self.two_gauss_params, atol=self.std/2)
        failures = 0
        for i in range(10):
            self.noisy_two_gauss = self.two_gauss + np.random.normal(0, 0.05, size=len(self.two_gauss))
            params, residual = peaks.findAllPeaks(self.x, self.noisy_two_gauss, rt_peak=-1, filter=True, max_peaks=30, debug=False, bigauss_fit=True)
            # we compare just the means here because the amplitude and variance can change too much due to the noise for reliable testing, but manual
            # inspect reveals the fits to be accurate
            try:
                np.testing.assert_allclose(params[1::4], self.two_gauss_params[1::3], atol=self.std/2)
            except AssertionError:
                failures += 1
        self.assertLess(failures, 3)
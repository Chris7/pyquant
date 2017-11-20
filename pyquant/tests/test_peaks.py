__author__ = 'chris'
import os
import unittest

import numpy as np
import six
import six.moves.cPickle as pickle

from pyquant.tests.utils import timer
from pyquant.tests.mixins import FileMixins, GaussianMixin
from pyquant import peaks
from pyquant import PEAK_FINDING_DERIVATIVE, PEAK_FINDING_REL_MAX, PEAK_FIT_MODE_FAST

def get_gauss_value(x, amp, mu, std):
    return amp*np.exp(-(x - mu)**2/(2*std**2))


class PeakFindingTests(FileMixins, unittest.TestCase):
    def test_returns_with_bad_data(self):
        params, residual = peaks.findAllPeaks(np.array([1,2,3,4,5,6]), np.array([0,0,0,0,0,0]))
        self.assertEqual(len(params), 0)
        self.assertEqual(residual, np.inf)

    def test_max_peaks(self):
        # Regression where relative-max is reporting 2 peaks when max_peaks is set to 1. This occurred
        # because we enforced max_peaks for each peak width when using the relative-max setting. Thus,
        # the max peak for each peak width was combined to the final peak report. The update was to
        # pick the peak_width with the lowest BIC.
        with open(os.path.join(self.data_dir, 'peak_data.pickle'), 'rb') as peak_file:
            data = pickle.load(peak_file, encoding='latin1') if six.PY3 else pickle.load(peak_file)

        x, y = data['max_peaks_relative-max']
        params, residual = peaks.findAllPeaks(x, y, max_peaks=1, peak_find_method=PEAK_FINDING_REL_MAX, fit_mode=PEAK_FIT_MODE_FAST)
        self.assertEqual(len(params), 3)

    def test_max_peaks_with_rt_peak_regression(self):
        with open(os.path.join(self.data_dir, 'peak_data.pickle'), 'rb') as peak_file:
            data = pickle.load(peak_file, encoding='latin1') if six.PY3 else pickle.load(peak_file)

        x, y = data['max_peaks_rt-peak-regression']
        params, residual = peaks.findAllPeaks(x, y, max_peaks=1, rt_peak=360, fit_mode=PEAK_FIT_MODE_FAST)
        np.testing.assert_allclose(params[1], desired=365.78, atol=0.1)

    def test_fit_baseline_derivative(self):
        with open(os.path.join(self.data_dir, 'peak_data.pickle'), 'rb') as peak_file:
            data = pickle.load(peak_file, encoding='latin1') if six.PY3 else pickle.load(peak_file)

        x, y = data['max_peaks_rt-peak-regression']
        params, residual = peaks.findAllPeaks(
            x,
            y,
            max_peaks=1,
            fit_baseline=True,
            rt_peak=328,
            peak_find_method='derivative',
            fit_mode=PEAK_FIT_MODE_FAST,
        )
        np.testing.assert_allclose(
            params[:3],
            desired=np.array([1320.60, 330.15, 4.22]),
            atol=10
        )


    def test_segmenty_negatives(self):
        # Regression where a mostly positive dataset with negatives led to -inf values in the data array
        # due to np.max(segment_y) being 0 since all data was negative
        with open(os.path.join(self.data_dir, 'peak_data.pickle'), 'rb') as peak_file:
            data = pickle.load(peak_file, encoding='latin1') if six.PY3 else pickle.load(peak_file)

        x, y = data['invalid_operands']
        params, res = peaks.findAllPeaks(x, y, max_peaks=-1, bigauss_fit=True, peak_find_method=PEAK_FINDING_DERIVATIVE)
        means = params[1::4]
        desired = np.array([
            0.13515435795212014, 0.33, 1.474992882679938, 1.799090776628427, 2.1804381077669395, 2.6350000000000002,
            3.227084689771589, 3.617021549048893, 4.903333333333333, 5.296162908137783, 5.8366172292356175
        ])
        np.testing.assert_allclose(means, desired=desired, atol=0.1)

class GaussianTests(GaussianMixin, unittest.TestCase):

    @timer
    def test_gauss_ndim(self):
        assert np.round(self.two_gauss[np.where(self.x==0)],2) == np.round(get_gauss_value(0, self.amp, self.mu, self.std)+get_gauss_value(0, self.amp, self.mu2, self.std),2)

    @timer
    def test_peak_fitting(self):
        # first, a simple case
        params, residual = peaks.findAllPeaks(self.x, self.one_gauss, )
        np.testing.assert_allclose(params, self.one_gauss_params, atol=self.std / 2)
        params, residual = peaks.findAllPeaks(self.x, self.one_gauss, peak_find_method=PEAK_FINDING_DERIVATIVE)
        np.testing.assert_allclose(params, self.one_gauss_params, atol=self.std / 2)

        params, residual = peaks.findAllPeaks(self.x, self.two_gauss)
        np.testing.assert_allclose(params, self.two_gauss_params, atol=self.std / 2)
        params, residual = peaks.findAllPeaks(self.x, self.two_gauss, peak_find_method=PEAK_FINDING_DERIVATIVE)
        np.testing.assert_allclose(params, self.two_gauss_params, atol=self.std / 2)
        params, residual = peaks.findAllPeaks(self.x, self.noisy_two_gauss, rt_peak=None, filter=True, max_peaks=30, bigauss_fit=True, snr=1)
        np.testing.assert_allclose(params[1::4], self.two_gauss_params[1::3], atol=self.std/2)

    def test_targeted_search(self):
        # We should not find anything where there are no peaks
        res, residual = peaks.targeted_search(self.x, self.two_gauss, self.x[2], attempts=2, peak_finding_kwargs={'max_peaks': 2})
        self.assertIsNone(res)

        # Should find the peak when we are in its area
        res, residual = peaks.targeted_search(self.x, self.two_gauss, self.two_gauss_params[1])
        self.assertIsNotNone(res)

    def test_experimental(self):
        # Experimental data
        x, y = self.peak_data['offset_fit']
        params, residual = peaks.findAllPeaks(x, y, bigauss_fit=True, filter=True)
        np.testing.assert_allclose(
            params,
            np.array([
                13219262.587656807, 46.821340819991505, 0.06523272363478014, 0.18374422913588656,
                3347200.309180678, 47.497, 0.6166821402545103, 0.3817876338966981,
                1880722.1582992678, 48.14756707645761, 0.17537885391522443, 0.4846763157315077,
                1473766.5256005626, 49.52607160264086, 0.020199999999999108, 0.22250905781532157
            ]),
            atol=10,
        )

if __name__ == '__main__':
    unittest.main()

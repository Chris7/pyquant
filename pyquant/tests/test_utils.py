__author__ = 'chris'
import os
import unittest

import numpy as np
import pandas as pd

from pyquant import utils
from pyquant.tests.mixins import GaussianMixin

class UtilsTests(GaussianMixin, unittest.TestCase):
    def setUp(self):
        super(UtilsTests, self).setUp()
        self.base_dir = os.path.split(os.path.abspath(__file__))[0]
        self.data_dir = os.path.join(self.base_dir, 'data')

    def test_select_window(self):
        x = list(range(10))
        selection = utils.select_window(x, 0, 3)
        self.assertListEqual(selection, [0, 1, 2, 3])
        selection = utils.select_window(x, 3, 3)
        self.assertListEqual(selection, [0, 1, 2, 3, 4, 5, 6])
        selection = utils.select_window(x, 8, 3)
        self.assertListEqual(selection, [5, 6, 7, 8, 9])
        selection = utils.select_window(x, 8, 20)
        self.assertListEqual(selection, x)

    def test_divide_peaks(self):
        chunks = utils.divide_peaks(self.one_gauss)
        two_gauss_chunks = utils.divide_peaks(self.two_gauss)
        self.assertEqual(len(chunks), 0)
        self.assertEqual(len(two_gauss_chunks), 1)
        self.assertEqual(two_gauss_chunks[0], 65)

    def test_calculate_theoretical_distribution(self):
        peptide = 'PEPTIDE'
        pep_comp = utils.calculate_theoretical_distribution(peptide=peptide)
        ele_comp = utils.calculate_theoretical_distribution(elemental_composition={'C': 7})
        np.testing.assert_almost_equal(pep_comp.values.tolist(), [0.6411550319843632, 0.2662471681269686, 0.07401847648709056, 0.015434213671511215, 0.002681646815294711])
        np.testing.assert_almost_equal(ele_comp.values.tolist(), [0.9254949240653104, 0.07205572209608584, 0.002404285974894674])

    def test_ml(self):
        data = os.path.join(self.data_dir, 'ml_data.tsv')
        dat = pd.read_table(data)
        labels = ['Heavy', 'Medium', 'Light']
        utils.perform_ml(dat, {i: [] for i in labels})
        for label1 in labels:
            for label2 in labels:
                if label1 == label2:
                    continue
                col = '{}/{} Confidence'.format(label1, label2)
                self.assertNotEqual(sum(pd.isnull(dat['Heavy/Light Confidence']) == False), 0)

    def test_merge_peaks(self):
        peaks = {1: {'minima': [0,1,2,4,5], 'peaks': [3]}, 2: {'minima': [0,1,2,4,5], 'peaks': [3]}, 7: {'minima': [0,1,2,4,5], 'peaks': [3]}}
        merged = utils.merge_peaks(peaks)
        self.assertDictEqual(merged, {7: {'minima': [0, 1, 2, 4, 5], 'peaks': [3]}})

        peaks = {1: {'minima': [0,1,2,4,5], 'peaks': [3]}}
        merged = utils.merge_peaks(peaks)
        self.assertDictEqual(merged, {1: {'minima': [0,1,2,4,5], 'peaks': [3]}})

        peaks = {1: {'minima': [0,5], 'peaks': [3,7,8]}, 2: {'minima': [0,5], 'peaks': [3,7]}, 7: {'minima': [0,5], 'peaks': [3,7]}}
        merged = utils.merge_peaks(peaks)
        self.assertDictEqual(merged, {1: {'minima': [0,5], 'peaks': [3,7,8]}, 7: {'minima': [0,5], 'peaks': [3,7]}})

    def test_get_cross_points(self):
        y = [1, 1, 1, 1, 1, 1, 1]
        self.assertListEqual(utils.get_cross_points(y), [])

        y = [1, 1, 1, 1, 1, -1, 1]
        self.assertListEqual(utils.get_cross_points(y, pad=False), [4, 5])
        self.assertListEqual(utils.get_cross_points(y, pad=True), [0, 4, 5, 6])

        y = [1, -1, 1]
        self.assertListEqual(utils.get_cross_points(y, pad=False), [0, 1])
        self.assertListEqual(utils.get_cross_points(y, pad=True), [0, 1, 2])

        y = [1, 1, 1, 1, -1, -1, -1, -1]
        self.assertListEqual(utils.get_cross_points(y, pad=True), [0, 3, 7])

    def test_find_peaks_derivative(self):
        import pickle
        data = pickle.load(open(os.path.join(self.data_dir, 'peak_data.pickle')))
        x, y = data['large_range']
        peaks = utils.find_peaks_derivative(x, y)
        peaks = peaks.values()[0]

        np.testing.assert_array_equal(
            peaks['peaks'],
            np.array([357, 377, 432, 1668, 1754, 1811, 1835, 1911, 2008, 2398, 2576, 2952, 3170])
        )
        np.testing.assert_array_equal(
            peaks['minima'],
            np.array([337, 364, 364, 423, 423, 694, 1654, 1679, 1730, 1775, 1797,
                      1826, 1826, 1845, 1901, 1930, 1992, 2025, 2370, 2410, 2484, 2610,
                      2905, 2997, 3156, 3188]),
        )

    def test_interpolate_data(self):
        y = [0, 0, 0, 6311371.403331924, 24368020.237973947, 33309587.186450623, 0, 0, 22678022.890094325,
             12544950.520046625, 9621327.844190728, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        x = [23.6401, 23.6813, 23.7225, 23.7650, 23.8063, 23.8484, 23.8905, 23.9343, 23.9759, 24.0183, 24.0602, 24.1009,
             24.1440, 24.1877, 24.2282, 24.2738, 24.3195, 24.3662, 24.4168, 24.4607, 24.5016, 24.5417, 24.5812, 24.6235,
             24.7088, 24.7942]
        interp_y = utils.interpolate_data(x, y, gap_limit=2)
        self.assertNotEqual(interp_y[6], 0)
        self.assertNotEqual(interp_y[7], 0)
        self.assertItemsEqual(interp_y[:3], [0,0,0])
        interp_y = utils.interpolate_data(x, y, gap_limit=1)
        self.assertItemsEqual(interp_y[6:8], [0,0])

    def test_merge_close_peaks(self):
        ty = np.array([0, 1, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 3, 3])
        merged = utils.merge_close_peaks(np.array([7, 12]), ty, distance=6)
        np.testing.assert_array_equal(merged, np.array([7, 12]))

        merged = utils.merge_close_peaks(np.array([2, 7, 12]), ty, distance=5)
        np.testing.assert_array_equal(merged, np.array([2, 7, 12]))

        ty = np.array([0, 1, 2, 1, 0, 1, 2, 4, 2, 1, 0, 1, 3, 3])
        merged = utils.merge_close_peaks(np.array([2, 7, 12]), ty, distance=6)
        np.testing.assert_array_equal(merged, np.array([7]))


if __name__ == '__main__':
    unittest.main()

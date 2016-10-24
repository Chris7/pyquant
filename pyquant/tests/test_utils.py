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


if __name__ == '__main__':
    unittest.main()

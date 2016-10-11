__author__ = 'chris'
import unittest

from pyquant import utils
from pyquant.tests.mixins import GaussianMixin

class UtilsTests(GaussianMixin, unittest.TestCase):
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
        self.assertListEqual(pep_comp.values.tolist(), [0.6411550319843632, 0.2662471681269686, 0.07401847648709056, 0.015434213671511215, 0.002681646815294711])
        self.assertListEqual(ele_comp.values.tolist(), [0.9254949240653104, 0.07205572209608584, 0.002404285974894674])


if __name__ == '__main__':
    unittest.main()

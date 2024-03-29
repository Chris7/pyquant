__author__ = "chris"
import pickle
import os
import unittest

import numpy as np
import pandas as pd
from nose.plugins.attrib import attr

from pyquant import utils
from .mixins import GaussianMixin


class UtilsTests(GaussianMixin, unittest.TestCase):
    def setUp(self):
        super(UtilsTests, self).setUp()
        self.base_dir = os.path.split(os.path.abspath(__file__))[0]
        self.data_dir = os.path.join(self.base_dir, "data")

    def test_savgol_smooth(self):
        y = np.array([0, 1, 1.5, 0])
        new_y = utils.savgol_smooth(y)
        self.assertAlmostEqual(new_y[1], 1.0)
        self.assertAlmostEqual(new_y[2], 1.5)

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
        # Make sure we don't break if a float is passed
        selection = utils.select_window(x, 8, 20.0)
        self.assertListEqual(selection, x)

    def test_divide_peaks(self):
        chunks = utils.divide_peaks(self.one_gauss)
        two_gauss_chunks = utils.divide_peaks(self.two_gauss, chunk_factor=1.0)
        self.assertEqual(len(chunks), 0)
        self.assertEqual(len(two_gauss_chunks), 1)
        self.assertEqual(two_gauss_chunks[0], 65)

    def test_calculate_theoretical_distribution(self):
        peptide = "PEPTIDE"
        pep_comp = utils.calculate_theoretical_distribution(peptide=peptide)
        ele_comp = utils.calculate_theoretical_distribution(
            elemental_composition={"C": 7}
        )
        np.testing.assert_almost_equal(
            pep_comp.values.tolist(),
            [
                0.6411550319843632,
                0.2662471681269686,
                0.07401847648709056,
                0.015434213671511215,
                0.002681646815294711,
            ],
        )
        np.testing.assert_almost_equal(
            ele_comp.values.tolist(),
            [0.9254949240653104, 0.07205572209608584, 0.002404285974894674],
        )

    @attr("linux")
    def test_ml(self):
        data = os.path.join(self.data_dir, "ml_data.tsv")
        dat = pd.read_table(data)
        labels = ["Heavy", "Medium", "Light"]
        utils.perform_ml(dat, {i: [] for i in labels})
        for label1 in labels:
            for label2 in labels:
                if label1 == label2:
                    continue
                col = "{}/{} Confidence".format(label1, label2)
                self.assertNotEqual(sum(pd.isnull(dat["Heavy/Light Confidence"])), 0)

    def test_merge_peaks(self):
        peaks = {
            1: {"minima": [0, 1, 2, 4, 5], "peaks": [3]},
            2: {"minima": [0, 1, 2, 4, 5], "peaks": [3]},
            7: {"minima": [0, 1, 2, 4, 5], "peaks": [3]},
        }
        merged = utils.merge_peaks(peaks)
        self.assertDictEqual(merged, {7: {"minima": [0, 1, 2, 4, 5], "peaks": [3]}})

        peaks = {1: {"minima": [0, 1, 2, 4, 5], "peaks": [3]}}
        merged = utils.merge_peaks(peaks)
        self.assertDictEqual(merged, {1: {"minima": [0, 1, 2, 4, 5], "peaks": [3]}})

        peaks = {
            1: {"minima": [0, 5], "peaks": [3, 7, 8]},
            2: {"minima": [0, 5], "peaks": [3, 7]},
            7: {"minima": [0, 5], "peaks": [3, 7]},
        }
        merged = utils.merge_peaks(peaks)
        self.assertDictEqual(
            merged,
            {
                1: {"minima": [0, 5], "peaks": [3, 7, 8]},
                7: {"minima": [0, 5], "peaks": [3, 7]},
            },
        )

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
        with open(os.path.join(self.data_dir, "peak_data.pickle"), "rb") as peak_file:
            data = pickle.load(peak_file, encoding="latin1")

        x, y = data["large_range"]
        peaks = utils.find_peaks_derivative(x, y, smooth=False)
        peaks = next(iter(peaks.values()))

        np.testing.assert_array_equal(
            peaks["peaks"],
            np.array(
                [
                    357,
                    378,
                    432,
                    1668,
                    1755,
                    1811,
                    1835,
                    1912,
                    2009,
                    2399,
                    2577,
                    2952,
                    3171,
                ]
            ),
        )
        np.testing.assert_array_equal(
            peaks["minima"],
            np.array(
                [
                    337,
                    366,
                    366,
                    423,
                    423,
                    667,
                    1654,
                    1678,
                    1732,
                    1774,
                    1798,
                    1825,
                    1825,
                    1844,
                    1901,
                    1930,
                    1992,
                    2024,
                    2369,
                    2409,
                    2543,
                    2608,
                    2905,
                    2996,
                    3155,
                    3187,
                ]
            ),
        )

    def test_interpolate_data(self):
        y = [
            0,
            0,
            0,
            6311371.403331924,
            24368020.237973947,
            33309587.186450623,
            0,
            0,
            22678022.890094325,
            12544950.520046625,
            9621327.844190728,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        x = [
            23.6401,
            23.6813,
            23.7225,
            23.7650,
            23.8063,
            23.8484,
            23.8905,
            23.9343,
            23.9759,
            24.0183,
            24.0602,
            24.1009,
            24.1440,
            24.1877,
            24.2282,
            24.2738,
            24.3195,
            24.3662,
            24.4168,
            24.4607,
            24.5016,
            24.5417,
            24.5812,
            24.6235,
            24.7088,
            24.7942,
        ]
        interp_y = utils.interpolate_data(x, y, gap_limit=2)
        self.assertNotEqual(interp_y[6], 0)
        self.assertNotEqual(interp_y[7], 0)
        self.assertCountEqual(interp_y[:3], [0, 0, 0])
        interp_y = utils.interpolate_data(x, y, gap_limit=1)
        self.assertCountEqual(interp_y[6:8], [0, 0])

    def test_merge_close_peaks(self):
        ty = np.array([0, 1, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 3, 3])
        merged = utils.merge_close_peaks(np.array([7, 12]), ty, distance=6)
        np.testing.assert_array_equal(merged, np.array([7, 12]))

        merged = utils.merge_close_peaks(np.array([2, 7, 12]), ty, distance=5)
        np.testing.assert_array_equal(merged, np.array([2, 7, 12]))

        ty = np.array([0, 1, 2, 1, 0, 1, 2, 4, 2, 1, 0, 1, 3, 3])
        merged = utils.merge_close_peaks(np.array([2, 7, 12]), ty, distance=6)
        np.testing.assert_array_equal(merged, np.array([7]))

        merged = utils.merge_close_peaks(np.array([]), ty, distance=6)
        np.testing.assert_array_equal(merged, np.array([]))

    def test_rel_extrema(self):
        from scipy.signal import argrelextrema

        y = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                720001.9375,
                3056311.5,
                5417428.5,
                5494928.0,
                2185260.5,
                4099960.25,
                6359202.5,
                7897908.5,
                5956936.5,
                3124577.75,
                533339.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                554049.75,
                2336927.0,
                4342714.5,
                4671674.0,
                3267909.5,
                2324766.75,
                3356381.25,
                5258928.5,
                6305622.5,
                4235590.0,
                2183502.75,
                306638.96875,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        self.assertTrue(
            np.array_equal(
                argrelextrema(y, np.greater), utils.argrelextrema(y, np.greater)
            )
        )

    def test_get_formatted_mass(self):
        self.assertEqual(
            utils.get_formatted_mass("0.123"), utils.get_formatted_mass(0.123)
        )
        self.assertEqual(
            utils.get_formatted_mass("0.12300"), utils.get_formatted_mass(0.123)
        )
        self.assertEqual(
            utils.get_formatted_mass("123.12300"), utils.get_formatted_mass(123.123)
        )
        self.assertEqual(
            utils.get_formatted_mass("123.12300"), utils.get_formatted_mass(123.1230000)
        )

    def test_get_scan_resolution(self):
        with open(os.path.join(self.data_dir, "peak_data.pickle"), "rb") as peak_file:
            data = pickle.load(peak_file, encoding="latin1")

        x, y = data["low_res_scan"]
        scan = pd.Series(y, index=x)
        resolution = utils.get_scan_resolution(scan)
        self.assertAlmostEqual(int(resolution), 30459)

        x = np.array(
            [
                805.47264226,
                805.47265226,
                805.47495304,
                805.47739824,
                805.47984345,
                805.49206968,
                805.49451496,
                805.49696025,
                805.49940556,
                805.50185087,
                805.50429619,
                805.50674153,
                805.50918687,
                805.51163223,
                805.51407748,
                805.51652286,
                805.51896825,
                805.52141365,
                805.7635636,
                805.76601011,
                805.76845664,
                805.77090318,
                805.77334973,
                805.77579629,
                805.77824286,
                805.78068944,
                805.78313604,
                805.78558264,
                805.79153526,
                805.79398188,
                805.79642852,
                805.79887517,
                805.80132183,
                805.8037685,
                805.80621519,
                805.80622519,
                805.80623519,
                805.80661184,
                805.80905857,
                806.09295462,
                806.09540266,
                806.0978507,
                806.10029875,
                806.10274682,
                806.1051949,
                806.10764298,
                806.11009108,
                806.11253919,
                806.11498731,
                806.11743545,
                806.12539827,
                806.12784643,
                806.13029459,
                806.13274277,
                806.13519096,
                806.13763916,
                806.14008737,
                806.14009737,
                806.14206697,
                806.14451522,
            ]
        )
        y = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                720001.9375,
                3056311.5,
                5417428.5,
                5494928.0,
                2185260.5,
                4099960.25,
                6359202.5,
                7897908.5,
                5956936.5,
                3124577.75,
                533339.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                554049.75,
                2336927.0,
                4342714.5,
                4671674.0,
                3267909.5,
                2324766.75,
                3356381.25,
                5258928.5,
                6305622.5,
                4235590.0,
                2183502.75,
                306638.96875,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

        scan = pd.Series(y, index=x)
        resolution = utils.get_scan_resolution(scan)
        self.assertAlmostEqual(int(resolution), 90781)

    def test_subtract_baseline(self):
        # This test data is a simple sin function with a parabola function applied to the ends,
        # giving a smile shape.

        x = np.linspace(-50, 50)
        noise = 0.00125 * x ** 2
        c1 = np.sin(x)
        y = noise + c1

        baselined = utils.subtract_baseline(y)

        # Assert we're about zero and the noise (the parabola) has been removed
        self.assertTrue(sum(baselined - c1) < 1e-10)


class TestGetScansUnderPeak(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super(TestGetScansUnderPeak, cls).setUpClass()
        cls.rt_scan_map = pd.Series({1 + i / 10.0: i for i in range(30)})
        cls.found_peaks = {
            0: {
                0: {"peak_mean": 2, "std": 0.3, "std2": 0.2,},
                1: {"peak_mean": 1.3, "std": 0.2, "std2": 0.45,},
            }
        }

    def test_gets_peaks_under_scan(self):
        scans = utils.get_scans_under_peaks(self.rt_scan_map, self.found_peaks)
        self.assertEqual(
            scans,
            {
                0: {
                    0: {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14},
                    1: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                }
            },
        )


class TestGetScanIdFromRt(unittest.TestCase):
    def test_get_scan_id_from_rt(self):
        scan_map = pd.Series({"id 1": 1.2, "id 2": 1.3, "id 3": 2.3})
        self.assertEqual(utils.get_scan_id_from_rt(scan_map, 1.0), "id 1")
        self.assertEqual(utils.get_scan_id_from_rt(scan_map, 1.2), "id 1")
        self.assertEqual(utils.get_scan_id_from_rt(scan_map, 1.25), "id 2")
        self.assertEqual(utils.get_scan_id_from_rt(scan_map, 10), "id 3")


if __name__ == "__main__":
    unittest.main()

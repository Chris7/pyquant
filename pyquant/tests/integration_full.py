__author__ = 'chris'
import os
import unittest
import pandas as pd
import numpy as np
import subprocess

from pyquant.tests import mixins, utils, config


class EColiTest(mixins.FileMixins, unittest.TestCase):
    def setUp(self):
        super(EColiTest, self).setUp()
        self.output = os.path.join(self.out_dir, 'pqtest2')
        self.output_stats = '{}_stats'.format(self.output)
        self.r_std = 0.5
        self.k_std = 0.7

    @utils.timer
    def test_pyquant(self):
        com = [self.executable, '--search-file', self.ecoli_search_file, '--scan-file', self.ecoli_mzml, '-p', str(config.CORES), '-o', self.output, '--html', '--precursor-ppm', '2', '--xic-window-size', '12']
        subprocess.call(com)
        pyquant = pd.read_table(self.output)
        label = 'Medium'
        pq_sel = '{}/Light'.format(label)
        pyquant.loc[((pyquant['Peptide'].str.upper().str.count('R')==1) & (pyquant['Peptide'].str.upper().str.count('K')==0)),'Class'] = 'R'
        pyquant.loc[((pyquant['Peptide'].str.upper().str.count('K')==1) & (pyquant['Peptide'].str.upper().str.count('R')==0)),'Class'] = 'K'
        pyquant[pq_sel] = np.log2(pyquant[pq_sel]+0.000001)
        # the median is robust, we care about the standard deviation since changes to the backend can alter the peak width
        r_std = np.std(pyquant.loc[pyquant['Class'] == 'R', pq_sel])
        k_std = np.std(pyquant.loc[pyquant['Class'] == 'K', pq_sel])
        self.assertLess(r_std, self.r_std)
        self.assertLess(k_std, self.k_std)
        label = 'Heavy'
        pq_sel = '{}/Light'.format(label)
        pyquant[pq_sel] = np.log2(pyquant[pq_sel]+0.000001)
        # the median is robust, we care about the standard deviation since changes to the backend can alter the peak width
        r_stdh = np.std(pyquant.loc[pyquant['Class'] == 'R', pq_sel])
        k_stdh = np.std(pyquant.loc[pyquant['Class'] == 'K', pq_sel])


class ITraqTest(mixins.FileMixins, unittest.TestCase):
    def setUp(self):
        super(ITraqTest, self).setUp()
        self.output = os.path.join(self.out_dir, 'itraq_test')

    @utils.timer
    def test_itraq_processing(self):
        com = [self.executable, '--scan-file', self.itraq_mzml, '-o', self.output, '--precursor-ppm', '200', '--isobaric-tags', '--label-method', 'iTRAQ4']
        subprocess.call(com)
        data = pd.read_table(self.output)
        self.assertListEqual(map(int, data['114 Intensity'].values.tolist()), [1215, 9201, 1218, 83983, 10266, 2995, 7160])
        self.assertListEqual(map(int, data['115 Intensity'].values.tolist()), [1428, 38772, 946, 1161, 12032, 4109, 8421])
        self.assertListEqual(map(int, data['116 Intensity'].values.tolist()), [1031, 15314, 0, 94243, 11381, 3350, 8577])
        self.assertListEqual(map(int, data['117 Intensity'].values.tolist()), [0, 42004, 1032, 383, 14290, 2557, 8969])

if __name__ == '__main__':
    unittest.main()

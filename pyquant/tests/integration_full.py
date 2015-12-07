__author__ = 'chris'
from unittest import TestCase
import pandas as pd
import multiprocessing
import os
import numpy as np
import subprocess


class EColiTest(TestCase):
    def setUp(self):
        self.executable = os.path.abspath(os.path.join(__file__, '..', '..', '..', 'scripts', 'pyQuant'))
        self.search_file = "/media/chris/data/Dropbox/Manuscripts/SILAC Fix/EColi/PD/SILAC 1_2_4.msf"
        self.mzml =  "/media/chris/data/Dropbox/Manuscripts/SILAC Fix/EColi/Chris_Ecoli_1-2-4.mzML"
        self.out_dir = 'pq_tests'
        try:
            os.mkdir(self.out_dir)
        except OSError:
            pass
        self.output = os.path.join(self.out_dir, 'pqtest')
        self.output_stats = os.path.join(self.out_dir, 'pqtest_stats')
        self.r_std = 0.5
        self.k_std = 0.7

    def tearDown(self):
        os.remove(self.output)
        os.remove(self.output_stats)
        os.rmdir(self.out_dir)

    def test_pyquant(self):
        com = [self.executable, '--search-file', self.search_file, '--scan-file', self.mzml, '-p', str(multiprocessing.cpu_count()), '-o', self.output]
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
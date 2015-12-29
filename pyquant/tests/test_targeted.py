import os
import subprocess
from unittest import TestCase

from .mixins import FileMixins
from . import config

class TestTargeted(FileMixins, TestCase):
    def setUp(self):
        super(TestTargeted, self).setUp()
        self.output = os.path.join(self.out_dir, 'targeted')

    def test_pyquant_immonium(self):
        com = [self.executable, '--scan-file', self.mzml, '-p', str(config.CORES), '-o', self.output, '--msn-ion', '216.043']
        subprocess.call(com)
        # pyquant = pd.read_table(self.output)
        # label = 'Medium'
        # pq_sel = '{}/Light'.format(label)
        # pyquant.loc[((pyquant['Peptide'].str.upper().str.count('R')==1) & (pyquant['Peptide'].str.upper().str.count('K')==0)),'Class'] = 'R'
        # pyquant.loc[((pyquant['Peptide'].str.upper().str.count('K')==1) & (pyquant['Peptide'].str.upper().str.count('R')==0)),'Class'] = 'K'
        # pyquant[pq_sel] = np.log2(pyquant[pq_sel]+0.000001)
        # # the median is robust, we care about the standard deviation since changes to the backend can alter the peak width
        # r_std = np.std(pyquant.loc[pyquant['Class'] == 'R', pq_sel])
        # k_std = np.std(pyquant.loc[pyquant['Class'] == 'K', pq_sel])
        # self.assertLess(r_std, self.r_std)
        # self.assertLess(k_std, self.k_std)
        # label = 'Heavy'
        # pq_sel = '{}/Light'.format(label)
        # pyquant[pq_sel] = np.log2(pyquant[pq_sel]+0.000001)
        # # the median is robust, we care about the standard deviation since changes to the backend can alter the peak width
        # r_stdh = np.std(pyquant.loc[pyquant['Class'] == 'R', pq_sel])
        # k_stdh = np.std(pyquant.loc[pyquant['Class'] == 'K', pq_sel])
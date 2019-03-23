__author__ = 'chris'
import unittest
import pandas as pd
import multiprocessing
import numpy as np
import os
import subprocess

from pyquant.tests.mixins import FileMixins
from pyquant.tests.utils import timer
from pyquant.tests import config


peps_scans = (
    # possible incomplete envelope identification but good fits
    'GQGYLFELR 5059',
    'IFLDASSEER 3147',
    'ATIGEVGNAEHMLR 2607',
    'YMANAMGPEGVR 1975',
    'VPDSQVLADLDHVASWASR 7203',
    'VPGTIGFATVR 3830',
    'VVANFLSSVGVDR 5699',
    'IHVAVAQEVPGTGVDTPEDLER 4157',
    'INAMLQDYELQR 4442',
    'AFIGIDGWQPETGFTGR 6770',
    'GMNVVFELR 5091',
    'ILHIQQQLAGEQVALSDEVNQSEQTTNFHNR 5404',
    'EQGYALDSEENEQGVR 2536',
    'VGSDNLLMINAHIAHDCTVGNR 4150',
    'FDGFVHSIGFAPGDQLDGDYVNAVTR 6357',
    'GIWNHGSPLFMEIEPR 6173',

    # bad data
    'VGYIELDLNSGK 6246',
    'FVESVDVAVNLGIDAR 9184',

    # bad fit
    'TIPSVLTALFCAR 8606',
    'VANLEAQLAEAQTR 3878',

    # uncertain
    'ITGIDSSPAMIAEAR 4149',

    # good fit
    'TVYSTENPDLLVLEFR 7102',
    'VMMIDEPAILDQAIAR 6949',
    'IIAIDNSPAMIER 4301',
    'IIAIDNSPAMIER 4307',
    'DAGNIIIDDDDISLLPLHAR 6921',
    'NASASAPTALPLR 3089',
    'SLEHEVTLVDDTLAR 4955',
    'MTELPIDENTPR 3494',
    'LAQALANPLFPALDSALR 7905',
    'SDWNPSLYLHFSAER 7079',

    # could be better at complete fit
    'GIWNHGSPLFMEIEPR 6175',
    'VANLEAQLAEAQTR 3835',
    'HVAIIMDGNGR 2069',
    'GAGEIVLNMMNQDGVR 6081',
    'ITGIDSSPAMIAEAR 4159',

    # could be better/maybe incomplete envelope
    'FDGFVHSIGFAPGDQLDGDYVNAVTR 6747',
    'DAGNIIIDDDDISLLPLHAR 6930',
    'MQQLQNIIETAFER 7023',
    'DQQIPLLISGGIGHSTTFLYSAIAQHPHYNTIR 7545',
)


class EColiManualTest(FileMixins, unittest.TestCase):
    def setUp(self):
        super(EColiManualTest, self).setUp()
        self.output = os.path.join(self.out_dir, 'pqtest_manual')
        self.output_stats = os.path.join(self.out_dir, 'pqtest_stats')

    @timer
    def test_pyquant_manual(self):
        com = [self.executable, '--search-file', self.ecoli_search_file, '--scan-file', self.ecoli_mzml, '--html', '-p', str(config.CORES), '-o', self.output, '--xic-window-size', '12']
        peptides, scans = zip(*map(lambda x: x.split(' '),peps_scans))
        com.extend(['--peptide'])
        com.extend(peptides)
        com.extend(['--scan'])
        com.extend(scans)
        subprocess.call(com)
        pyquant = pd.read_table(self.output)
        label = 'Medium'
        pq_sel = '{}/Light'.format(label)
        pyquant.loc[((pyquant['Peptide'].str.upper().str.count('R')==1) & (pyquant['Peptide'].str.upper().str.count('K')==0)),'Class'] = 'R'
        pyquant.loc[((pyquant['Peptide'].str.upper().str.count('K')==1) & (pyquant['Peptide'].str.upper().str.count('R')==0)),'Class'] = 'K'
        pyquant[pq_sel] = np.log2(pyquant[pq_sel]+0.000001)
        label = 'Heavy'
        pq_sel = '{}/Light'.format(label)
        pyquant[pq_sel] = np.log2(pyquant[pq_sel]+0.000001)


if __name__ == '__main__':
    unittest.main()

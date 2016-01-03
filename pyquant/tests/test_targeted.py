import os
import subprocess
from unittest import TestCase

import pandas as pd
import numpy as np

import tempfile

from .mixins import FileMixins
from . import config

class TestTargeted(FileMixins, TestCase):
    def setUp(self):
        super(TestTargeted, self).setUp()
        self.output = os.path.join(self.out_dir, 'targeted')

    def test_pyquant_trypsin(self):
        # This searches for the y1 ions of arginine. It also is a check that the label-scheme parameter works.
        f = tempfile.NamedTemporaryFile('wb')
        f.write('\t'.join(['0', 'R', '10.008269', 'R10'])+'\n')
        f.write('\t'.join(['1', 'R', '0', 'Light'])+'\n')
        f.seek(0)
        com = [self.executable, '--scan-file', self.mzml, '-p', str(config.CORES), '-o', self.output, '--msn-ion', '175', '--html', '--label-scheme', f.name]
        subprocess.call(com)
        pyquant = pd.read_table(self.output)
        label = 'R10'
        pq_sel = '{}/Light'.format(label)
        pyquant[pq_sel] = np.log2(pyquant[pq_sel]+0.000001)
        r10_med = pyquant[pq_sel].median()
        self.assertNotAlmostEqual(r10_med, -2.0, places=2)
import os
import subprocess
from unittest import TestCase

import pandas as pd
import numpy as np

import tempfile

from .mixins import FileMixins
from . import config

class TestConvenience(FileMixins, TestCase):
    def setUp(self):
        super(TestConvenience, self).setUp()
        self.output = os.path.join(self.out_dir, 'mq')
        self.mq_file = os.path.join(self.data_dir, 'ecoli_mq_124_ms2.txt')

    def test_mq_convenience(self):
        com = [self.executable, '--scan-file', self.ecoli_mzml, '-p', str(config.CORES), '-o', self.output, '--maxquant', '--tsv', self.mq_file, '--html', '--label-method', 'K4K8R6R10', '--sample', '0.1']
        subprocess.call(com)
        pyquant = pd.read_table(self.output)
        label = 'Heavy'
        pq_sel = '{}/Light'.format(label)
        pyquant[pq_sel] = np.log2(pyquant[pq_sel]+0.000001)
        r10_med = pyquant[pq_sel].median()
        self.assertNotAlmostEqual(r10_med, -2.0, places=2)

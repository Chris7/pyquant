import os
import unittest
import pandas as pd
import subprocess

from pyquant.tests import mixins, utils


class ITraqTest(mixins.FileMixins, unittest.TestCase):
    def setUp(self):
        super(ITraqTest, self).setUp()
        self.labels = os.path.join(self.data_dir, 'itraq_labels.tsv')
        self.output = os.path.join(self.out_dir, 'itraq_test')

    @utils.timer
    def test_itraq_processing(self):
        com = [self.executable, '--scan-file', self.itraq_mzml, '-o', self.output, '--precursor-ppm', '200', '--isobaric-tags', '--label-scheme', self.labels]
        subprocess.call(com)
        data = pd.read_table(self.output)
        for i,j in zip([int(i) for i in data['114 Intensity'].values.tolist()], [1215, 9201, 1218, 83983, 10266, 2995, 7160]):
            delta = j*0.1
            if delta != 0 and delta < 300:
                delta = 300
            self.assertAlmostEqual(i, j, delta=delta)
        for i, j in zip([int(i) for i in data['115 Intensity'].values.tolist()], [1428, 38772, 946, 4041, 12032, 4109, 8421]):
            delta = j * 0.1
            if delta != 0 and delta < 300:
                delta = 300
            self.assertAlmostEqual(i, j, delta=delta)
        for i, j in zip([int(i) for i in data['116 Intensity'].values.tolist()], [1031, 15314, 0, 94729, 11381, 3350, 8577]):
            delta = j * 0.1
            if delta != 0 and delta < 300:
                delta = 300
            self.assertAlmostEqual(i, j, delta=delta)
        for i, j in zip([int(i) for i in data['117 Intensity'].values.tolist()], [0, 42004, 1032, 164251, 14290, 2557, 8969]):
            delta = j * 0.1
            if delta != 0 and delta < 300:
                delta = 300
            self.assertAlmostEqual(i, j, delta=delta)

if __name__ == '__main__':
    unittest.main()

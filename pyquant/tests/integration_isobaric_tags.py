import os
import unittest
import pandas as pd
import subprocess

from pyquant.tests import mixins, utils


class ITraqTest(mixins.FileMixins, unittest.TestCase):
    def setUp(self):
        super(ITraqTest, self).setUp()
        self.output = os.path.join(self.out_dir, 'itraq_test')

    @utils.timer
    def test_itraq_processing(self):
        com = [self.executable, '--scan-file', self.itraq_mzml, '-o', self.output, '--precursor-ppm', '200', '--isobaric-tags', '--label-method', 'iTRAQ4']
        subprocess.call(com)
        data = pd.read_table(self.output)
        self.assertListEqual([int(i) for i in data['114 Intensity'].values.tolist()], [1215, 9201, 1218, 83983, 10266, 2995, 7160])
        self.assertListEqual([int(i) for i in data['115 Intensity'].values.tolist()], [1428, 38772, 946, 1161, 12032, 4109, 8421])
        self.assertListEqual([int(i) for i in data['116 Intensity'].values.tolist()], [1031, 15314, 0, 94243, 11381, 3350, 8577])
        self.assertListEqual([int(i) for i in data['117 Intensity'].values.tolist()], [0, 42004, 1032, 383, 14290, 2557, 8969])

if __name__ == '__main__':
    unittest.main()

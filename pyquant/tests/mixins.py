import os

import numpy as np

from pyquant import peaks

class FileMixins(object):
    def setUp(self):
        super(FileMixins, self).setUp()
        self.base_dir = os.path.split(os.path.abspath(__file__))[0]
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.executable = os.path.abspath(os.path.join(self.base_dir, '..', '..', 'scripts', 'pyQuant'))
        self.ecoli_search_file = os.path.join(self.data_dir, 'Chris_Ecoli_1-2-4-(01).msf')
        self.ecoli_mzml = os.path.join(self.data_dir, 'Chris_Ecoli_1-2-4.mzML')
        self.itraq_mzml = os.path.join(self.data_dir, 'iTRAQ_Data.mzML')
        self.out_dir = 'pq_tests'
        try:
            os.mkdir(self.out_dir)
        except OSError:
            pass

    # def tearDown(self):
    #     os.remove(self.output)
    #     os.remove(self.output_stats)
    #     os.rmdir(self.out_dir)

class GaussianMixin(object):
    def setUp(self):
        self.amp, self.mu, self.std, self.mu2 = 1., 0., 1., 3.
        self.one_gauss_params = np.array([self.amp, self.mu, self.std], dtype=np.float)
        self.two_gauss_params = np.array([self.amp, self.mu, self.std, self.amp, self.mu2, self.std], dtype=np.float)
        self.x = np.array(np.linspace(-5, 5, 101), dtype=np.float)
        self.one_gauss = peaks.gauss_ndim(self.x, self.one_gauss_params)
        self.two_gauss = peaks.gauss_ndim(self.x, self.two_gauss_params)
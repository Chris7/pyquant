import pickle
import os

import numpy as np
import six

BASE_DIR = os.path.split(os.path.abspath(__file__))[0]
DATA_DIR = os.path.join(BASE_DIR, 'data')

class FileMixins(object):
    def setUp(self):
        super(FileMixins, self).setUp()
        self.base_dir = BASE_DIR
        self.data_dir = DATA_DIR
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
        with open(os.path.join(DATA_DIR, 'peak_data.pickle'), 'rb') as peak_file:
            self.peak_data = pickle.load(peak_file, encoding='latin1') if six.PY3 else pickle.load(peak_file)
        self.x = self.peak_data['one_gauss'][0]
        self.one_gauss = self.peak_data['one_gauss'][1]
        self.two_gauss = self.peak_data['two_gauss'][1]
        self.noisy_two_gauss = self.peak_data['noisy_two_gauss'][1]

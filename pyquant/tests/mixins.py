import os

class FileMixins(object):
    def setUp(self):
        super(FileMixins, self).setUp()
        base_dir = os.path.split(os.path.abspath(__file__))[0]
        self.executable = os.path.abspath(os.path.join(base_dir, '..', '..', 'scripts', 'pyQuant'))
        self.search_file = os.path.join(base_dir, 'data', 'SILAC_1_2_4.msf')
        self.mzml =  os.path.join(base_dir, 'data', 'Chris_Ecoli_1-2-4.mzML')
        self.out_dir = 'pq_tests'
        try:
            os.mkdir(self.out_dir)
        except OSError:
            pass

    # def tearDown(self):
    #     os.remove(self.output)
    #     os.remove(self.output_stats)
    #     os.rmdir(self.out_dir)
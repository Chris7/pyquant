import os

class FileMixins(object):
    def setUp(self):
        super(FileMixins, self).setUp()
        self.base_dir = os.path.split(os.path.abspath(__file__))[0]
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.executable = os.path.abspath(os.path.join(self.base_dir, '..', '..', 'scripts', 'pyQuant'))
        self.search_file = os.path.join(self.data_dir, 'Chris_Ecoli_1-2-4-(01).msf')
        self.mzml =  os.path.join(self.data_dir, 'Chris_Ecoli_1-2-4.mzML')
        self.out_dir = 'pq_tests'
        try:
            os.mkdir(self.out_dir)
        except OSError:
            pass

    # def tearDown(self):
    #     os.remove(self.output)
    #     os.remove(self.output_stats)
    #     os.rmdir(self.out_dir)

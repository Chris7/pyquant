__author__ = 'chris'
import time
from unittest import TestCase
from multiprocessing import Queue

from pyquant.reader import Reader
from pyquant.tests import mixins

class TestReader(mixins.FileMixins, TestCase):
    def setUp(self):
        super(TestReader, self).setUp()
        self.input = Queue()

    def tearDown(self):
        super(TestReader, self).tearDown()
        self.input.put(None)

    def test_reader(self):
        output = {0: Queue()}
        reader = Reader(self.input, output, raw_file=self.ecoli_mzml)
        reader.start()
        queue = output[0]
        self.input.put((0, '80', 0, 10000))
        # Test we get the scan
        scan = queue.get()
        self.assertEqual(scan['title'], '80')
        # test we get a subset
        mz_min, mz_max = scan['vals'][300, 0], scan['vals'][500, 0]
        self.input.put((0, '80', mz_min, mz_max))
        scan = queue.get()
        self.assertTrue(scan['vals'][0, 0] >= mz_min)
        self.assertTrue(scan['vals'][-1, 0] <= mz_max)

        # test we exit on a sentinel
        self.input.put(None)
        # wait for the process to end
        time.sleep(0.5)
        self.assertFalse(reader.is_alive())

        # test that we can get scans that have been deleted
        reader = Reader(self.input, output, raw_file=self.ecoli_mzml, timeout_minutes=0.01)
        reader.start()
        self.input.put((0, '80', 0, 10000))
        self.input.put((0, '81', 0, 10000))
        scan = queue.get()
        scan = queue.get()
        time.sleep(1)
        # accessing 81 should cause 80 to be deleted
        self.input.put((0, '81', 0, 10000))
        scan = queue.get()
        # make sure we can get it back
        self.input.put((0, '80', 0, 10000))
        scan = queue.get()
        self.assertEqual(scan['title'], '80')
import time
import unittest
from multiprocessing import Queue

from pyquant.reader import Reader

from . import mixins


class TestReader(mixins.FileMixins, unittest.TestCase):
    def setUp(self):
        super(TestReader, self).setUp()
        self.input = Queue()
        self.output = {0: Queue()}
        self.reader = Reader(self.input, self.output, raw_file=self.small_mzml)
        self.addCleanup(lambda: self.input.put(None))
        self.reader.start()
        self.queue = self.output[0]

    def test_fetches_scan(self):
        self.input.put((0, "S19", 0, 10000))
        # Test we get the scan
        scan = self.queue.get()
        self.assertEqual(scan["title"], "S19")

    def test_fetches_subset(self):
        # test we get a subset
        self.input.put((0, "S19", 0, 10000))
        scan = self.queue.get()
        mz_min, mz_max = scan["vals"][2, 0], scan["vals"][5, 0]
        self.input.put((0, "S19", mz_min, mz_max))
        scan = self.queue.get()
        self.assertTrue(scan["vals"][0, 0] >= mz_min)
        self.assertTrue(scan["vals"][-1, 0] <= mz_max)

    def test_exits_on_sentinel(self):
        # test we exit on a sentinel
        self.input.put(None)
        # wait for the process to end
        time.sleep(0.5)
        self.assertFalse(self.reader.is_alive())

    def test_refetches_deleted_scans(self):
        # test that we can get scans that have been deleted
        self.reader.timeout_minutes = 0.01
        self.input.put((0, "S19", 0, 10000))
        self.input.put((0, "S20", 0, 10000))
        scan = self.queue.get()
        scan = self.queue.get()
        time.sleep(1)
        # accessing 81 should cause 80 to be deleted
        self.input.put((0, "S20", 0, 10000))
        scan = self.queue.get()
        # make sure we can get it back
        self.input.put((0, "S19", 0, 10000))
        scan = self.queue.get()
        self.assertEqual(scan["title"], "S19")


if __name__ == "__main__":
    unittest.main()

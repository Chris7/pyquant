__author__ = "chris"
import os
import unittest
import pandas as pd
import numpy as np
import subprocess

from . import mixins, utils, config


class EColiTest(mixins.FileMixins, unittest.TestCase):
    def setUp(self):
        super(EColiTest, self).setUp()
        self.output = os.path.join(self.out_dir, "pqtest2")
        self.output_stats = "{}_stats".format(self.output)
        self.r_std = 0.6
        self.k_std = 0.9

    @utils.timer
    def test_pyquant(self):
        com = [
            self.executable,
            "--search-file",
            self.ecoli_search_file,
            "--scan-file",
            self.ecoli_mzml,
            "-p",
            str(config.CORES),
            "-o",
            self.output,
            "--html",
            "--precursor-ppm",
            "2",
            "--xic-window-size",
            "12",
        ]
        subprocess.call(com)
        pyquant = pd.read_table(self.output)
        label = "Medium"
        pq_sel = "{}/Light".format(label)
        pyquant.loc[
            (
                (pyquant["Peptide"].str.upper().str.count("R") == 1)
                & (pyquant["Peptide"].str.upper().str.count("K") == 0)
            ),
            "Class",
        ] = "R"
        pyquant.loc[
            (
                (pyquant["Peptide"].str.upper().str.count("K") == 1)
                & (pyquant["Peptide"].str.upper().str.count("R") == 0)
            ),
            "Class",
        ] = "K"
        pyquant[pq_sel].apply(lambda x: np.log2(x) if x != 0 else x)
        # Mark zeros as nan to exclude from calculation
        pyquant.loc[pyquant[pq_sel] == 0, pq_sel] = np.NaN
        # the median is robust, we care about the standard deviation since changes to the backend can alter the peak width
        r_std = np.std(pyquant.loc[pyquant["Class"] == "R", pq_sel])
        k_std = np.std(pyquant.loc[pyquant["Class"] == "K", pq_sel])
        self.assertLess(r_std, self.r_std)
        self.assertLess(k_std, self.k_std)


if __name__ == "__main__":
    unittest.main()

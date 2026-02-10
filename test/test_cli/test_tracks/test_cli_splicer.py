import os
from unittest.mock import patch

import sys

import time

from pathlib import Path

import cli
from utils.test_base import TestBase


"""This file tests different CLI operations and checks for the expected outcome"""


class TestCliSplicer(TestBase):

    def run_cmd(self, other_params=None):
        testargs = [
            Path(__file__).parent.parent.parent.parent / 'src' / 'toolkit',
            "tracks",
            "splice",
            "--src", self.get_lapsim_data_path() / 'optimal_tracks',
            "--dest", self.get_temp_output_path()
        ] + (other_params if other_params else [])

        with patch.object(sys, 'argv', [str(x) for x in testargs]):
            cli.parse()

    def test_basic_export(self):
        """Test by default cli works"""
        self.get_temp_output_path().mkdir(exist_ok=True, parents=True)
        self.run_cmd()
        self.assertEqual(9, len(os.listdir(self.get_temp_output_path())))

    def test_portion_export(self):
        """Test that using portion discards data and that seeding leads o consistence results"""
        self.get_temp_output_path().mkdir(exist_ok=True, parents=True)
        self.run_cmd(["--portion", "0.5", "--seed", "1"])
        run_1_files = os.listdir(self.get_temp_output_path())
        self.assertEqual(5, len(run_1_files))
        self.clear_temp_dir()

        self.get_temp_output_path().mkdir(exist_ok=True, parents=True)
        self.run_cmd(["--portion", "0.5", "--seed", "10"])
        run_2_files = os.listdir(self.get_temp_output_path())
        self.clear_temp_dir()

        self.get_temp_output_path().mkdir(exist_ok=True, parents=True)
        self.run_cmd(["--portion", "0.5", "--seed", "10"])
        run_3_files = os.listdir(self.get_temp_output_path())

        # Test that the lists are differen twith a different seed
        self.assertTrue(set(run_1_files).symmetric_difference(set(run_2_files)))

        # Run 2 and 3 used same seed so should be same files
        self.assertListEqual(run_2_files, run_3_files)

    def test_spacing(self):
        """Test that increased spacing result in higher file size"""
        self.get_temp_output_path().mkdir(exist_ok=True, parents=True)

        self.run_cmd(["--portion", "0.1", "--seed", "1", "--spacing", "1"])
        run_1_files = os.listdir(self.get_temp_output_path())
        spacing_1_size = os.stat(self.get_temp_output_path() / run_1_files[0]).st_size

        self.run_cmd(["--portion", "0.1", "--seed", "1", "--spacing", "10"])
        run_1_files = os.listdir(self.get_temp_output_path())
        spacing_10_size = os.stat(self.get_temp_output_path() / run_1_files[0]).st_size

        # Calculate file size different
        self.assertLess(spacing_10_size, spacing_1_size)

    def test_cpu(self):
        """Test that four core time is quicker than one core"""
        start = time.time()
        self.run_cmd(["--cores", "1"])
        one_core_time = time.time() - start

        start = time.time()
        self.run_cmd(["--cores", "4"])
        four_core_time = time.time() - start

        self.assertLess(four_core_time, one_core_time)

    def test_precisions(self):
        """Test that decreased precision result in lower file size"""
        self.get_temp_output_path().mkdir(exist_ok=True, parents=True)

        self.run_cmd(["--portion", "0.1", "--seed", "1", "--precision", "1"])
        run_1_files = os.listdir(self.get_temp_output_path())
        precision_1_size = os.stat(self.get_temp_output_path() / run_1_files[0]).st_size

        self.run_cmd(["--portion", "0.1", "--seed", "1", "--precision", "10"])
        run_1_files = os.listdir(self.get_temp_output_path())
        precision_10_size = os.stat(self.get_temp_output_path() / run_1_files[0]).st_size

        # Calculate file size different
        self.assertLess(precision_1_size, precision_10_size)

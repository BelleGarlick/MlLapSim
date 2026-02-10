import os
import sys
from unittest.mock import patch

from pathlib import Path

import cli
from lapsim.encoder.partition import Partition
from utils.test_base import TestBase


"""This file tests different CLI operations and checks for the expected outcome"""


class TestCliEncoder(TestBase):

    def run_cmd(self, other_params=None):
        testargs = [
            Path(__file__).parent.parent.parent.parent / 'src' / 'toolkit',
            "lapsim",
            "encode",
            "--src", self.get_lapsim_data_path() / 'spliced',
            "--dest", self.get_temp_output_path()
        ] + (other_params if other_params else [])

        with patch.object(sys, 'argv', [str(x) for x in testargs]):
            cli.parse()

    def test_basic_export(self):
        """Test by default cli works"""
        self.get_temp_output_path().mkdir(parents=True, exist_ok=True)
        self.run_cmd()
        self.assertEqual(9, len(os.listdir(self.get_temp_output_path())))

    def test_portion_export(self):
        """Test that using portion discards data and that seeding leads o consistence results"""
        self.get_temp_output_path().mkdir(parents=True, exist_ok=True)
        self.run_cmd(["--portion", "0.5", "--seed", "1"])
        run_1_files = os.listdir(self.get_temp_output_path())
        self.assertEqual(5, len(run_1_files))
        self.clear_temp_dir()

        self.get_temp_output_path().mkdir(parents=True, exist_ok=True)
        self.run_cmd(["--portion", "0.5", "--seed", "10"])
        run_2_files = os.listdir(self.get_temp_output_path())
        self.clear_temp_dir()

        self.get_temp_output_path().mkdir(parents=True, exist_ok=True)
        self.run_cmd(["--portion", "0.5", "--seed", "10"])
        run_3_files = os.listdir(self.get_temp_output_path())

        # Test that the lists are differen twith a different seed
        self.assertTrue(set(run_1_files).symmetric_difference(set(run_2_files)))

        # Run 2 and 3 used same seed so should be same files
        self.assertListEqual(run_2_files, run_3_files)

    def test_n_partitions(self):
        """Test npartitions work from cli"""
        self.get_temp_output_path().mkdir(parents=True, exist_ok=True)
        self.run_cmd(['--partitions', '0'])
        self.assertEqual(9, len(os.listdir(self.get_temp_output_path())))

        self.clear_temp_dir()
        self.get_temp_output_path().mkdir(parents=True, exist_ok=True)
        self.run_cmd(['--partitions', '3'])
        self.assertEqual(3, len(os.listdir(self.get_temp_output_path())))

        self.clear_temp_dir()
        self.get_temp_output_path().mkdir(parents=True, exist_ok=True)
        self.run_cmd(['--partitions', '1'])
        self.assertEqual(1, len(os.listdir(self.get_temp_output_path())))

    def test_flip(self):
        """Test flipping from cli"""
        # Test flip for one partition
        self.get_temp_output_path().mkdir(parents=True, exist_ok=True)
        self.run_cmd(['--partitions', '1', '--flip'])
        partition = Partition.load(self.get_temp_output_path() / 'partition-0.json')
        self.assertEqual(18, len(partition.velocities))

        # Test flipping individual tracks
        self.clear_temp_dir()
        self.get_temp_output_path().mkdir(parents=True, exist_ok=True)
        self.run_cmd(['--flip'])
        self.assertEqual(18, len(os.listdir(self.get_temp_output_path())))
        reg_partition = Partition.load(self.get_temp_output_path() / '100586536.json')
        flip_partition = Partition.load(self.get_temp_output_path() / '100586536-flipped.json')
        self.assertEqual(len(reg_partition.positions), len(flip_partition.positions))

        # Assets values are flipped in individual items
        self.assertEqual(1, round(flip_partition.positions[0][0] + reg_partition.positions[0][0], 4))
        self.assertEqual(0, round(flip_partition.widths[0][0] - reg_partition.widths[0][0], 4))
        self.assertEqual(0, round(flip_partition.angles[0][10] + reg_partition.angles[0][10], 4))
        self.assertEqual(0, round(flip_partition.offsets[0][10] + reg_partition.offsets[0][10], 4))

import os.path
import shutil
from pathlib import Path
from unittest import TestCase

import numpy as np


class TestBase(TestCase):

    @staticmethod
    def get_data_path():
        return Path(__file__).parent.parent / 'data'

    def get_lapsim_data_path(self):
        return self.get_data_path() / 'lapsim'

    def get_tracks_data_path(self):
        return self.get_data_path() / 'tracks'

    def get_temp_output_path(self):
        return self.get_data_path() / 'temp'

    def clear_temp_dir(self):
        if os.path.exists(self.get_temp_output_path()):
            shutil.rmtree(self.get_temp_output_path())

    def setUp(self) -> None:
        self.clear_temp_dir()

    def tearDown(self) -> None:
        self.clear_temp_dir()

    def assertFloatEqual(self, a, b):
        self.assertEqual(
            np.array([a], dtype=np.float32)[0],
            np.array([b], dtype=np.float32)[0],
        )

    def assertFloatListAlmostEqual(self, a, b, places=5):
        self.assertEqual(len(a), len(b))
        for i in range(len(a)):
            self.assertAlmostEqual(a[i], b[i], places)

    def assert2dFloatListAlmostEqual(self, a, b, places=5):
        self.assertEqual(len(a), len(b))
        for i in range(len(a)):
            self.assertFloatListAlmostEqual(a[i], b[i], places=places)

    def assertFloatListEqual(self, a, b):
        all_equal = np.all(np.array(a, dtype=np.float32) == np.array(b, dtype=np.float32))
        self.assertTrue(all_equal)

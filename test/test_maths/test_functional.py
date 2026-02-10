import numpy as np

from toolkit import maths
from utils.test_base import TestBase


class TestFunctional(TestBase):

    def test_at_indexes(self):
        self.assertEqual(
            [3, 6, 7],
            maths.at_indexes([3, 4, 5, 6, 7], [0, 3, 4])
        )

    def test_roll(self):
        def rolled_equal(shift):
            self.assertEqual(
                maths.roll([0, 1, 2, 3], shift),
                np.roll([0, 1, 2, 3], shift).tolist()
            )

        rolled_equal(-100)
        rolled_equal(-1)
        rolled_equal(0)
        rolled_equal(1)
        rolled_equal(100)

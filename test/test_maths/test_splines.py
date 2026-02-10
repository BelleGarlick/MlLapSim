from toolkit import maths
from utils.test_base import TestBase


class TestSplines(TestBase):

    def test_catmull_rom_spline(self):
        """Test the correct number of points are returned"""
        points = [(-5, 0), (0, 5), (5, 0), (0, -5)]
        self.assertEqual(16, len(maths.catmull_rom_spline(points, 4, True)))
        self.assertEqual(4, len(maths.catmull_rom_spline(points, 4, False)))

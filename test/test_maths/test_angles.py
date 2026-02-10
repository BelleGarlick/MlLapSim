from unittest import TestCase
from toolkit import maths
import math


class TestAngles(TestCase):

    def test_rotate(self):
        rotated = maths.rotate((0, 10), math.pi/2)
        self.assertAlmostEqual(rotated[0], -10, 1)
        self.assertAlmostEqual(rotated[1], 0, 1)

        rotated = maths.rotate((0, 10), math.pi / 2, around=(0, 9))
        self.assertAlmostEqual(rotated[0], -1, 1)
        self.assertAlmostEqual(rotated[1], 9, 1)

    def test_angle_to(self):
        self.assertEqual(0, maths.angle_to((0, 10), (10, 10)))
        self.assertAlmostEqual(-math.pi/4, maths.angle_to((0, 1), (1, 0)), 5)
        self.assertAlmostEqual(-math.pi/2, maths.angle_to((0, 1), (0, 0)),  5)

    def test_angle_between(self):
        self.assertAlmostEqual(math.pi * 0.5, maths.angle_between((0, 0), (0, 1), (1, 1)), 5)
        self.assertAlmostEqual(math.pi, maths.angle_between((0, 0), (0, 1), (0, 2)), 5)
        self.assertAlmostEqual(math.pi * 1.5, maths.angle_between((0, 0), (0, 1), (-1, 1)), 5)

    def test_angle3(self):
        self.assertAlmostEqual(-math.pi / 2, maths.angle3((0, 0), (0, 1), (1, 1)), 5)
        self.assertAlmostEqual(0, maths.angle3((0, 0), (0, 1), (0, 2)), 5)
        self.assertAlmostEqual(math.pi / 2, maths.angle3((0, 0), (0, 1), (-1, 1)), 5)

    def test_line_angle(self):
        self.assertEqual(0, maths.line_angle([0, 0, 10, 0]))
        self.assertAlmostEqual(math.pi / 2, maths.line_angle([0, 0, 0, 10]), 5)
        self.assertAlmostEqual(-math.pi / 2, maths.line_angle([0, 0, 0, -10]), 5)
        self.assertAlmostEqual(math.pi, maths.line_angle([10, 0, 0, 0]), 5)

    def test_angle_between_lines(self):
        self.assertAlmostEqual(0, maths.angle_between_lines([0, 0, 10, 0], [0, 10, 10, 10]), 5)
        self.assertAlmostEqual(math.pi / 2, maths.angle_between_lines([0, 0, 10, 0], [10, 0, 10, 10]), 5)
        self.assertAlmostEqual(math.pi / 4, maths.angle_between_lines([0, 0, 10, 0], [0, 10, 10, 0]), 5)

    def test_multi_angle_between_lines(self):
        angles = maths.multi_angle_between_lines(
            [[0, 0, 10, 0], [0, 0, 10, 0], [0, 0, 10, 0]],
            [[0, 10, 10, 10], [10, 0, 10, 10], [0, 10, 10, 0]]
        )

        self.assertAlmostEqual(angles[0], 0, 5)
        self.assertAlmostEqual(angles[1], math.pi / 2, 5)
        self.assertAlmostEqual(angles[2], math.pi / 4, 5)

from unittest import TestCase

from toolkit import maths


class TestLines(TestCase):

    def test_line_centers(self):
        centers = maths.line_centers([
            [20, 20, 10, 10],
            [10, 20, 0, 0],
        ])
        self.assertListEqual(centers, [[15, 15], [5, 10]])

    def test_line_lengths(self):
        lengths = maths.line_lengths([
            [20, 0, 10, 0],
            [3, 0, 0, 4],
        ])
        self.assertListEqual(lengths, [10, 5])

        self.assertEqual(maths.line_length([20, 0, 10, 0]), 10)

    def test_line_normalisation(self):
        lengths = maths.normalise_lines([[20, 0, 10, 0]])
        self.assertTupleEqual(lengths[0], (-1, 0))

    def test_set_line_lengths(self):
        lengths = maths.set_line_lengths([
            [20, 0, 10, 0],
            [3, 0, 0, 4],
        ], [5, 10])
        self.assertListEqual(lengths[0], [17.5, 0, 12.5, 0])
        self.assertListEqual(lengths[1], [4.5, -2.0, -1.5, 6.0])

    def test_extend_lines(self):
        extended = maths.extend_lines([
            [20, 0, 10, 0],
            [3, 0, 0, 4],
        ], 10)
        self.assertListEqual(extended[0], [25.0, 0.0, 5.0, 0.0])
        self.assertListEqual(extended[1], [4.5, -2.0, -1.5, 6.0])

    def test_start_line_end_line(self):
        points = maths.start_points([
            [20, 0, 10, 0],
            [3, 0, 0, 4],
        ])
        self.assertListEqual(points[0], [20, 0])
        self.assertListEqual(points[1], [3, 0])

        points = maths.end_points([
            [20, 0, 10, 0],
            [3, 0, 0, 4],
        ])
        self.assertListEqual(points[0], [10, 0])
        self.assertListEqual(points[1], [0, 4])

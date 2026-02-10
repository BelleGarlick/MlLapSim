from toolkit import maths
from utils.test_base import TestBase


class TestPoints(TestBase):

    def test_normalising_a_point(self):
        self.assertTupleEqual(maths.normalise_point((1, 0)), (1, 0))
        self.assertTupleEqual(maths.normalise_point((10, 0)), (1, 0))
        self.assertTupleEqual(maths.normalise_point((0, 5)), (0, 1))

    def test_normalising_points(self):
        self.assertEqual(
            maths.normalise_points([(1, 0), (10, 0), (0, 5),]),
            [(1, 0), (1, 0), (0, 1)]
        )

    def test_distance(self):
        self.assertEqual(
            5,
            maths.distance((0, 0), (3, 4))
        )
        self.assertEqual(
            25,
            maths.distance((10, 4), (34, 11))
        )

    def test_distances(self):
        self.assertEqual(
            [10, 8],
            maths.distances(
                [(10, 0), (2, 10)],
                (10, 10)
            )
        )

    def test_closest_point(self):
        points = [(10, 10), (-10, 10), (2, 1), (3, 0)]
        origin = (0, 0)

        self.assertTupleEqual((2, 1), maths.closest_point(origin, points))
        self.assertEqual(2, maths.closest_point(origin, points, True))

    def test_points_to_lines(self):
        points = [(10, 10), (-10, 10), (2, 1), (3, 0)]

        self.assertEqual(
            maths.points_to_lines(points),
            [(10, 10, -10, 10), (-10, 10, 2, 1), (2, 1, 3, 0), (3, 0, 10, 10)]
        )

    def test_sub_point(self):
        self.assertTupleEqual(
            (2, 3),
            maths.sub_point((10, 10), (8, 7))
        )

    def test_sub_points(self):
        self.assertEqual(
            [(2, 3), (5, 5)],
            maths.sub_points(
                [(10, 10), (20, 20)],
                [(8, 7), (15, 15)]
            )
        )

    def test_interpolate_points(self):
        self.assertEqual(
            maths.interpolate_points_between((0, 0), (10, 10), 9),
            [(i, i) for i in range(1, 10)]
        )

    def test_getting_points_on_line_large_lines(self):
        lines = [(0, 0), (0, 10)]
        resultant_points = maths.get_points_on_paths(lines, spacing=2, loop=False)
        self.assert2dFloatListAlmostEqual(resultant_points,
            [[0, 0.], [0, 2.], [0, 4.], [0, 6.], [0, 8.], [0, 10.]]
        )

        resultant_points = maths.get_points_on_paths(lines, spacing=2, loop=True)
        self.assert2dFloatListAlmostEqual(
            resultant_points,
            [[0, 0.], [0, 2.], [0, 4.], [0, 6.], [0, 8.],
            [0, 10.], [0, 8.], [0, 6.], [0, 4.], [0, 2.]]
        )

    def test_getting_points_on_line_small_lines(self):
        lines = [(0, i) for i in range(11)]
        resultant_points = maths.get_points_on_paths(lines, spacing=2, loop=False)
        self.assert2dFloatListAlmostEqual(resultant_points,
            [[0, 0.], [0, 2.], [0, 4.], [0, 6.], [0, 8.], [0, 10.]]
        )

        resultant_points = maths.get_points_on_paths(lines, spacing=2, loop=True)
        self.assert2dFloatListAlmostEqual(resultant_points,
            [[0, 0.], [0, 2.], [0, 4.], [0, 6.], [0, 8.],
            [0, 10.], [0, 8.], [0, 6.], [0, 4.], [0, 2.]]
        )

    def test_getting_points_on_line_non_divisible(self):
        lines = [[0, 0], [0, 10]]
        resultant_points = maths.get_points_on_paths(lines, spacing=3, loop=False)
        self.assert2dFloatListAlmostEqual(resultant_points,
            [[0, 0.], [0, 3.], [0, 6.], [0, 9]]
        )

        resultant_points = maths.get_points_on_paths(lines, spacing=3, loop=True)
        self.assert2dFloatListAlmostEqual(resultant_points,
            [[0, 0.], [0, 3.], [0, 6.], [0, 9.], [0, 8.], [0, 5.], [0, 2]]
        )

    def test_getting_points_on_line_small_lines_non_divisible(self):
        lines = [[0, i] for i in range(11)]
        resultant_points = maths.get_points_on_paths(lines, spacing=3, loop=False)
        self.assert2dFloatListAlmostEqual(resultant_points,
            [[0, 0.], [0, 3.], [0, 6.], [0, 9]]
        )

        resultant_points = maths.get_points_on_paths(lines, spacing=3, loop=True)
        self.assert2dFloatListAlmostEqual(resultant_points,
            [[0, 0.], [0, 3.], [0, 6.], [0, 9.], [0, 8.], [0, 5.], [0, 2]]
        )

from unittest import TestCase

from toolkit import maths


class TestIntersections(TestCase):

    def test_segment_intersection(self):
        intersections = maths.segment_intersections([0, 0, 0, 0], [
            [0, 0, 0, 0],
        ])
        self.assertEqual(intersections, [])

        intersections = maths.segment_intersections([10, 0, -10, 0], [
            [0, 0, 0, 0],
            [10, 0, -10, 0],
            [0, 10, 0, -10],
            [-5, -5, 5, 5]
        ])
        self.assertEqual([(0, 0), (0, 0)], intersections)

        intersections, indexes = maths.segment_intersections([10, 0, -10, 0], [
            [0, 0, 0, 0],
            [10, 0, -10, 0],
            [0, 10, 0, -10],
            [-5, -5, 5, 5]
        ], return_indexes=True)
        self.assertEqual([(0, 0), (0, 0)], intersections)
        self.assertEqual([2, 3], indexes)

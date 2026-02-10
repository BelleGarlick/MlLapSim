from toolkit import maths
from utils.test_base import TestBase


class TestNormals(TestBase):

    def test_create_line_normals_from_points(self):
        normals = maths.create_line_normals_from_points(
            [(0, -2), (-2, 0), (0, 2), (2, 0)],
            length=1
        )
        self.assert2dFloatListAlmostEqual(normals, [
            [0, -2.5, 0, -1.5],
            [-2.5, 0, -1.5, 0],
            [0, 2.5, 0, 1.5],
            [2.5, 0, 1.5, 0],
        ])

    def test_create_normals_on_path(self):
        # This uses the get_points_on_path and create_line_normals
        # which are tested elswhere.
        normals = maths.create_normals_on_path(
            [(0, -2), (-2, 0), (0, 2), (2, 0)],
            spacing=1,
            width=10
        )
        line_centers = maths.line_centers(normals)
        # Check distance between all points rounds to 1
        self.assertTrue(
            [
                round(maths.distance(line_centers[x], line_centers[x - 1])) == 1
                for x in range(len(line_centers))
            ]

        )

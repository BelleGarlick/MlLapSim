import math
import numpy as np
from toolkit import maths
from toolkit.tracks.models import InvalidTrackGeneration
from toolkit.tracks.smoother.smoother import (
    CollisionPairs,
    _smooth_normals,
    _extend_normals_until_collision,
    _collapse_collisions_pairs,
    _get_closest_collision_index
)

from utils.test_base import TestBase


class TestNormalsCreation(TestBase):

    def test_smooth_normals(self):
        """Test smoothing normals is correct"""
        centers = []
        original_normals = []

        for i in range(16):
            center = np.array((
                math.sin((i / 16) * math.pi * 2),
                math.cos((i / 16) * math.pi * 2)
            ))
            p1 = center / 2
            p2 = center * 1.5

            r = np.random.uniform(-0.1, 0.1, (2,))
            p1 += r
            p2 -= r

            centers.append(center)
            original_normals.append([p1[0], p1[1], p2[0], p2[1]])

        centers = np.array(centers)

        smooth_normals = _smooth_normals(original_normals, iterations=1000, width=1)
        new_centers = maths.line_centers(smooth_normals)

        # Check normals sizes don't change
        self.assertFloatListEqual(np.array(centers, np.float16), np.array(new_centers, np.float16))
        for length in maths.line_lengths(smooth_normals):
            self.assertAlmostEqual(1, length, 4)

        # Sum the change in angles sequared and compare. og normals should have some large
        # values so we square them to see that the smooth normals is smoother.
        self.assertLess(
            sum([maths.angle_between_lines(smooth_normals[i - 1], smooth_normals[i]) ** 2 for i in
                 range(len(smooth_normals))]),
            sum([maths.angle_between_lines(original_normals[i - 1], original_normals[i]) ** 2 for i in
                 range(len(original_normals))])
        )

    def test_extend_normals_until_collision(self):
        """Test correct normals are returns

        tests 1 intersection: x=-3
        tests 2 intersection: x=-1
        tests 0 to 1 intersection after 1 repeat: x=-7
        tests 0 to 2 intersection after 1 repeat: x=-5
        tests fail to find a collision: x=-23
        """
        boundary_circle = [[0, 0], [0, 10], [2, 10], [2, 0]]

        def create_normals(x):
            return [[x, 3, x + 4, 3], [x, 7, x + 4, 7]]

        # Test collisions where lines naturally only intersect once
        collisions = _extend_normals_until_collision(create_normals(-3), boundary_circle)
        self.assertEqual(collisions, [
            ([(0.0, 3.0)], [1]),
            ([(0.0, 7.0)], [1])
        ])

        # Test collisions where lines naturally intersect twice
        collisions = _extend_normals_until_collision(create_normals(-1), boundary_circle)
        self.assertEqual(collisions, [
            ([(0.0, 3.0), (2.0, 3.0)], [1, 3]),
            ([(0.0, 7.0), (2.0, 7.0)], [1, 3])
        ])

        # Test 1 collision after extending the line
        collisions = _extend_normals_until_collision(create_normals(-7), boundary_circle)
        self.assertEqual(collisions, [([(0.0, 3.0)], [1]), ([(0.0, 7.0)], [1])])

        # Test 2 collision after extending the line
        collisions = _extend_normals_until_collision(create_normals(-5), boundary_circle)
        self.assertEqual(collisions, [
            ([(0.0, 3.0), (2.0, 3.0)], [1, 3]),
            ([(0.0, 7.0), (2.0, 7.0)], [1, 3])
        ])

        # Test exception after no collisions found
        with self.assertRaises(InvalidTrackGeneration):
            _extend_normals_until_collision(create_normals(-23), boundary_circle)

    def test_collapse_collisions_pairs(self):
        # Test creation is correct even for looping
        result = _collapse_collisions_pairs(
            [
                [0, 0, 0, 0],
                [1, 2, 1, 2],
                [-1, -2, -1, -2],
            ],
            [
                [[(0, 0)], [0]],
                [[(1, 2), (3, 4)], [1, 2]],
                [[(-1, -2), (5, 6)], [99, 20]],
            ],
            100)
        self.assertTrue(np.all((np.array([[0, 0], [1, 2], [-1, -2]]) == result)))

        # Test a circle where each normal has more than one intersection
        test_normals = []
        collisions: CollisionPairs = []
        n = 8
        for i in range(n):
            x, y = math.sin(i / n * math.tau), math.cos(i / n * math.tau)
            nx, ny = math.sin((i + 1) / n * math.tau), math.cos((i + 1) / n * math.tau)

            test_normals.append([x * 10, y * 10, x * -2, y * -2])
            collisions.append(([[nx, ny], [x, y]], [i, (i + 4) % n]))

        # Test the outputs are the outside of the circle not the inside, this
        # is 9 away for the normal
        resultant_boundaries = _collapse_collisions_pairs(test_normals, collisions, n)

        # All values should be 3
        resultant_distances = [
            maths.distance(x, resultant_boundaries[i])
            for i, x in enumerate(maths.line_centers(test_normals))
        ]

        for value in resultant_distances:
            self.assertAlmostEqual(value, 3, 6)

    def test_get_closest_collision_index(self):
        # Test finding the closest collision index
        self.assertEqual(5, _get_closest_collision_index(10, [5, 20, 100], 100))
        self.assertEqual(5, _get_closest_collision_index(99, [5, 20, 90], 100))
        self.assertEqual(99, _get_closest_collision_index(2, [30, 20, 99], 100))

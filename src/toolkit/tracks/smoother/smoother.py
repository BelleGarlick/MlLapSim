import math
from typing import List, Tuple

import numpy as np

from toolkit import maths
from toolkit.tracks.models import InvalidTrackGeneration
from toolkit.utils.logger import log_time
from toolkit.utils.spacial_map import SpatialLineItem, SpatialMap


# A tuple of the list collission points and collided boundary indexes per normal
#  [
#    ([[0.1, 0.3], [0.2, 0.4]], [0, 2])
#  ]
CollisionPairs = List[Tuple[List[Tuple[float]], List[int]]]


def _smooth_normals(normals: List[List[float]], iterations: int, width: float) -> List[List[float]]:
    """This function smooths out a list of normals of the track. This prevents
    intersecting normals at tight corner edges.

    Args:
        normals: List of normals to smooth
        iterations: Number of iterations to smooth
        width: Resultant widths of the normals

    Returns:
        Smoothed normals
    """
    # Smooth the normals
    centeres = maths.line_centers(normals)
    vectors = maths.sub_points(
        maths.start_points(normals),
        centeres
    )

    max_angle = math.pi / 3

    for _ in range(iterations):
        updated_vectors = maths.add_points_lists([
            maths.roll(vectors, -1), vectors, maths.roll(vectors, 1)
        ])

        angles = maths.multi_angle_between_lines(
            [[px + vx, py + vy, px, py]
             for (px, py), (vx, vy) in
             zip(centeres, updated_vectors)],
            normals
        )

        for i, a in enumerate(angles):
            if abs(a) > max_angle:
                updated_vectors[i] = vectors[i]

        vectors = maths.normalise_points(updated_vectors)

    multipler = width / 2
    vectors = [
        (px * multipler, py * multipler)
        for px, py in vectors
    ]

    return [
        [
            centeres[i][0] + vectors[i][0],
            centeres[i][1] + vectors[i][1],
            centeres[i][0] - vectors[i][0],
            centeres[i][1] - vectors[i][1]]
        for i in range(len(centeres))
    ]


def _split_normals(normals: List[List[float]]) -> Tuple[List[List[float]], List[List[float]]]:
    """Split normals into two half normals sides"""
    n = len(normals)

    centers = maths.line_centers(normals)

    left_normals = [[]] * n
    right_normals = [[]] * n

    for i in range(n):
        center = centers[i]
        normal = normals[i]
        left_normals[i] = [center[0], center[1], normal[0], normal[1]]
        right_normals[i] = [center[0], center[1], normal[2], normal[3]]

    return left_normals, right_normals


def _extend_normals_until_collision(
        half_normal: List[List[float]],
        boundary_points: List[List[float]],
        max_extensions=5
) -> CollisionPairs:
    """Extend the given half normal until it intersects with the boundary line
    and return all the collision points and indexes of the boundary lines where
    the normal intersects.

    Args:
        half_normal: The half normal line
        boundary_points: The list of points which make up the boundary
        max_extensions: How many times the line may be extended before raising
            an error.

    Returns:
        A list of tuples (one per normal) where the tuple is a list of points
        that the normal collides with and a list of all the boundary indexes
        where those collisions occur.
    """
    # Define pairs of collision points and boundary indexes per normal
    normal_collisions: CollisionPairs = [([], []) for _ in half_normal]

    # Create boundary line and spatial map of boundary lines
    boundary_lines = maths.points_to_lines(boundary_points)
    boundary_lines = maths.roll(boundary_lines, 1)
    spatial_map_cell_size = max(maths.line_lengths(half_normal)) * max_extensions
    boundary_spatial_map = SpatialMap(spatial_map_cell_size * 2)
    for i in range(len(boundary_points)):
        boundary_spatial_map.add_item(SpatialLineItem(boundary_points[i - 1], boundary_points[i]))

    # Loop through each normal and see the collision and the boundary index
    # where the collision occurs. If no collision is found, extend the line a
    # few times until the collision is found, if not throw an error.
    for i, normal in enumerate(half_normal):
        vec = maths.sub_point(normal[2:], normal[:2])
        for extension in range(max_extensions):
            extended_normal = np.array([  # TODO Remove np.array
                normal[0],
                normal[1],
                normal[2] + vec[0] * extension,
                normal[3] + vec[1] * extension
            ])
            center = (extended_normal[:2] + extended_normal[2:]) / 2

            # Find all collisions with the extended normal
            possible_indicies = boundary_spatial_map.get_items(center)
            collisions, intersections = maths.segment_intersections(
                extended_normal,
                maths.at_indexes(boundary_lines, possible_indicies),
                return_indexes=True)

            normal_collisions[i] = collisions, maths.at_indexes(possible_indicies, intersections)

            if len(collisions) > 0:
                break

        if len(normal_collisions[i][0]) == 0:
            # normal_collisions[i] = [], []
            # TODO Change this to another exception for smoothing lines. Will need to check usages
            raise InvalidTrackGeneration("Unable to find boundary collision against smoothed lines")

    return normal_collisions


def _collapse_collisions_pairs(
        half_normals: List[List[float]],
        half_normal_collisions: CollisionPairs,
        n_boundaries: int
) -> List[List[float]]:
    """Collapse the collisions pairs until we get just one selected collision
    per-pair of points.

    The theory behind this is that where there are intersections, the normals
    may intersect both the correct and incorrect lines. If there is a normal
    that intersects two parts of the boundary, then we may not know which point
    is correct. This function solves this by finding a single collision, and
    following it round the track, selecing the next point that collides with
    the closest boundary segment, not collision point (which in some instances
    where the track curves back on iterself may otherwise fail).

    When, for example the track is a small loop, every normal may have multiple
    collisions, to overcome this, two normals are selected at opposing ends of
    the track. Those normals are used as the starting indexes, then normals
    are collapsed from that point. We can then compare where those results are
    the same, and used the first same value as the start index for final
    run. This should result in the correct final output.

    Args:
        half_normal_collisions: List of collisions pairs on this side of the
            normal lines.
        n_boundaries: Number of boundary segments the normals may intersect

    Returns:
        The list of correct points resulting from the collapsed inputs.
    """
    # Loop through to find a collission index that only has one intersection
    # to use as the base intersection
    start_normal: int = -1
    start_boundary_index = -1
    for i, left_normal_collision in enumerate(half_normal_collisions):
        if len(left_normal_collision[0]) == 1:
            start_normal = i
            start_boundary_index = half_normal_collisions[start_normal][1][0]
            break

    if start_normal == -1:
        # If no starting point works, then this method will choose two normal
        # at opposing ends of the track. These points act as the start points
        # which two attempts at collapsing the circuit run from. The result of
        # these two runs is compared, any common index is used as the start
        # of the new run, otherwise, the first index is used as the starting
        # point
        centers = maths.line_centers(half_normals)
        idx1, idx2 = 0, len(centers) // 2
        idx1_points, idx1_segs_idxs = half_normal_collisions[idx1]
        idx2_points, idx2_segs_idxs = half_normal_collisions[idx2]

        # Calculate the closest points to the normals, use the closest
        # as the start reference intersection
        d1s = maths.distances(idx1_points, centers[idx1])
        d2s = maths.distances(idx2_points, centers[idx2])
        seg_intersec_1, seg_intersec_2 = idx1_segs_idxs[np.argmin(d1s)], idx2_segs_idxs[np.argmin(d2s)]

        # Create the two new attempts at the solutions
        final_solutions_1 = _collapse_collisions_from_index(idx1, seg_intersec_1, half_normal_collisions, n_boundaries)
        final_solutions_2 = _collapse_collisions_from_index(idx2, seg_intersec_2, half_normal_collisions, n_boundaries)

        # Find where attempts are equal and use the first index as the starting value
        where_equal = [final_solutions_1[i] == final_solutions_2[i] for i in range(len(final_solutions_1))]
        start_normal = 0
        start_boundary_index = half_normal_collisions[start_normal][1][0]
        for i, val in enumerate(where_equal):
            if val:
                start_normal = i
                starting_point = final_solutions_1[i]
                boundary_index = [
                    i for i, p in enumerate(half_normal_collisions[i][0])
                    if np.all(p == starting_point)][0]
                start_boundary_index = half_normal_collisions[i][1][boundary_index]
                break

    return _collapse_collisions_from_index(
        start_normal,
        start_boundary_index,
        half_normal_collisions,
        n_boundaries
    )


def _collapse_collisions_from_index(
        start_point: int,
        start_intersection_index: int,
        half_normal_collisions: CollisionPairs,
        n_boundaries: int
) -> List[List[float]]:
    """This function takes the starting normal index and boundary intersection
    index and walkes the normals finding the next closest boundary intersection
    and using that as the collision point. This allows for handling normals
    with overlap for figure of 8 tracks.

    This function is tested indirectly via unit tests in the
    `_collapse_collisions_pairs`sts.

    Args:
        start_point: The starting normal index
        start_intersection_index: The starting boundary intersection index
        half_normal_collisions: The normal collision:index pairs
        n_boundaries: Number of boundary line segments.

    Returns:
        The coordinates where the normals collide with the boundary
    """
    resultant_collisions: List[List[float]] = [[0, 0] for _ in half_normal_collisions]
    n_normals = len(half_normal_collisions)

    intersection_index = start_intersection_index
    for i in range(start_point + 1, n_normals + start_point + 1):
        index_collisions = half_normal_collisions[i % n_normals]
        closest_index = _get_closest_collision_index(intersection_index, index_collisions[1], n_boundaries)

        # Set index to the first items, so it's initialised to some values (it will be overwritten)
        resultant_collisions[i % n_normals] = index_collisions[0][0]
        intersection_index = index_collisions[1][0]

        # Find the intersection with that closest index
        for point, idx in zip(index_collisions[0], index_collisions[1]):
            if idx == closest_index:
                resultant_collisions[i % n_normals] = point
                intersection_index = idx

    return resultant_collisions


def _get_closest_collision_index(idx: int, possible_idxs: List[int], n: int):
    """Test finding the closest indexes to one another.

    For example, 2 is closst to 99 out of [87, 99] where n = 100

    Args:
        idx: The current normal index
        possible_idxs: The possible indexes the intersection is closest to
        n: The number of indexs

    Returns:
        The closest index
    """
    idx = idx % n

    best_idx = None
    for i, pos_idx in enumerate(possible_idxs):
        delta = min(
            abs(idx - pos_idx),
            abs(idx - (pos_idx + n)),
            abs(idx + n - pos_idx),
            abs(idx - n - pos_idx)
        )

        if best_idx is None:
            best_idx = delta, pos_idx
        else:
            if delta < best_idx[0]:
                best_idx = delta, pos_idx

    return best_idx[1]

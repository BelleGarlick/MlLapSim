import dataclasses
from typing import List, Tuple

from toolkit import maths
from toolkit.tracks.models import Track
from toolkit.utils.logger import log_time


"""Shortest Path

This module calculates the shortest path areound a track. This process is 
performed in two steps. 
 - Step 1: Discerete search for the optimal path. 
    This step divides each normal into a series of steps resulting in several
    discrete points upon which a dijkstra inspired algorithm can run. However,
    being only discretely found, the resultant path does not necessarily result
    in smooth shortest line.
 - Step 2: Interpolate between the discretely found points to find a shorter 
    line. This is done by iterativelly calculating the point directly between 
    next line's point and the previous lines point then trying to set it upon 
    the current line. This results in a smoother line.
    
There is a trade-off in performance between these two steps. Step 2 will never
 find the global optima. Step 1 find's a better global optima, however, with 
 a low number of steps, it isn't a good optimal. More steps improves the output
 but takes longer. The closest step 1 is to the global optimal, the fewer 
 iterations step 2 requires, The predefined value was found by running tests to
 find a number of steps which results in a lower amount of run time.
"""


@dataclasses.dataclass
class ShortestPathResponse:

    positions: List[Tuple[float, float]]
    normals: List[List[float]]
    interpolations: List[float]


@dataclasses.dataclass
class ReferencePoint:

    position: Tuple[float, float]
    interpolations: List[float]
    total_distance: float


def shortest_path_walk_track(lines: List[Tuple[float, float, float, float]], steps=70):
    """Find the shortest path on the track

    This dijkstra's inspired algorithm works, by going from one normal to the
    next, calculating a series of steps along each normal and comparing each
    point with the closest total distance.

    Unlike dijkstra which looks at all avilable points, this only looks at the
    current or previous line which is slighlty more memory effient.
    Additionally, we can look at the curent point in the same index, and compare
    the distances from the forward/backwards steps. If a step is ever larger
    than the previous, then we can halt the search on that line. Most of the time
    the closest point will be the same step interpolation as the previous meaning
    we can masivally reduce the search to almost O(n) per interpolation point.

    Each reference point stores the list of interpolations taken to get to
    the current point.

    Args:
        lines: The lines to search on
        steps: The number of steps to create control points for

    Returns:
        The list of interpolations
    """
    step_values = [step / steps for step in range(steps+1)]

    # Calculate the initial reference points for the first line
    line = lines[0]
    reference_points = [
        ReferencePoint(
            position=maths.lerp_point(line[:2], line[2:], value),
            interpolations=[value],
            total_distance=0
        )
        for value in step_values
    ]

    # Iterate through all succeeding lines to find the shortest path
    for line in lines[1:]:
        new_best = []
        for step_index, step_value in enumerate(step_values):
            point = maths.lerp_point(line[:2], line[2:], step_value)

            closest_index = step_index
            closest_distance = maths.distance(point, reference_points[step_index].position) + reference_points[step_index].total_distance

            # Walk forward until we find a bigger distance, then halt
            for forward_index in range(closest_index + 1, steps):
                distance = maths.distance(point, reference_points[forward_index].position) + reference_points[forward_index].total_distance
                if distance >= closest_distance:
                    break
                else:
                    closest_distance = distance
                    closest_index = forward_index

            # Walk backwards until we find a bigger distance, then halt
            for reverse_index in range(closest_index - 1, -1, -1):
                distance = maths.distance(point, reference_points[reverse_index].position) + reference_points[reverse_index].total_distance
                if distance >= closest_distance:
                    break
                else:
                    closest_distance = distance
                    closest_index = reverse_index

            new_best.append(ReferencePoint(
                position=point,
                interpolations=reference_points[closest_index].interpolations + [step_value],
                total_distance=closest_distance
            ))

        reference_points = new_best

    # Find the reference point with the smallest total distance.
    closest_point = reference_points[0]
    for point in reference_points:
        if point.total_distance < closest_point.total_distance:
            closest_point = point

    return closest_point.interpolations


@log_time("Shortest Path Calculated", indent=0)
def shortest_path(track: Track, padding=1, max_iterations=10_000, early_stop_threshold=0.001) -> ShortestPathResponse:
    """Calculate the shortest path on a track

    Args:
        track: Track to calculate the shortest path upon
        padding: Boundary padding to inset the path from
        max_iterations: Maximum number of iterations to calculate the shortest
         path from
        early_stop_threshold: Early stopping threshold

    Returns:
        List of points representing the shortest path
    """

    # normals = _sparse_normals(track)
    normals = list(map(lambda norm: norm.arr(), track.segmentations))
    normals = maths.extend_lines(normals, -padding, min=0.1)

    left_points = maths.start_points(normals)
    right_points = maths.end_points(normals)
    lengths = maths.line_lengths(normals)
    n = len(normals)

    interpolations = shortest_path_walk_track(normals)

    points = maths.lerp_points_on_lines(normals, interpolations)
    total_length = sum(maths.line_lengths(maths.points_to_lines(points)))

    points_to_check = {i for i in range(n)}

    for iterations in range(max_iterations):
        changed_points = set()
        for i in range(n):
            if i not in points_to_check:
                continue

            pi, ni = i - 1, i + 1 - n

            initial_interpolation = interpolations[i]
            prev_point, next_point = points[pi], points[ni]
            intersection_point = maths.segment_intersections(normals[i], [prev_point + next_point])

            if not intersection_point:
                l_distance = maths.distance(prev_point, left_points[i]) + maths.distance(next_point, left_points[i])
                r_distance = maths.distance(prev_point, right_points[i]) + maths.distance(next_point, right_points[i])

                # 1 if left distance is bigger than the right side means
                # this'll move to the edge and choose the shortest distance
                interpolations[i] = int(l_distance > r_distance)

            else:
                l_distance = maths.distance(left_points[i], intersection_point[0])
                interpolations[i] = l_distance / lengths[i]

            if interpolations[i] != initial_interpolation:
                changed_points.update([i, (i + 1) % n, (i + n - 1) % n])

        points_to_check = changed_points

        # Recalc existing points
        points = maths.lerp_points_on_lines(normals, interpolations)

        # Calculate length for early stopping
        new_length = sum(maths.line_lengths(maths.points_to_lines(points)))
        improvement = total_length - new_length
        if improvement < early_stop_threshold:
            print(f"\rEarly stopping reached on iteration {iterations} ({new_length}).")
            break
        total_length = new_length

    return ShortestPathResponse(
        positions=points,
        interpolations=interpolations,
        normals=normals
    )

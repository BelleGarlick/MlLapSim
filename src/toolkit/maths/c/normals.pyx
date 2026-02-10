cimport cython
from libc.math cimport (
    pow,
    sqrt
)
from .points import points_to_lines, closest_point, get_points_on_paths
from .angles import rotate, angle3, angle_to
from .lines import line_centers
from .intersections import segment_intersections
from .points import get_points_on_paths


# Create shorthand types for points and lines
ctypedef tuple[double, double] Point
ctypedef list[tuple[double, double]] Points
ctypedef tuple[double, double, double, double] Line
ctypedef list[tuple[double, double, double, double]] Lines


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef create_line_normals_from_points(points: List[Tuple[float, float]], length=10):
    """Create normal lines from a list of points

    Args:
        points: List of points to form normals from
        length: Length of the line

    Returns:
        Normal lines
    """
    cdef int count = len(points)
    cdef lines = [0] * count
    cdef float half_width = length / 2

    cdef int index
    cdef float current_angle, to_angle, normal
    cdef Point point

    if count >= 3:
        for index in range(count):
            point = points[index]
            prev_point = points[(index + count - 1) % count]

            current_angle = angle3(prev_point, point, points[(index + 1) % count])
            to_angle = angle_to(point, prev_point)
            normal = to_angle + (current_angle / 2)

            left_end = rotate((point[0], point[1] - half_width), normal, point)
            right_end = rotate((point[0], point[1] + half_width), normal, point)

            lines[index] = left_end + right_end

    return lines


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef Lines create_normals_on_path(Points path, double width, double spacing):
    """Create normals on path

    Args:
        path: The points to calculate normals using
        width: The width of the lines
        spacing: The gap between lines

    Returns:
        The generated lines
    """
    spaced_points = get_points_on_paths(path, spacing, loop=True)
    normals = create_line_normals_from_points(spaced_points, width)

    return normals


# TODO Optimise with a spatial map
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef Lines trim_normals_to_boundary(Lines lines, Points left_boundary, Points right_boundary):
    """This function has no unit tests as it's only using in one part of the code
    which has it's own unit tests.
    """
    cdef int n_lines = len(lines)

    cdef Points centers = line_centers(lines)
    cdef Lines lb = points_to_lines(left_boundary)
    cdef Lines rb = points_to_lines(right_boundary)

    new_normals = [0] * n_lines

    cdef normal, center_point, left_line, right_line
    cdef left_intersections, right_intersections
    cdef left_intersection, right_intersection

    for i in range(n_lines):
        normal = lines[i]
        center_point = centers[i]
        left_line = [normal[0], normal[1], center_point[0], center_point[1]]
        right_line = [normal[2], normal[3], center_point[0], center_point[1]]

        left_intersections = segment_intersections(left_line, lb)
        right_intersections = segment_intersections(right_line, rb)

        left_intersection = closest_point(center_point, left_intersections)
        right_intersection = closest_point(center_point, right_intersections)

        if left_intersection is None:
            left_intersection = normal[0:2]
        if right_intersection is None:
            right_intersection = normal[2:4]

        new_normals[i] = left_intersection + right_intersection

    return new_normals

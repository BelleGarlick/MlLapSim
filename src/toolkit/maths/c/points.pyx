cimport cython
from libc.math cimport hypot
from .lines import line_lengths


# Create shorthand types for points and lines
ctypedef tuple[double, double] Point
ctypedef list[tuple[double, double]] Points
ctypedef tuple[double, double, double, double] Line
ctypedef list[tuple[double, double, double, double]] Lines


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef Point normalise_point(Point point):
    """Normalise a point"""
    cdef double x = point[0]
    cdef double y = point[1]
    cdef double d = hypot(x, y)

    if d == 0:
        return (0, 0)

    return (x / d, y / d)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef Points normalise_points(Points points):
    """Normalise a list of points."""
    return [
        normalise_point(p)
        for p in points
    ]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double cdistance(tuple[double, double] point_a, tuple[double, double] point_b):
    """Calculate the distance between two points"""
    cdef double dx = point_a[0] - point_b[0]
    cdef double dy = point_a[1] - point_b[1]
    return hypot(dx, dy)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double distance(tuple[double, double] point_a, tuple[double, double] point_b):
    """Calculate the distance between two points"""
    return cdistance(point_a, point_b)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list[double] distances(list[tuple[double, double]] points, tuple[double, double] origin):
    """Calculate the distances from the list of points to the origin.

    Args:
        points: List of points to calculate distance to.
        origin: The reference point to calculate distances from

    Returns:
        List of distances of each point from the reference
    """
    return [
        cdistance(p, origin)
        for p in points
    ]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef closest_point(Point origin, Points points, return_index: bool = False):
    """Find the closest point between the list of points and the given origin. 13x faster"""
    cdef closest_point = None
    cdef int closest_index = -1
    cdef double closest_distance = -1
    cdef double dist

    cdef int i = 0

    for i in range(len(points)):
        point = points[i]
        dist = cdistance(origin, point)

        if closest_distance == -1 or dist < closest_distance:
            closest_distance = dist
            closest_index = i
            closest_point = point

    if return_index:
        return closest_index

    return closest_point


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef points_to_lines(Points points):
    """Turn a list of points to lines. 4x faster"""
    cdef int count = len(points)
    if count == 0:
        return []

    cdef lines = [0] * count

    cdef lp = points[0]
    cdef float last_x = lp[0]
    cdef float last_y = lp[1]
    cdef float x, y

    for i in range(count):
        x, y = points[(i + 1) % count]
        lines[i] = (last_x, last_y, x, y)
        last_x, last_y = x, y

    return lines


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef Point sub_point(Point a, Point b):
    return (
        a[0] - b[0],
        a[1] - b[1]
    )


cpdef Points sub_points(Points a, Points b):
    # todo make sure same length
    cdef int count = len(a)

    cdef Points res = [[0, 0]] * count

    # check same length a as b
    for i in range(count):
        res[i] = sub_point(a[i], b[i])

    return res



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef interpolate_points_between(Point p1, Point p2, int n):
    """This function returns the interpolated points between two points. The
    interpolated points are based on a target seperation distance. The final
    spacing between the lines will never be less than this value but it can
    be more than the value.

    This function is used to calculate control points during track generation
    so should avoid being altered to change the above functionality

    Args:
        p1: Point 1
        p2: Point 2
        n: Number of points to interpolate

    Returns:
        The interpolated points between p1 and p2
    """
    cdef float x_start = p1[0]
    cdef float y_start = p1[1]
    cdef float x_range = p2[0] - x_start
    cdef float y_range = p2[1] - y_start

    # This generates [.2, .4, .6, .8] for n=4 points. purposely exlcuding the bounds
    cdef float idx
    cdef list[float] interpolated_distances = [idx / (n + 1) for idx in range(1, n + 1)]

    cdef Points interpolated_points = [(0, 0)] * (n)
    cdef long i = 0
    cdef float interpolation_portion

    for i in range(n):
        interpolation_portion = interpolated_distances[i]
        interpolated_points[i] = (
            x_start + (interpolation_portion * x_range),
            y_start + (interpolation_portion * y_range)
        )

    return interpolated_points


@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cpdef get_points_on_paths(Points path, float spacing, loop: bool):
    """Generate points on the given path

    This is a reasonable complex function. Walking the track, moving by the
    spacing between points. A line is focused upon, then we keep walking the
    focus line until the length fo the line is reached, then we start focusing
    on the next line, and walking that until the distance has been walked etc.

    Args:
        path: Path to generate points on
        spacing: The distance between points
        loop: If true, loop the path to draw points on

    Returns:
        The points upon the initial path
    """
    # Convert points to lines
    lines = points_to_lines(path)
    if not loop: lines = lines[:-1]
    cdef list[float] lengths = line_lengths(lines)
    cdef int n_lines = len(lines)

    # Calculate the total distances
    cumulative_distances = [0] * (n_lines + 1)
    for i in range(n_lines):
        cumulative_distances[i + 1] = cumulative_distances[i] + lengths[i]

    # Calculate the cumulative spacings. e.g. 0, 2, 4, 6, 8, 10 for spacing=2
    cdef float max_length = cumulative_distances[len(cumulative_distances) - 1]
    if loop: max_length -= spacing
    cumulative_spacings = []
    cdef float v = 0
    cdef float max_v = max_length + (spacing / 2)
    while v < max_v:
        cumulative_spacings.append(v)
        v += spacing

    cdef list[float] final_positions = []

    cdef int focus_idx = 0
    cdef float percent_of_way_through, relative_position
    cdef Line line
    for relative_position in cumulative_spacings:
        while True:
            if cumulative_distances[focus_idx + 1] < relative_position:
                focus_idx += 1
            else:
                break

        # Percentage of way through
        percent_of_way_through = (relative_position - cumulative_distances[focus_idx]) / lengths[focus_idx]

        line = lines[focus_idx]
        final_positions.append(lerp_point(
            (line[0], line[1]),
            (line[2], line[3]),
            percent_of_way_through
        ))

    return final_positions;


cpdef Point lerp_point(Point point_a, Point point_b, float portion):
    """Linearly interpolate a point between the two points

    Args:
        point_a: Start point
        point_b: End point
        portion: How much to interpolate between

    Returns:
        The interpolated point
    """
    cdef double pax, pay, pbx, pby
    pax, pay = point_a
    pbx, pby = point_b

    return (
        ((pbx - pax) * portion) + pax,
        ((pby - pay) * portion) + pay
    )


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef Points add_points_lists(list[Points] points_lists):
    cdef int points_list_count = len(points_lists)
    if points_list_count == 0: return []
    cdef int count = len(points_lists[0])

    cdef int i = 0
    cdef int j = 0
    cdef float sum_x = 0
    cdef float sum_y = 0

    for j in range(points_list_count):
        if count != len(points_lists[j]):
            raise Exception("Lists have different number of items")

    cdef summed = [0] * count
    for i in range(count):
        sum_x = 0
        sum_y = 0

        for j in range(points_list_count):
            sum_x += points_lists[j][i][0]
            sum_y += points_lists[j][i][1]

        summed[i] = (sum_x, sum_y)

    return summed

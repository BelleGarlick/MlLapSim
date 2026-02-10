cimport cython
from libc.math cimport (
    pow,
    sqrt
)


ctypedef tuple[double, double] Point
ctypedef list[tuple[double, double]] Points
ctypedef tuple[double, double, double, double] Line
ctypedef list[tuple[double, double, double, double]] Lines


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef Points line_centers(Lines lines):
    """
    Calculate the line centers from a list of lines.

    This function returns the list of line centers from a list
    of lines. The list of lines should be given as a 2D list
    where the length of the list if the amount of lines, and
    each line in list is stored as [x1, y1, x2, y2]. The normal
    lines to comply to these standards although have an
    additional set of values.

    Cython:numpy perf: 36x faster

    Args:
        lines: The 2D array that stores the list of lines.

    Return:
        Returns the line centers of the given lines.
    """
    cdef long n_lines = len(lines)

    cdef line_centers = [[-1]] * n_lines

    cdef double x1
    cdef double x2
    cdef double y1
    cdef double y2
    cdef long i

    for i in range(n_lines):
        x1, y1, x2, y2 = lines[i]

        line_centers[i] = [(x1 + x2) / 2, (y1 + y2) / 2]

    return line_centers


@cython.wraparound(False)
cpdef line_lengths(Lines lines):
    """Calculate the length of each item in the list of lines

    Cython:numpy perf: 6.6x faster
    """
    cdef int count = len(lines)
    cdef items = [0] * count

    cdef long i
    for i in range(count):
        items[i] = line_length(lines[i])
    return items


@cython.wraparound(False)
cpdef double line_length(Line line):
    cdef double d0 = line[0] - line[2]
    cdef double d1 = line[1] - line[3]

    return sqrt(d0 * d0 + d1 * d1)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cpdef Points normalise_lines(Lines lines):
    """Normalise a list of lines to a list of points (1.7x faster)"""
    cdef int count = len(lines)
    cdef normalised_lines = [0] * count
    cdef long i
    cdef double x1, y1, x2, y2, l, nx, ny

    for i in range(count):
        line = lines[i]
        x1, y1, x2, y2 = line
        l = line_length(line)

        if l == 0:
            normalised_lines[i] = None
        else:
            nx = (x2 - x1) / l
            ny = (y2 - y1) / l
            normalised_lines[i] = (nx, ny)

    return normalised_lines


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cpdef Lines set_line_lengths(Lines normals, widths):
    """Set the lengths of the given lines. (36x faster)"""
    normalised_points = normalise_lines(normals)
    old_widths = line_lengths(normals)

    cdef int count = len(normals)
    cdef double offset_x, offset_y, old_width, width, norm_x, norm_y
    cdef double nx1, ny1, nx2, ny2

    cdef int i = 0
    cdef updated_normals = [0] * count

    for i in range(count):
        nx1, ny1, nx2, ny2 = normals[i]
        norm_x, norm_y = normalised_points[i]
        old_width = old_widths[i]
        width = widths[i]

        offset_x = norm_x * ((old_width - width) / 2)
        offset_y = norm_y * ((old_width - width) / 2)

        updated_normals[i] = [nx1 + offset_x, ny1 + offset_y, nx2 - offset_x, ny2 - offset_y]

    return updated_normals


# TODO Need to test `min` argument.
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cpdef Lines extend_lines(Lines lines, double amount = 20, min=0):
    """Extend lines by a given amount. 54x faster"""
    cdef list[double] lengths = line_lengths(lines)

    cdef int count = len(lines)
    cdef double min_width = min
    cdef double amount2 = amount + amount

    cdef double x1, x2, y1, y2
    cdef double nx, ny, l, delta
    cdef int i = 0

    shortened_lines = [None] * count
    for i in range(count):
        x1, y1, x2, y2 = lines[i]
        l = lengths[i]

        if l != 0:
            # Prevent delta from going too small then normalise the delta
            delta = (l - max(l - amount2, min_width)) / 2
            delta /= l

            nx = (x2 - x1) * delta
            ny = (y2 - y1) * delta
            x1 -= nx
            y1 -= ny
            x2 += nx
            y2 += ny
            shortened_lines[i] = [x1, y1, x2, y2]

    return shortened_lines


cpdef Points start_points(Lines lines):
    """Get the start points from a list of lines"""
    res = []
    for a in lines:
        res.append(a[:2])
    return res


cpdef Points end_points(Lines lines):
    """Get the end points from a list of lines"""
    res = []
    for a in lines:
        res.append(a[2:])
    return res


# TODO Test this
cpdef Points lerp_points_on_lines(Lines lines, list[double] interpolations):
    """Linearly interpoalte points upon a list of lines.

    Args:
        lines: List of lines to interpolate points on
        interpolations: List of doubles defining how far from left to right
            the points fall

    Returns:
        Returns the interpolated points.
    """
    cdef int n = len(lines)
    cdef Points points = [(0, 0)] * n
    cdef double x1, y1, x2, y2
    cdef double interp

    for i in range(n):
        x1, y1, x2, y2 = lines[i]
        interp = interpolations[i]

        points[i] = (
            x1 + (x2 - x1) * interp,
            y1 + (y2 - y1) * interp
        )

    return points

cimport cython
import math
from libc.math cimport (
    pow,
    sqrt,
    cos,
    sin,
    atan2,
    acos
)


ctypedef tuple[double, double] Point
ctypedef tuple[double, double, double, double] Line
ctypedef list[tuple[double, double, double, double]] Lines


cpdef Point rotate(point: Point, angle: float, around: Point = None):
    """Rotate a point around another point.
    57x faster

    Args:
        point: The point to rotate
        angle: The angle to rotate the point around
        around: The point to rotate it around, or (0, 0) if not given

    Returns:
        The rotated point
    """
    cdef double around_x = around[0] if around else 0
    cdef double around_y = around[1] if around else 0

    cdef double cos_f = cos(angle)
    cdef double sin_f = sin(angle)
    cdef double dif_x = point[0] - around_x
    cdef double dif_y = point[1] - around_y

    cdef double nx = cos_f * dif_x - sin_f * dif_y + around_x
    cdef double ny = sin_f * dif_x + cos_f * dif_y + around_y
    return (nx, ny)


cpdef float angle_to(Point a, Point b):
    return atan2(b[1] - a[1], b[0] - a[0])


cpdef double angle_between(Point a, Point b, Point c):
    """Calculate the angle from a -> b -> c.

    This function will return the angle from a to c passing through b
    and the value will be a value from -pi - pi. This means the
    direction is preserved.

    Args:
        a: A point in 2D euclidian space.
        b: A point in 2D euclidian space.
        c: A point in 2D euclidian space.

    Return:
        The angle between a -> b -> c
    """
    return ((atan2(c[1] - b[1], c[0] - b[0]) -
             atan2(a[1] - b[1], a[0] - b[0])) + math.tau) % math.tau


cpdef double angle3(Point a, Point b, Point c):
    """Calculate the delta angle between points. Between -pi and pi. The
    direction is preserved. But a straight line will result in angle 0
    because the delta angle is 0.

    Args:
        a: Point 1
        b: Point 2
        c: Point 3

    Returns:
        The angle change from a straight line.
    """
    return ((atan2(c[1] - b[1], c[0] - b[0]) -
             atan2(a[1] - b[1], a[0] - b[0])) +
            math.tau) % math.tau - math.pi


cpdef float line_angle(Line line):
    """Get the angle of the line relative to the standard coord axis

    Args:
        line: The line to calc the angle of.

    Returns:
        The angle of the line
    """
    return atan2(line[3] - line[1], line[2] - line[0])


cpdef double angle_between_lines(Line line1, Line line2):
    """Calculate the angle between two lines, using dot product"""
    cdef Point d1 = (line1[2] - line1[0], line1[3] - line1[1])
    cdef Point d2 = (line2[2] - line2[0], line2[3] - line2[1])

    cdef double p = d1[0] * d2[0] + d1[1] * d2[1]
    cdef double n1 = sqrt(d1[0] * d1[0] + d1[1] * d1[1])
    cdef double n2 = sqrt(d2[0] * d2[0] + d2[1] * d2[1])

    if n1 * n2 == 0:
        return 0

    if round(p / (n1 * n2), 8) == 1:
        return 0

    # Compute angle
    return acos(p / (n1 * n2))


cpdef list[double] multi_angle_between_lines(Lines lines1, Lines lines2):
    """Calculate the angles between a list of lines"""
    cdef int count = len(lines1)
    cdef list[double] angles = [0] * count

    for i in range(count):
        angles[i] = angle_between_lines(lines1[i], lines2[i])

    return angles

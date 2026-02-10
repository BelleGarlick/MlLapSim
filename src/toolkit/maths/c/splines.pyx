cimport cython
from libcpp cimport bool

ctypedef tuple[double, double] Point
ctypedef list[tuple[double, double]] Points


# TODO Properly implement cardinal spline
# TODO Fully document
# TODO Possibly remove the extra calculated point that is then popped in the catmull_rom_spline
# TODO Add BSpline


cdef int QUADRUPLE_SIZE = 4


#cpdef cardinal_spline(Points points, int num_of_seg=5, float tension=0.5, bint close=True):
#    # Duplicate the array
#    pts: Points = [x for x in points]
#
#    l = points.length
#    r_pos = 0
#    r_len = (l-1) * num_of_seg + 1 + (num_of_seg if close else 0)
#    cache = [0 for _ in range((num_of_seg + 2) * 4)]
#    cache_ptr = 4
#
#    res: Points = [None] * r_len
#
#    if close:
#        pts.unshift(points[l - 1])  # insert end point as first point
#        pts.push(points[0])  # first point as last point
#    else:
#        pts.unshift(points[0])  # copy 1. point and insert at beginning
#        pts.push(points[l - 1])  # duplicate end-points
#
#    # cache inner-loop calculations as they are based on t alone
#    cache[0] = 1                                # 1,0,0,0
#
#    for i in range(1, num_of_seg):
#        st = i / num_of_seg
#        st2 = st * st
#        st3 = st2 * st
#        st23 = st3 * 2
#        st32 = st2 * 3
#
#        cache[cache_ptr] = st23 - st32 + 1    # c1
#        cache[cache_ptr+1] = st32 - st23        # c2
#        cache[cache_ptr+2] = st3 - 2 * st2 + st    # c3
#        cache[cache_ptr+3] = st3 - st2            # c4
#        cache_ptr += 4
#
#    cache[++cache_ptr] = 1                        # 0,1,0,0
#
#    def parse(Points pts, list[float] cache, int l):
#        for i in range(1, l):
#            pt1 = pts[i].x
#            pt2 = pts[i].y
#            pt3 = pts[i+1].x
#            pt4 = pts[i+1].y
#
#            t1x = (pt3 - pts[i-1].x) * tension
#            t1y = (pt4 - pts[i-1].y) * tension
#            t2x = (pts[i+2].x - pt1) * tension
#            t2y = (pts[i+2].y - pt2) * tension
#
#            for t in range(num_of_seg):
#                c = t << 2 #t * 4
#                c1 = cache[c]
#                c2 = cache[c+1]
#                c3 = cache[c+2]
#                c4 = cache[c+3]
#
#                res[r_pos] = (
#                    c1 * pt1 + c2 * pt3 + c3 * t1x + c4 * t2x,
#                    c1 * pt2 + c2 * pt4 + c3 * t1y + c4 * t2y
#                )
#                r_pos += 1
#
#    # calc. points
#    parse(pts, cache, l)
#
#    if close:
#        # l = points.length
#        pts = []
#        pts.push(points[l - 2], points[l - 1])  # second last and last
#        pts.push(points[0], points[1])  # first and second
#        parse(pts, cache, 2)
#
#    # add last point
#    l = 0 if close else points.length - 1
#    res[r_pos] = points[l]
#
#    return res


@cython.wraparound(False)
cdef int num_segments(Points points):
    # There is 1 segment per 4 points, so we must subtract 3 from the number of points
    cdef int n = len(points)
    return n - (QUADRUPLE_SIZE - 1)


@cython.wraparound(False)
cdef list[float] linspace(float min, float max, int n):
    cdef float gap = (max - min) / n
    cdef int i
    return [min + gap * i for i in range(n + 1)]


@cython.wraparound(False)
cdef double tj(float ti, Point pi, Point pj, float alpha):
    """Calculate t0 to t4. Then only calculate points between P1 and P2."""
    xi, yi = pi[0], pi[1]
    xj, yj = pj[0], pj[1]
    dx, dy = xj - xi, yj - yi
    l = (dx ** 2 + dy ** 2) ** 0.5
    return ti + l ** alpha


@cython.wraparound(False)
@cython.cdivision(True)
cdef Points sub_catmull_rom_spline(
    Point p0,
    Point p1,
    Point p2,
    Point p3,
    int num_points,
    float alpha = 0.5,
):
    """Compute the points in the spline segment

    Args:
        :param P0, P1, P2, and P3: The (x,y) point pairs that define the Catmull-Rom spline
        num_points: The number of points to include in the resulting curve segment
        alpha: 0.5 for the centripetal spline, 0.0 for the uniform spline, 1.0 for the chordal spline.

    Returns:
        The points
    """
    cdef double t0 = 0.0
    cdef double t1 = tj(t0, p0, p1, alpha)
    cdef double t2 = tj(t1, p1, p2, alpha)
    cdef double t3 = tj(t2, p2, p3, alpha)

    cdef double d0 = t1 - t0
    cdef double d1 = t2 - t1
    cdef double d2 = t3 - t2
    cdef double d3 = t2 - t0
    cdef double d4 = t3 - t1

    cdef double e0, e1a, e1b, e2a, e2b, e3a, e3b, e4a, e4b, e5
    cdef double a1x, a1y, a2x, a2y, a3x, a3y, b1x, b1y, b2x, b2y

    items = []
    cdef float t
    for t in linspace(t1, t2, num_points):
        if d0 == 0 or d1 == 0 or d2 == 0 or d3 == 0 or d4 == 0:
            raise Exception("Invalid input")

        e0 = (t1 - t) / d0
        e1a = (t2 - t) / d1
        e1b = (t2 - t) / d3
        e2a = (t3 - t) / d2
        e2b = (t3 - t) / d4
        e3a = (t - t0) / d0
        e3b = (t - t0) / d3
        e4a = (t - t1) / d1
        e4b = (t - t1) / d4
        e5 = (t - t2) / d2

        a1x = e0 * p0[0] + e3a * p1[0]
        a1y = e0 * p0[1] + e3a * p1[1]
        a2x = e1a * p1[0] + e4a * p2[0]
        a2y = e1a * p1[1] + e4a * p2[1]
        a3x = e2a * p2[0] + e5 * p3[0]
        a3y = e2a * p2[1] + e5 * p3[1]
        a3y = e2a * p2[1] + e5 * p3[1]
        b1x = e1b * a1x + e3b * a2x
        b1y = e1b * a1y + e3b * a2y
        b2x = e2b * a2x + e4b * a3x
        b2y = e2b * a2y + e4b * a3y

        items.append((
            e1a * b1x + e4a * b2x,
            e1a * b1y + e4a * b2y
        ))

    return items


@cython.boundscheck(False)
cpdef Points catmull_rom_spline(Points points, int num_points=10, bint loop=False):
    """Calculate Catmull-Rom for a sequence of initial points and return the combined curve.

    Args:
        points: Base points from which the quadruples for the algorithm are taken
        num_points: The number of points to include in each curve segment

    Returns:
        The chain of all points (points of all segments)
    """
    if len(points) == 0:
        return []

    cdef Points control_points = ([points[-1]] + points + points[:2]) if loop else points

    cdef Points all_splines = []

    cdef int i
    for i in range(num_segments(control_points)):
        subspline = sub_catmull_rom_spline(control_points[i], control_points[i+1], control_points[i+2], control_points[i+3], num_points)
        all_splines += subspline[:-1]  # Remove end item since it'll overlap with the previous/next subsplines

    return all_splines

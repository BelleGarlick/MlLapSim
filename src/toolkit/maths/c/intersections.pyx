cimport cython
from libc.math cimport (
    pow,
    sqrt
)


ctypedef tuple[double, double] Point
ctypedef list[tuple[double, double]] Points
ctypedef tuple[double, double, double, double] Line
ctypedef list[tuple[double, double, double, double]] Lines


cdef double cmin(double a, double b):
    return a if a < b else b


cdef double cmax(double a, double b):
    return a if a > b else b


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef segment_intersections(Line line, Lines segments, return_indexes: bool = False):
    """Calculate points of intrsections between one line and a list of other
    lines.

    Args:
        line: Line to compare other lines with
        segments: List of other lines to compare
        return_indexes: If true, return indexes of the segments that are
            intersected

    Returns:
        List of intersections or intersections and interseection indexes if
        return_indexes is true.
    """
    cdef double lin_0, lin_1, lin_2, lin_3
    lin_0, lin_1, lin_2, lin_3 = line

    cdef double min_x_seg = cmin(lin_0, lin_2)
    cdef double min_y_seg = cmin(lin_1, lin_3)
    cdef double max_x_seg = cmax(lin_0, lin_2)
    cdef double max_y_seg = cmax(lin_1, lin_3)

    a1 = lin_3 - lin_1
    b1 = lin_0 - lin_2
    c1 = a1 * lin_0 + b1 * lin_1

    intersections = []
    valid_indexes = []

    cdef double seg_0, seg_1, seg_2, seg_3
    cdef double a2, b2, c2, delta
    cdef double min_x, min_y, max_x, max_y
    cdef int count = len(segments)
    cdef int i = 0
    for i in range(count):
        seg_0, seg_1, seg_2, seg_3 = segments[i]

        a2 = seg_3 - seg_1
        b2 = seg_0 - seg_2
        c2 = a2 * seg_0 + b2 * seg_1
        delta = a1 * b2 - a2 * b1

        if delta == 0:
            continue

        min_x = cmax(cmin(seg_0, seg_2), min_x_seg) - 1e-9
        min_y = cmax(cmin(seg_1, seg_3), min_y_seg) - 1e-9
        max_x = cmin(cmax(seg_0, seg_2), max_x_seg) + 1e-9
        max_y = cmin(cmax(seg_1, seg_3), max_y_seg) + 1e-9

        x = (b2 * c1 - b1 * c2) / delta
        y = (a1 * c2 - a2 * c1) / delta

        if min_x <= x <= max_x and min_y <= y <= max_y:
            intersections.append((x, y))
            valid_indexes.append(i)

    if return_indexes:
        return intersections, valid_indexes

    return intersections


# TODO Implement and test this at somepoint
"""
#cpdef Points circle_line_intersections(Point point, double radius, Lines lines):
#    c_x, c_y = point
#
#    lines = np.array(lines)
#    axs = lines[:, 0]
#    ays = lines[:, 1]
#    bxs = lines[:, 2]
#    bys = lines[:, 3]
#
#    min_xs = np.round(np.minimum(axs, bxs), 3)
#    min_ys = np.round(np.minimum(ays, bys), 3)
#    max_xs = np.round(np.maximum(axs, bxs), 3)
#    max_ys = np.round(np.maximum(ays, bys), 3)
#
#    dxs, dys = bxs - axs, bys - ays
#    As = np.square(dxs) + np.square(dys)
#    Bs = 2 * (dxs * (axs - c_x) + dys * (ays - c_y))
#    Cs = (axs - c_x) * (axs - c_x) + (ays - c_y) * (ays - c_y) - radius * radius
#    dets = np.square(Bs) - 4 * As * Cs
#
#    # No solutions
#    no_solutions = np.logical_or(As <= 0.0000001, dets < 0)
#    one_solution = dets == 0
#    two_solutions = np.logical_not(np.logical_or(no_solutions, one_solution))
#
#    # calculate ts
#    twoAs, negBs, rootDets = 2 * As, -Bs, np.sqrt(np.abs(dets))
#
#    one_solution_t = np.divide(negBs, twoAs, out=np.zeros_like(As), where=one_solution)
#    two_solitions_ta = np.divide(negBs + rootDets, twoAs, out=np.zeros_like(As), where=two_solutions)
#    two_solitions_tb = np.divide(negBs - rootDets, twoAs, out=np.zeros_like(As), where=two_solutions)
#
#    one_solutions_pxs = np.round(axs + one_solution_t * dxs, 3)
#    one_solutions_pys = np.round(ays + one_solution_t * dys, 3)
#
#    two_solutions_paxs = np.round(axs + two_solitions_ta * dxs, 3)
#    two_solutions_pays = np.round(ays + two_solitions_ta * dys, 3)
#    two_solutions_pbxs = np.round(axs + two_solitions_tb * dxs, 3)
#    two_solutions_pbys = np.round(ays + two_solitions_tb * dys, 3)
#
#    one_solutions_valid = np.logical_and.reduce(
#        (min_xs <= one_solutions_pxs, one_solutions_pxs <= max_xs,
#         min_ys <= one_solutions_pys, one_solutions_pys <= max_ys)
#    )
#
#    two_solutions_a_valid = np.logical_and.reduce(
#        (min_xs <= two_solutions_paxs, two_solutions_paxs <= max_xs,
#         min_ys <= two_solutions_pays, two_solutions_pays <= max_ys)
#    )
#
#    two_solutions_b_valid = np.logical_and.reduce(
#        (min_xs <= two_solutions_pbxs, two_solutions_pbxs <= max_xs,
#         min_ys <= two_solutions_pbys, two_solutions_pbys <= max_ys)
#    )
#
#    intersections = []
#    for index in range(len(lines)):
#        if one_solution[index] and one_solutions_valid[index]:
#            intersections.append([one_solutions_pxs[index], one_solutions_pys[index]])
#
#        if two_solutions[index]:
#            if two_solutions_a_valid[index]:
#                intersections.append([two_solutions_paxs[index], two_solutions_pays[index]])
#
#            if two_solutions_b_valid[index]:
#                intersections.append([two_solutions_pbxs[index], two_solutions_pbys[index]])
#
#    return np.array(intersections)
"""
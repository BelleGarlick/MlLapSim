import numpy as np
from toolkit import maths


# TODO Optimise using better methods, add documentation and testing


def bspline(points, interpolations=5_000):
    from scipy.interpolate import splrep, splev

    initial_point = points[0].copy()
    points = np.vstack((points, points))

    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]

    t = range(len(points))
    ipl_t = np.linspace(0.0, len(points) - 1, interpolations * 2)

    x_tup = splrep(t, x, k=3)
    y_tup = splrep(t, y, k=3)

    x_list = list(x_tup)
    xl = x.tolist()
    x_list[1] = xl + [0.0, 0.0, 0.0, 0.0]

    y_list = list(y_tup)
    yl = y.tolist()
    y_list[1] = yl + [0.0, 0.0, 0.0, 0.0]

    x_i = splev(ipl_t, x_list)
    y_i = splev(ipl_t, y_list)

    # Conect track as a loop as close to start - this doesn't mean the line starts at same place though
    start_index, end_index = interpolations // 2, None
    start_point = np.array([x_i[start_index], y_i[start_index]])
    closest_distance = None
    for i in range(interpolations, 2 * interpolations):
        end_point = np.array([x_i[i], y_i[i]])
        dist = maths.distance(start_point, end_point)
        if closest_distance is None or dist < closest_distance:
            closest_distance = dist
            end_index = i - 1

    x_i = np.expand_dims(x_i[start_index:end_index], axis=0)
    y_i = np.expand_dims(y_i[start_index:end_index], axis=0)
    splined_points = np.concatenate((x_i, y_i)).T

    # Start at orignal starting point
    distances = [maths.distance(splined_points[i], initial_point) for i in range(len(splined_points))]
    start_point = np.argmin(distances)
    splined_points = np.concatenate((splined_points[start_point:], splined_points[:start_point]))

    return splined_points

from . import splines

from toolkit.maths.c.lines import *
from toolkit.maths.c.points import *
from toolkit.maths.c.intersections import *
from toolkit.maths.c.functional import *
from toolkit.maths.c.angles import *
from toolkit.maths.c.normals import *
from toolkit.maths.c.splines import *

# noinspection PyUnresolvedReferences
__all__ = [
    # Lines
    "line_centers",
    "line_lengths",
    "line_length",
    "normalise_lines",
    "set_line_lengths",
    "extend_lines",
    "start_points",
    "start_points",
    "end_points",

    # Points
    "distance",
    "distances",
    "normalise_points",
    "normalise_point",
    "closest_point",
    "points_to_lines",
    "sub_point",
    "sub_points",
    "interpolate_points_between",
    "get_points_on_paths",
    "lerp_point",
    "lerp_points_on_lines",
    "add_points_lists",

    # Intersections
    "segment_intersections",
    # "circle_line_intersections"

    # Functional
    "at_indexes",
    "roll",

    # Angles
    "rotate",
    "angle_to",
    "angle_between",
    "angle3",
    "line_angle",
    "angle_between_lines",
    "multi_angle_between_lines",

    # Normals
    "create_normals_on_path",
    "create_line_normals_from_points",
    "trim_normals_to_boundary",  # TODO Use a better method for this based on normal generation as used in track generation and xyrl conversion

    # Splines
    "catmull_rom_spline"
]

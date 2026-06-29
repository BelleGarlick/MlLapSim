import math
from typing import List, Tuple

import numpy as np

from toolkit import maths

"""This module encodes a given spliced track into the encoding outlined in 
Garlick & Bradley 2021.
"""


# TODO Make it possible to track to not loop
# TODO Check that the lines dont cross


def extract_features(seg_lines: np.ndarray) -> Tuple[List[float], List[float], List[float]]:
    """This function will extract the relevant features from the normal lines.
    This method extracts the relevant features from the given normals array. As
    defined in the paper, we extract the widths, the alpha angles and offset
    theta angles from normal lines.

    Args:
        seg_lines: The normal lines that make up the track.

    Returns:
        The tuple of arrays: widths, angles and offsets stored as vectors.
    """
    seg_lines = [[*x] for x in seg_lines]

    count = len(seg_lines)
    normal_centers = maths.line_centers(seg_lines)

    widths, angles, offsets = maths.line_lengths(seg_lines), [], []
    for i, normal in enumerate(seg_lines):
        # Calculate line seals.
        pc = normal_centers[i - 1]
        c = normal_centers[i]
        nc = normal_centers[(i + 1) % count]

        lp = normal[:2]

        between_angle = maths.angle_between(nc, c, pc)

        offset_angle_to_next = maths.angle3(nc, c, lp)
        offset_angle_to_prev = maths.angle3(lp, c, pc)
        offset = offset_angle_to_prev - (offset_angle_to_next + offset_angle_to_prev) / 2

        # Append the calculated data into the relevant arrays
        angles.append(between_angle - math.pi)
        offsets.append(offset)

    # If not looped then we need to remove the outer normals data since they're
    #     calculated by wrapping the normals as a loop
    # if not loop:
    #     widths = widths[1: -1]
    #     angles = angles[1: -1]
    #     offsets = offsets[1: -1]

    return widths, angles, offsets

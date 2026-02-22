from typing import Tuple, List, Optional

import numpy as np

from toolkit import maths
from toolkit.tracks.models import Track, SegmentationLine
from ..path import shortest_path
from ..smoother import _split_normals, _extend_normals_until_collision, _collapse_collisions_pairs
from ...maths import create_line_normals_from_points

"""XYRL Track Conversion

The XYRL format, found in TUMFTM's track database and used by/for their 
simulator stores tracks in the format x,y,r,l where x,y make up the xy 
coordinates for a reference line which normal line are created from. L 
represents the left distance in meters of the line from the ref line and R
is the right distance of the normal end to the reference line.

This is in contrast to the normal lines used by this project's track object
whereby lines are created around the midline and the normals are equidistance
eitherside to the midline. This has the benefit that a well constructed 
reference line does not need normals to be smoothed as the reference line 
adjusts to compensate. However, compensating for these intersecting ref lines
are a slow and heavy process which our method avoids.

This function exists to allow tracks to work together interoperabily e.g. with
the dataset output.
"""


def from_xyrl(
        raw: Optional[str] = None,
        data: Optional[List[Tuple[float, float, float, float]]] = None,
        skip_first: bool = True
) -> Track:
    """Convert from xyrl format to the track model

    Args:
        raw: Raw string containing the csv data.
        data: The data (not as a string) of XYRL
        skip_first: If yes, ignore the first row (header row) of the csv string

    Reutrns:
        Track object.
    """
    # Check that one item is and one is not None
    assert (raw is not None or data is not None), "One param `raw` or `data` must not be null"
    assert (raw is None or data is None), "Only one param `raw` or `data` may not be null"

    ref_points, left_lengths, right_lengths = [], [], []
    if raw:
        for line in raw.split("\n")[int(skip_first):]:
            tokens = line.split(",")

            # Skip tokens at end of file which may be an empty line
            if len(tokens) > 2:
                tokens = [float(x) for x in tokens]
                ref_points.append(tuple(tokens[:2]))
                right_lengths.append(tokens[2])
                left_lengths.append(tokens[3])
    else:
        ref_points = [x[:2] for x in data]
        right_lengths = [x[2] for x in data]
        left_lengths = [x[3] for x in data]

    # Create normals upon the ref line
    normals = maths.create_line_normals_from_points(ref_points)
    left_segments = np.array(maths.normalise_points(maths.sub_points(maths.start_points(normals), ref_points)))
    right_segments = np.array(maths.normalise_points(maths.sub_points(maths.end_points(normals), ref_points)))
    for i, normal in enumerate(normals):
        left_segment = left_segments[i] * left_lengths[i]
        right_segment = right_segments[i] * right_lengths[i]

        l = left_segment + ref_points[i]
        r = right_segment + ref_points[i]
        normals[i] = [l[0], l[1], r[0], r[1]]

    return Track(
        segmentations=[
            SegmentationLine(
                x1=normal[0],
                y1=normal[1],
                x2=normal[2],
                y2=normal[3]
            )
            for normal in normals
        ]
    )


def _cut_normals(normals, left_xy, right_xy):
    # Extend normals and find all collisions they half normals make with the boundary
    left_normals, right_normals = _split_normals(normals)
    left_normal_collisions = _extend_normals_until_collision(left_normals, left_xy)
    right_normal_collisions = _extend_normals_until_collision(right_normals, right_xy)

    # Collapse all collisions into one collision per normal
    left_collisions = _collapse_collisions_pairs(left_normals, left_normal_collisions, len(left_xy))
    right_collisions = _collapse_collisions_pairs(right_normals, right_normal_collisions, len(right_xy))

    return [
        left_collisions[i] + right_collisions[i]
        for i in range(len(left_collisions))
    ]


def to_xyrl(normals, spacing=2) -> List[Tuple[float, float, float, float]]:
    """Convert a track to the XYRL format.

    Note, this function utilises the track normal generation functions
    which deal with overlapping track geometry.

    Args:
        normals: The normals which define the track
        normal_spacing: Distance between normals

    Returns: List of the four items (X, Y, R, L)
    """
    track = Track(
        segmentations=[
            SegmentationLine(x1=x1, y1=y1, x2=x2, y2=y2)
            for (x1, y1, x2, y2) in normals
        ]
    )
    left_xy, right_xy = track.left_line(), track.right_line()

    shortest = shortest_path(track, padding=1)
    normals = maths.create_normals_on_path(shortest.positions, 80, spacing)
    #
    # import matplotlib as mpl
    # import matplotlib.pyplot as plt
    #
    # mpl.use('macosx')
    #
    # plt.plot([x[0] for x in shortest.positions], [x[1] for x in shortest.positions])
    # for normal in normals:
    #     plt.plot([normal[0], normal[2]], [normal[1], normal[3]])
    # # plt.plot([x[0] for x in ref_line], [x[1] for x in ref_line])
    # # plt.plot([x[0] for x in left_boundary], [x[1] for x in left_boundary])
    # # plt.plot([x[0] for x in right_boundary], [x[1] for x in right_boundary])
    # for n in normals:
    #     plt.plot(n[0::2], n[1::2])
    # plt.axis("equal")
    # plt.show()

    normals = _cut_normals(normals, left_xy, right_xy)

    track = Track(
        segmentations=[
            SegmentationLine(x1=x1, y1=y1, x2=x2, y2=y2)
            for (x1, y1, x2, y2) in normals
        ]
    )
    shortest = shortest_path(track, padding=1)
    normals = create_line_normals_from_points(shortest.positions, 80)
    normals = _cut_normals(normals, left_xy, right_xy)

    # Calculate left and right distance
    lefts = [
        maths.distance(center, n_start[:2])
        for center, n_start in
        zip(shortest.positions, normals)]
    rights = [
        maths.distance(center, n_start[2:])
        for center, n_start in
        zip(shortest.positions, normals)]

    return [
        (x, y, right, left)
        for i, ((x, y), left, right) in
        enumerate(zip(shortest.positions, lefts, rights))
    ]

from typing import Optional

from toolkit import maths

from toolkit.tracks.models import Track, SegmentationLine
from toolkit.tracks.smoother.smoother import (
    _smooth_normals,
    _split_normals,
    _extend_normals_until_collision,
    _collapse_collisions_pairs
)

# TODO Add tests

SPLINE = 5


def smooth_track(track: Track, spacing: Optional[int] = None) -> Track:
    normals = [seg.arr() for seg in track.segmentations]
    max_width = max(maths.line_lengths(normals)) * 2

    if spacing > 0:
        # Calculate adjusted distance for spacing so that start/finish is same as all other normals
        track_length = sum(maths.line_lengths(maths.points_to_lines(track.midline())))
        corrected_normal_spacing = track_length / (track_length // spacing)

        path = maths.catmull_rom_spline(track.midline(), SPLINE, True)

        # Calculate new normals on track
        normals = maths.create_normals_on_path(path, 80, corrected_normal_spacing)

    left_boundary = maths.catmull_rom_spline(track.left_line(), SPLINE, True)
    right_boundary = maths.catmull_rom_spline(track.right_line(), SPLINE, True)

    smooth_normals = _smooth_normals(normals, iterations=400, width=max_width)

    # Extend normals and find all collisions they half normals make with the boundary
    left_normals, right_normals = _split_normals(smooth_normals)

    left_normal_collisions = _extend_normals_until_collision(left_normals, left_boundary)
    right_normal_collisions = _extend_normals_until_collision(right_normals, right_boundary)

    # Collapse all collisions into one collision per normal
    left_collisions = _collapse_collisions_pairs(left_normals, left_normal_collisions, len(left_boundary))
    right_collisions = _collapse_collisions_pairs(right_normals, right_normal_collisions, len(right_boundary))

    # Combine the calculated collisions into the resultant normals
    normals = [
        left_collisions[i] + right_collisions[i]
        for i in range(len(left_collisions))
    ]

    return Track(
        segmentations=[
            SegmentationLine(x1=normal[0], y1=normal[1], x2=normal[2], y2=normal[3])
            for normal in normals
        ]
    )

# TODO In encoder, check for an intersecting lines that still exist and throw an errror

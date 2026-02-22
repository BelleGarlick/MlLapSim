from typing import List

import numpy as np

from toolkit import maths
from .models import SplicerInput
from toolkit.tracks.models import SegmentationLine, Track
from .models.splicer_input import PathInput

"""The raw data splice module.

This module takes raw track data from different sources, such as
path and optimal data and combines it together to the spliced
track format. Which can be used later by encoding and fed to the
network.

This module can be expanded to incorperate more data by adding
more options to the SplicedInput path (such as controls) which
is loaded in the splice function. Then, in the `get_path_data`
function to take that input and extract out seperate points
which should be interpolated onto the track.
"""


def splice(params: SplicerInput) -> Track:
    """Calculate the segmentation lines and split the lapsim data
    into those lines.

    Args:
        params: An input object specifically for defining the inputs.
            This object is used rather than **kwargs to make it easier
            to both transfer and document. It also allows for methods
            applied to the object to load information such as loading
            the data where the object itself handles loading the paths.

    Returns:
        SplicedData object with spliced data filled in accordingly
    """
    # Extract objects from params
    track = params.track

    # Extract data that we need to interpolate
    path_x, path_y, vels, accs = get_path_data(track, params.path)

    # Create array off all lines for the segment intersections
    optimal_path_line = [
        [path_x[i - 1], path_y[i - 1], path_x[i], path_y[i]]
        for i in range(len(path_x))
    ]

    normals = [seg.arr() for seg in track.segmentations]

    encoded_normals: List[SegmentationLine] = []
    for i, normal in enumerate(normals):
        # Find intersection point
        intersection_point, index = None, None

        # TODO Optimise this to not cap it
        for extension in [0, 0.2, 0.4, 0.6, 0.8, 1]:
            # Extend line slightly to see if we can extend it to the path
            line = maths.extend_lines([normal], extension)[0]
            intersection_point, indexes = maths.segment_intersections(line, optimal_path_line, return_indexes=True)

            if len(indexes) == 1:
                index = indexes[0]
                intersection_point = intersection_point[0]
                break

            elif len(indexes) == 2:
                optimal_path_line_1 = optimal_path_line[indexes[0]]
                optimal_path_line_2 = optimal_path_line[indexes[1]]
                index = indexes[0]

                if np.array_equal(optimal_path_line_1[0:2], optimal_path_line_2[0:2]):
                    intersection_point = optimal_path_line_1[0:2]

                elif np.array_equal(optimal_path_line_1[0:2], optimal_path_line_2[2:4]):
                    intersection_point = optimal_path_line_2[2:4]

                elif np.array_equal(optimal_path_line_1[2:4], optimal_path_line_2[0:2]):
                    intersection_point = optimal_path_line_2[0:2]

                elif np.array_equal(optimal_path_line_1[2:4], optimal_path_line_2[2:4]):
                    intersection_point = optimal_path_line_2[2:4]

                else:
                    intersection_point = optimal_path_line_1[0:2]
                break

        if index is None:
            raise Exception("Err... we have an error")

        # Compute optimal point on normal line.
        p = maths.distance(intersection_point, normal[0:2]) / maths.line_length(normal)
        if maths.distance(intersection_point, normal[2:4]) > maths.line_length(normal):
            p = 0
        if maths.distance(intersection_point, normal[0:2]) > maths.line_length(normal):
            p = 1

        # Compute where on the optimal path, the normal line intersects.
        optimal_line = optimal_path_line[index]
        l = max(min(maths.distance(optimal_line[0:2], intersection_point) / maths.line_length(optimal_line), 1), 0)

        # Interpolate velocity and acceleration on at the point of intersection.
        vel = (vels[index - 1] * (1 - l)) + (vels[index] * l)
        acc = (accs[index - 1] * (1 - l)) + (accs[index] * l)

        # Add new normal with all data.
        data = {
            "x1": normal[0],
            "y1": normal[1],
            "x2": normal[2],
            "y2": normal[3],
            "pos": p,
            "vel": vel,
            "acc": acc
        }
        if params.precision is not None:
            data = {key: round(data[key], params.precision) for key in data}

        encoded_normals.append(SegmentationLine.model_validate(data))

    # Create new object
    result = Track(segmentations=encoded_normals)

    if params.on_complete:
        params.on_complete(result, *(params.on_complete_args or []))

    return result


def get_path_data(track, optimal_path: List[PathInput]):
    """This function gets the relevant path data from the inputs.

    If we want to expand capabilities, e.g. controls/acceleration, then
    we need to extract out the relevant data here. Defaulting it initi-
    aly, but if the data is given then we use that.

    Args:
        track: data to get the default path from if no optimal data is
            given, allowing us to create spliced data for tracks with
            no known optimal output.
        optimal_path: Optimal path csv data

    Returns:
        Tuple of path and velocity/acceleration data.
    """
    vels = [0 for _ in range(len(track.segmentations))]
    accs = [0 for _ in range(len(track.segmentations))]
    midline = track.midline()
    path_x = [x[0] for x in midline]
    path_y = [x[1] for x in midline]

    if optimal_path:
        vels = list(map(lambda p: p.vel or -1, optimal_path))
        accs = list(map(lambda p: p.acc or 0, optimal_path))
        path_x = list(map(lambda p: p.x, optimal_path))
        path_y = list(map(lambda p: p.y, optimal_path))
    else:
        print("No optimal path data given.")

    vels = np.array(vels, dtype=np.float32)
    accs = np.array(accs, dtype=np.float32)
    path_x = np.array(path_x, dtype=np.float32)
    path_y = np.array(path_y, dtype=np.float32)

    return path_x, path_y, vels, accs

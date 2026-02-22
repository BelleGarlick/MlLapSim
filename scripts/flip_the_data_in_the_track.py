import json
import os

from pyproj import Proj, Transformer

from toolkit.maths import get_points_on_paths
from toolkit.tracks.conversion import from_xyrl
from toolkit.tracks.smoother.smoother import _extend_normals_until_collision, _collapse_collisions_pairs

from toolkit import maths
from toolkit.tracks.smoother import _smooth_normals, _split_normals


path = '/Users/belle/Developer/Hyperdrive-Labs-LapSim/datasets/Real Tracks/Project Files'
out_path = '/Users/belle/Developer/MlLapSim/output-tests'


# TODO Take a look at the latlon stuff, it's off. check the final output lat/lon is similar place


def to_xy_positions(transformer, coords):
    return [[*transformer.transform(lon, lat)] for lon, lat in coords]


def cut_normals(normals, left_xy, right_xy, do_thing=False):
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


def create_segmentation_lines(reference_xy, left_xy, right_xy):
    reference_xy = get_points_on_paths(reference_xy, 2, loop=True)

    normals = maths.create_line_normals_from_points(reference_xy, length=50)
    normals = _smooth_normals(normals, 50, 80)

    normals = cut_normals(normals, left_xy, right_xy)

    return normals


if __name__ == "__main__":
    tracks = os.listdir(path)

    for track in tracks:
        with open(path + "/" + track, 'r') as f:
            project_file = json.load(f)

        for layer in project_file['project']['layers']:
            for datum in layer['curve']:
                datum['left'], datum['right'] = datum['right'], datum['left']
                # breakpoint()

        with open(out_path + "/" + track, 'w+') as f:
            f.write(json.dumps(project_file, indent=4))
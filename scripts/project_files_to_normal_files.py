import json
import os
import string
import random

from pyproj import Proj, Transformer

from toolkit.tracks.conversion import from_xyrl, to_xyrl
from toolkit.tracks.smoother.smoother import _extend_normals_until_collision, _collapse_collisions_pairs, \
    _smooth_normals

from toolkit import maths
from toolkit.tracks.smoother import _split_normals


path = '/Users/belle/Developer/Hyperdrive-Labs-LapSim/datasets/Final Release/Real Track Projects'
output = "/Users/belle/Developer/Hyperdrive-Labs-LapSim/datasets/Final Release/Real Tracks XYRL"


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


if __name__ == "__main__":
    project_files = os.listdir(path)
    random.shuffle(project_files)

    for file_name in project_files:
        print(file_name)
        if not file_name.endswith(".json"):
            continue

        with open(os.path.join(path, file_name), 'r') as f:
            project_file = json.load(f).get("project", {})

        proj = Proj(proj="aeqd", lon_0=project_file['origin']['lon'], lat_0=project_file['origin']['lat'], units="m")
        to_xy = Transformer.from_proj(Proj(proj="latlong", datum="WGS84"), proj, always_xy=True)

        layers = []
        for layer_data in project_file.get("layers", []):
            track_layer_name = f"{project_file['name']} - {layer_data.get("name")}"
            if "Karting" in track_layer_name:
                continue

            print(" - " + track_layer_name)
            output_csv_path = os.path.join(output, f"{track_layer_name}.csv")
            if os.path.exists(output_csv_path):
                continue

            if "GP Circuit" in track_layer_name:
                if "DE Nurburgring" in track_layer_name or "Nordschleife" in track_layer_name:
                    pass
                else:
                    raise Exception(track_layer_name)

            if [x for x in track_layer_name if x not in set(string.ascii_letters + string.digits + " -+_'")]:
                raise Exception("Invalid track name: " + track_layer_name)

            corrected_lat_lon = [
                {
                    "lat": x['lat'] + project_file['origin']['lat'],
                    "lon": x['lon'] + project_file['origin']['lon'],
                    "right": x['right'],
                    "left": x['left']
                }
                for x in layer_data["curve"]
            ]
            ref_line = [
                list(to_xy.transform(x['lon'], x['lat'])) + [x['right'], x['left']]
                for x in corrected_lat_lon
            ]

            track = from_xyrl(data=ref_line)

            left_boundary = maths.catmull_rom_spline(track.left_line(), 20, loop=True)
            right_boundary = maths.catmull_rom_spline(track.right_line(), 20, loop=True)
            ref_line = maths.catmull_rom_spline([x[:2] for x in ref_line], 20, loop=True)

            normals = maths.create_normals_on_path(ref_line, 80, 2)
            normals = _smooth_normals(normals, 50, 80)
            normals = cut_normals(normals, left_boundary, right_boundary)


            xyrl_format = to_xyrl(normals, spacing=0.5)

            # todo interpolate everything so all the normals are done and are consistent

            if os.path.exists(output_csv_path):
                continue

            import csv
            with open(output_csv_path, 'w', newline='') as csvfile:
                spamwriter = csv.writer(csvfile)
                spamwriter.writerow(['x_m', 'y_m', 'w_tr_right_m', 'w_tr_left_m'])
                for x, y, r, l in xyrl_format:
                    spamwriter.writerow([str(v) for v in [x, y, r, l]])


            # import matplotlib as mpl
            # import matplotlib.pyplot as plt
            #
            # mpl.use('macosx')
            #
            # plt.plot([x[0] for x in ref_line], [x[1] for x in ref_line])
            # plt.plot([x[0] for x in left_boundary], [x[1] for x in left_boundary])
            # plt.plot([x[0] for x in right_boundary], [x[1] for x in right_boundary])
            # for n in normals:
            #     plt.plot(n[0::2], n[1::2])
            # plt.axis("equal")
            # plt.show()

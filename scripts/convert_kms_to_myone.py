import json
import os
import xml.etree.ElementTree as ET

import numpy as np
from pyproj import Proj, Transformer

from toolkit.maths import get_points_on_paths
from toolkit.tracks.conversion import to_xyrl, from_xyrl
from toolkit.tracks.smoother.smoother import _extend_normals_until_collision, _collapse_collisions_pairs

from toolkit import maths
from toolkit.tracks.smoother import _smooth_normals, _split_normals

path = "/Users/belle/Developer/Hyperdrive-Labs-LapSim/datasets/KML copy"


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

    for file in sorted(tracks):
        # these are fine im just bored of seeing them when checking through tracks
        if file in [
            "AU Adelaide Street Circuit.kml",
            "AU Bathurst Mount Panorama.kml",
            "AU Sydney Olympic Park Street Circuit.kml",
            "AZ Baku.kml",
            "BH Bahrain International Circuit.kml",
            "CA Mosport International Raceway.kml",
            "CH Zürich.kml",
            "DE AVUS.kml",
            "DE Nordschleife.kml",
            "DE Norisring.kml",
            "DE Nürburgring.kml",
            "ES Circuit de Barcelona-Catalunya.kml",
            "ES València Street Circuit.kml",
            "FR Paul Ricard.kml",
            "MX Autódromo Hermanos Rodríguez.kml",
            "MC Monaco.kml",
            "NL Zandvoort.kml",
            "IN Buddh International Circuit.kml",
            "PT Autódromo do Estoril.kml",
            "QA Losail International Circuit.kml",
            "SA Jeddah Corniche Circuit.kml",
            "SA Yas Marina.kml",
            "SG Marina Bay.kml",
            "TR Istanbul Park.kml",
            "US Indianapolis Motor Speedway.kml",
            "US Las Vegas Motor Speedway.kml",
            "US Long Beach Circuit.kml",
            "US Sonoma Raceway.kml",
            "UK Bedford Autodrome.kml"
        ]:
            pass

        print(file)

        with open(os.path.join(path, file), 'r') as f:
            data_string = f.read()
            data_string = data_string.replace("xmlns=\"http://www.opengis.net/kml/2.2\"", "")
            tracks = ET.fromstring(data_string)\
                .find("Document")\
                .findall("Folder")

        file_layers = []

        for track in tracks:
            track_name = track.find("name").text
            placemarks = track.findall("Placemark")
            print(track_name)

            def convert_to_gps_points(nodes):
                coordinates = [x.find("LineString").find("coordinates").text for x in nodes]
                ordinates = [x.split("\n") for x in coordinates]
                ordinates = [x for xx in ordinates for x in xx]
                ordinates = [x.strip() for x in ordinates if len(x.strip()) > 1]
                ordinates = [x.split(",")[:2] for x in ordinates]
                return [(float(x), float(y)) for x, y in ordinates]

            reference = convert_to_gps_points([x for x in placemarks if x.find("name").text.replace("\n", "") == "Reference"])
            left = convert_to_gps_points(
                [x for x in placemarks if x.find("name").text.replace("\n", "") == "Left"]
                + [x for x in placemarks if x.find("name").text.replace("\n", "") == "Left 2"]
            )
            right = convert_to_gps_points(
                [x for x in placemarks if x.find("name").text.replace("\n", "") == "Right"]
                + [x for x in placemarks if x.find("name").text.replace("\n", "") == "Right 2"]
            )

            proj = Proj(proj="aeqd", lon_0=reference[0][0], lat_0=reference[0][1], units="m")
            to_xy = Transformer.from_proj(Proj(proj="latlong", datum="WGS84"), proj, always_xy=True)
            to_latlon = Transformer.from_proj(proj, Proj(proj="latlong", datum="WGS84"), always_xy=True)

            reference_xy = to_xy_positions(to_xy, reference)
            left_xy = to_xy_positions(to_xy, left)
            right_xy = to_xy_positions(to_xy, right)

            track = create_segmentation_lines(reference_xy, left_xy, right_xy)

            # Create the format
            xyrl_format = to_xyrl(track, spacing=2)

            # todo change from relative position at origin, to move to Lat lon in the projcet file

            new_format = []
            for x, y, r, l in xyrl_format:
                lon, lat = to_latlon.transform(x, y)
                new_format.append({
                    "lat":  lat,
                    "lon": lon,
                    "right": r,
                    "left": l,
                })

            # todo convert back to segmentation
            # import matplotlib as mpl
            # import matplotlib.pyplot as plt
            # mpl.use('macosx')
            #
            # plt.plot(
            #     [x[0] for x in xyrl_format],
            #     [x[1] for x in xyrl_format],
            # )
            # # for normal in ret:
            # #     plt.plot(normal[0::2], normal[1::2])
            # resegmentation_lines = from_xyrl(data=xyrl_format)
            # for normal in resegmentation_lines.segmentations:
            #     plt.plot(normal.arr()[0::2], normal.arr()[1::2], c='green')
            # plt.plot([x[0] for x in left_xy], [x[1] for x in left_xy], c='green')
            # plt.plot([x[0] for x in track], [x[1] for x in track], c='blue', linewidth=0.5)
            # plt.plot([x[0] for x in right_xy], [x[1] for x in right_xy], c='green')
            # plt.plot([x[2] for x in track], [x[3] for x in track], c='blue', linewidth=0.5)
            # plt.axis("equal")
            # plt.show()

            file_layers.append({
                "name": track_name,
                "curve": new_format
            })

        lats = [x['lat'] for y in file_layers for x in y['curve']]
        lons = [x['lon'] for y in file_layers for x in y['curve']]
        avg_lat = np.mean(lats)
        avg_lon = np.mean(lons)

        print(f"Lat: {avg_lat}, Lon: {avg_lon}")
        print(reference[0])

        new_projects = {
            "project": {
                "name": file.replace(".kml", ""),
                "origin": {
                    "lon": float(avg_lon),
                    "lat": float(avg_lat)
                },
                "layers": [{
                    "name": layer['name'],
                    "curve": [{
                        "lat": float(x['lat'] - avg_lat),
                        "lon": float(x['lon'] - avg_lon),
                        "right": float(x['right']),
                        "left": float(x['left']),
                    } for x in layer['curve']],
                } for layer in file_layers]
            }
        }

        with open("/Users/belle/Developer/MlLapSim/output-tests/" + file.replace(".kml", "") + ".json", "w") as f:
            f.write(json.dumps(new_projects, indent=4))

        # print(new_projects)
        #
        # import sys
        # sys.exit(0)

        pass
            # breakpoint()

    print(tracks)

import json

import numpy as np
import webdataset

import toolkit.tracks.conversion
from lapsim.encoder.encoder import extract_features
from toolkit.tracks.splicer import PathInput
from toolkit.tracks.splicer import splice

from lapsim.preprocessor.encoder import encode
from toolkit.tracks.models import Track
from toolkit.utils import readers
from toolkit.utils.logger import log
from lapsim.preprocessor.models import *


def get_vehicle(vehicle_path):
    vehicle = None

    if vehicle_path.exists():
        vehicle_params = readers.read_csv(vehicle_path)
        vehicle = {}
        for key, value in zip(vehicle_params['param'], vehicle_params['value']):
            vehicle[key] = value
    else:
        log.error(f"vehicle.csv not found in {vehicle_path}")

    return vehicle


def from_cli(
    src: str,
    dest: str,
    spacing: int | None = None,
    flip: bool = False
):
    """This function is to be called using params entered via the CLI

    src: /path/path2/dataset-{00-23}.tar
    dest: /path/path2/dataset-%02d.tar

    Args:
        src: The source directory to scan subdirs for tracks from
        dest: The output directory to write sliced tracks on
        spacing: The gap between segmentation lines
    """
    flip_options = [False, True] if flip else [False]
    spacing = spacing or 10

    source_dataset = webdataset.WebDataset(src)

    # Scan all files to build the splicer inputs
    with webdataset.ShardWriter(dest, maxsize=500_000_000) as writer:
        for i, item in enumerate(source_dataset):
            item_id = item['id'].decode()
            print(f"\r{i + 1} {item_id} " + " " * 20, end="")

            params = SplicerInput(
                track=Track(segmentations=[])
            )

            # Create vehicle params
            vehicle = item["vehicle"].decode().split("\n")[1:]
            vehicle = [x.split(",") for x in vehicle if "," in x]
            vehicle = {datum[0]: float(datum[1]) for datum in vehicle}

            # Extract track from the record
            params.track = toolkit.tracks.conversion.from_xyrl(item['track'].decode())
            params.track = toolkit.tracks.smoother.smooth_track(params.track, spacing=spacing)

            # Load optimal path data if exists
            optimal_path = item["optimal_path"].decode()
            optimal_path = readers.read_csv_reader(optimal_path, delimiter=";")
            params.path = [
                PathInput(
                    x=optimal_path['x_m'][i],
                    y=optimal_path['y_m'][i],
                    vel=optimal_path['vx_mps'][i],
                    acc=optimal_path['ax_mps2'][i]
                )
                for i in range(len(optimal_path['s_m']))
            ]

            # Extract out the positions, velocities and accelerations as they exist on the track
            positions, velocities, accelerations = splice(params)

            for perform_flip in flip_options:
                track = np.array([seg.arr() for seg in params.track.segmentations])

                if perform_flip:
                    for idx in range(len(positions)):
                        positions[idx] = 1 - positions[idx]

                        x1, y1, x2, y2 = track[idx]

                        track[idx] = [x2, -y2, x1, -y1]

                widths, angles, offsets = extract_features(track)

                writer.write(encode({
                    "__key__": item["__key__"] + ("-flipped" if perform_flip else ""),
                    "id": item_id + ("-flipped" if perform_flip else ""),
                    "track": track,
                    "vehicle": vehicle,
                    "pos": positions,
                    "vel": velocities,
                    "acc": accelerations,
                    "widths": widths,
                    "angles": angles,
                    "offsets": offsets,
                    "flipped": perform_flip
                }))

    print(f"\rSpliced items complete.")


if __name__ == "__main__":
    # from_cli(
    #     src="/Users/belle/Developer/MlLapSim/dataset/lapsim-train-{00..24}.tar",
    #     dest="/Users/belle/Developer/MlLapSim/dataset/spliced-again-10/lapsim-train-%02d.tar",
    #     flip=True
    # )
    from_cli(
        src="/Users/belle/Developer/MlLapSim/dataset/lapsim-validation-{00..00}.tar",
        dest="/Users/belle/Developer/MlLapSim/dataset/spliced-again-10/lapsim-validation-%02d.tar",
        flip=True
    )
    from_cli(
        src="/Users/belle/Developer/MlLapSim/dataset/lapsim-test-{00..00}.tar",
        dest="/Users/belle/Developer/MlLapSim/dataset/spliced-again-10/lapsim-test-%02d.tar",
        flip=True
    )
    from_cli(
        src="/Users/belle/Developer/MlLapSim/dataset/lapsim-real-{00..02}.tar",
        dest="/Users/belle/Developer/MlLapSim/dataset/spliced-again-10/lapsim-real-%02d.tar",
        flip=False
    )

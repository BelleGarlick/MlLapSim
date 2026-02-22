import json
import os
from pathlib import Path
from typing import List

import toolkit.tracks.conversion
from .models.splicer_input import PathInput
from .splicer import splice
from toolkit.tracks.models import Track
from toolkit.utils import readers
from toolkit.utils.logger import log
from .models import *


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
):
    """This function is to be called using params entered via the CLI

    Args:
        src: The source directory to scan subdirs for tracks from
        dest: The output directory to write sliced tracks on
        spacing: The gap between segmentation lines
    """
    spacing = spacing or 10

    if not os.path.exists(src):
        raise FileNotFoundError(f"Source directory {src} does not exist")

    if not os.path.exists(dest):
        raise FileNotFoundError(f"Destination directory {dest} does not exist")

    # Get all values
    src = Path(src)
    x: List[str] = sorted(os.listdir(src))
    x = [y for y in x if y[0] != "."]
    print(f"Found {len(x)} files folders to splice.")

    # Scan all files to build the splicer inputs
    for i, dir in enumerate(x):
        try:
            if not os.path.isdir(src / dir):
                continue

            print(f"\r{i + 1}/{len(x)} ({int(i / len(x) * 100)}%) {dir} " + " " * 20, end="")

            path = src / dir

            output_path = Path(dest) / (Path(dir).stem + ".json")
            if output_path.exists():
                continue

            params = SplicerInput(
                track=Track(segmentations=[])
            )

            vehicle = get_vehicle(path / 'vehicle.csv')

            # Load the track to pass as a params to the input
            if (path / 'track.csv').exists():
                with open(path / 'track.csv') as file:
                    params.track = toolkit.tracks.conversion.from_xyrl(file.read())
                    params.track = toolkit.tracks.smoother.smooth_track(params.track, spacing=spacing)
            else:
                log.error(f"track.csv not found in {src / dir}")
                continue

            # Load optimal path data if exists
            if (path / 'optimal_path.csv').exists():
                optimal_path = readers.read_csv(path / 'optimal_path.csv', delimiter=";")
                params.path = [
                    PathInput(
                        x=optimal_path['x_m'][i],
                        y=optimal_path['y_m'][i],
                        vel=optimal_path['vx_mps'][i],
                        acc=optimal_path['ax_mps2'][i]
                    )
                    for i in range(len(optimal_path['s_m']))
                ]

            else:
                log.error(f"optimal_path.csv not found in {src / dir}")
                continue

            spliced_track = splice(params)

            with open(output_path, "w+") as file:
                file.write(json.dumps({
                    "track": spliced_track.model_dump(by_alias=True),
                    "vehicle": vehicle
                }, indent=True))

        except Exception as e:
            log.error(e)

    print(f"\r{len(x)} spliced items complete.")

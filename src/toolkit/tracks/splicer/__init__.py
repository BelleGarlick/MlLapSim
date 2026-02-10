import json
import os
import math
import random
from pathlib import Path
from typing import Optional, List

import toolkit.tracks.conversion
from .models.splicer_input import PathInput
from .splicer import splice
from .parallel_splicer import parallel_splice
from toolkit.tracks.models import Track
from toolkit.utils import readers
from toolkit.utils.logger import log
from .models import *


def on_complete(track: Track, output_path: Path, vehicle_path: Path):
    """On complete call back, called when track is spliced"""
    output_data = {
        "track": track.model_dump(by_alias=True),
        "vehicle": None
    }

    if vehicle_path.exists():
        vehice_params = readers.read_csv(vehicle_path)
        vehicle = {}
        for key, value in zip(vehice_params['param'], vehice_params['value']):
            vehicle[key] = value
        output_data['vehicle'] = vehicle

    else:
        log.error(f"vehicle.csv not found in {vehicle_path}")

    with open(output_path, "w+") as file:
        file.write(json.dumps(output_data, indent=True))


def from_cli(
        src: str,
        dest: str,
        spacing: Optional[int],
        batch_size: int,
        cores: int,
        portion: float = 1,
        seed: Optional[int] = None,
        precision: Optional[int] = None
):
    """This function is to be called using params entered via the CLI

    Args:
        src: The source directory to scan subdirs for tracks from
        dest: The output directory to write sliced tracks on
        spacing: The gap between segmentation lines
        batch_size: The size passed through to the batch_size param
        cores: Number of cores to spread the load across
        portion: Percentage of tracks to splice
        seed: The value to seed the randomness
        precision: The number of decimal places to reduce the output down to.
            if none, no values will be truncated
    """
    # Populate empty args
    if spacing is None: spacing = 5
    if not portion: portion = 1.0
    portion = min(1.0, max(0.0, portion))

    # Get all values
    src = Path(src)
    x: List[str] = sorted(os.listdir(src))
    print(f"Found {len(x)} files folders to splice.")

    # Reduce the number of items based on a random sample
    random.seed(seed)
    x = random.sample(x, k=math.ceil(len(x) * portion))
    random.seed(None)
    print(f"Reduced to {len(x)} items due to --portion {portion}")

    # Scan all files to build the splicer inputs
    print("Started scanning")
    inputs = []
    for i, dir in enumerate(x):
        print(f"\r{i + 1}/{len(x)} {dir}", end="")
        path = src / dir

        params = SplicerInput(
            track=Track(segmentations=[]),
            precision=precision,
            on_complete_args=[
                Path(dest) / f"{dir}.json",
                path / 'vehicle.csv'
            ],
            on_complete=on_complete
        )

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

        inputs.append(params)

    print("\rScan complete.")

    # Execute the parallel splicing
    parallel_splice(inputs, batch_size=batch_size, cores=cores)

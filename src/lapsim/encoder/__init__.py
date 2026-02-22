import json
import math
import os
import random
from pathlib import Path
from typing import Optional, List, Tuple

from lapsim.encoder.encoder import encode
from toolkit.tracks.models import Track
from lapsim.encoder.encoder_input import EncoderInput
from lapsim.encoder.partition import Partition

"""This whole module handles encoding the data according to Garlick e Bradley 
(2021). This file specifically has the functional interface from the CLI. This
is called from lapsim.__init__.py
"""


def from_cli(
        src: str,
        dest: str,
        n_partitions: Optional[int] = None,
        flip: Optional[bool] = False
):
    """This function is to be called using params entered via the CLI

    Args:
        src: The source directory to scan subdirs for tracks from
        dest: The output directory to write spliced tracks on
        n_partitions: Number of partitions to export, defaults to 0 (meaning
            everything is exported individually).
        flip: If flip then items will be repeated but the repeated item is
            flipped.
    """
    if not os.path.exists(src):
        raise FileNotFoundError(f"Source directory {src} does not exist")

    if not os.path.exists(dest):
        raise FileNotFoundError(f"Destination directory {dest} does not exist")

    # Populate empty args
    if n_partitions is None: n_partitions = 0

    # Get all values
    src, dest = Path(src), Path(dest)

    if not src: raise FileNotFoundError("Source directory not found")
    if not dest: raise FileNotFoundError("Destination directory not found")

    # Check whether to compute individually
    if n_partitions < 1:
        encode_singular_tracks(src, dest, flip)

    else:
        encode_multiple_tracks(src, dest, flip, n_partitions)


def get_track_paths(src, flip=False) -> List[Tuple[Path, bool]]:
    files: List[str] = [x for x in os.listdir(src) if x[0] != '.']
    files = [x for x in files if x[0] != "."]

    file_flip_paths = []
    for file in files:
        input_path = src / file
        file_name = input_path.stem

        file_flip_paths.append((input_path, file_name, False))

        if flip:
            file_name += ".flipped"
            file_flip_paths.append((input_path, file_name, True))

    random.shuffle(file_flip_paths)

    return file_flip_paths


def encode_singular_tracks(
    src,
    dest,
    flip
):
    files = get_track_paths(src, flip)
    for i, (path, file_name, flip) in enumerate(files):
        print(f"\r{i + 1}/{len(files)} {file_name}" + " " * 20, end="")
        with open(path) as file_data:
            data = json.load(file_data)

        encode(EncoderInput(
            track=Track.model_validate(data['track']),
            vehicle=data['vehicle'],
            flip=flip,
        )).save(str(dest / file_name) + ".json")


def encode_multiple_tracks(
    src,
    dest,
    flip,
    n_partitions
):
        files = get_track_paths(src, flip)
        partition_size = math.ceil(len(files) / n_partitions)

        current_partition, count = Partition(), 0
        for i, (path, file_name, flip) in enumerate(files):
            if len(current_partition.positions) >= partition_size:
                current_partition.save(str(dest / f"p{count}.json"))
                current_partition = Partition()
                count += 1

            print(f"\r{i + 1}/{len(files)} {file_name}" + " " * 20, end="")
            with open(path) as file_data:
                data = json.load(file_data)

            current_partition.append(
                encode(EncoderInput(
                    track=Track.model_validate(data['track']),
                    vehicle=data['vehicle'],
                    flip=flip,
                ))
            )

        if len(current_partition.angles) > 0:
            current_partition.save(str(dest / f"p{count}.json"))

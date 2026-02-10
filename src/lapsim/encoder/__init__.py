import json
import os
import math
import random
from pathlib import Path
from typing import Optional, List

from .encoder import encode
from .parallel_encoder import parallel_encode
from lapsim.encoder.encoder_input import EncoderInput
from toolkit.tracks.models import Track

"""This whole module handles encoding the data according to Garlick e Bradley 
(2021). This file specifically has the functional interface from the CLI. This
is called from lapsim.__init__.py
"""


def from_cli(
        src: str,
        dest: str,
        n_partitions: Optional[int],
        flip: Optional[bool],
        batch_size: int,
        cores: int,
        portion: float = 1,
        seed: Optional[int] = None
):
    """This function is to be called using params entered via the CLI

    Args:
        src: The source directory to scan subdirs for tracks from
        dest: The output directory to write spliced tracks on
        n_partitions: Number of partitions to export, defaults to 0 (meaning
            everything is exported individually).
        flip: If flip then items will be repeated but the repeated item is
            flipped.
        batch_size: The size passed through to the batch_size param
        cores: Number of cores to spread the load across
        portion: Percentage of tracks to splice
        seed: The value to seed the randomness
    """
    # Populate empty args
    if n_partitions is None: n_partitions = 0
    if not portion: portion = 1.0
    portion = min(1.0, max(0.0, portion))

    # Get all values
    src, dest = Path(src), Path(dest)
    files: List[str] = [x for x in os.listdir(src) if x[0] != '.']
    print(f"Found {len(files)} files folders to encode.")

    # Reduce the number of items based on a random sample
    random.seed(seed)
    files = random.sample(files, k=math.ceil(len(files) * portion))
    random.seed(None)
    print(f"Reduced to {len(files)} items due to --portion {portion}")

    # Scan all files to build the splicer inputs
    print("Started scanning")
    inputs: List[EncoderInput] = []
    file_names = []
    for i, file in enumerate(files):
        if file[0] == ".": continue
        print(f"\r{i + 1}/{len(files)} {dir}", end="")

        with open(src / file) as file_data:
            data = json.load(file_data)

            file_names.append(Path(file).stem)
            inputs.append(EncoderInput(
                track=Track.model_validate(data['track']),
                vehicle=data['vehicle']
            ))

            if flip:
                file_names.append(f"{Path(file).stem}-flipped")
                inputs.append(
                    EncoderInput(
                        track=Track.model_validate(data['track']),
                        vehicle=data['vehicle'],
                        flip=True
                    )
                )

    print("\rScan complete.")

    # Execute the parallel encoder
    parallel_encode(
        inputs,
        n_partitions=n_partitions,
        batch_size=batch_size,
        cores=cores,
        path=(
            (lambda idx: dest / f"{file_names[idx]}.json")
            if n_partitions < 1 else
            (lambda idx: dest / f"partition-{idx}.json")
        ),
        return_partitions=False
    )

    # TODO Add complete here and in spliced

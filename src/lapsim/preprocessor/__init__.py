import numpy as np
import webdataset
from functools import partial
from multiprocessing import Pool, cpu_count
import logging

import toolkit.tracks.conversion

from lapsim.preprocessor.encoder import encode
from lapsim.preprocessor.features import extract_features
from lapsim.preprocessor.models.splicer_input import PathInput
from lapsim.preprocessor.splicer import splice
from toolkit.tracks.models import Track
from toolkit.utils import readers
from lapsim.preprocessor.models import *


def process_record(
        item: dict,
        spacing,
        flip: bool
):
    outputs = []
    item_id = item['id'].decode()

    params = SplicerInput(
        track=Track(segmentations=[])
    )

    # Create vehicle params
    vehicle = item["vehicle"].decode().split("\n")[1:]
    vehicle = [x.split(",") for x in vehicle if "," in x]
    vehicle = {datum[0]: float(datum[1]) for datum in vehicle}

    # Extract track from the record
    # todo check the smoothing cos it's still failing
    params.track = toolkit.tracks.conversion.from_xyrl(item['track'].decode())
    params.track = toolkit.tracks.smoother.smooth_track(params.track, spacing=spacing, iter=100)

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
    try:
        positions, velocities, accelerations = splice(params)
    except Exception as e:
        logging.exception("Gahhh this needs to be sorted")
        return []

    track = np.array([seg.arr() for seg in params.track.segmentations])

    for perform_flip in [False, True] if flip else [False]:
        if perform_flip:
            for idx in range(len(positions)):
                positions[idx] = 1 - positions[idx]
                x1, y1, x2, y2 = track[idx]
                track[idx] = [x2, -y2, x1, -y1]

        widths, angles, offsets = extract_features(track)

        outputs.append(
            encode({
                "__key__": item["__key__"] + ("-flipped" if flip else ""),
                "id": item_id + ("-flipped" if flip else ""),
                "track": track,
                "vehicle": vehicle,
                "pos": positions,
                "vel": velocities,
                "acc": accelerations,
                "widths": widths,
                "angles": angles,
                "offsets": offsets,
                "flipped": perform_flip
            })
        )

    return outputs


def from_cli(
    src: str,
    dest: str,
    spacing: int | None = None,
    flip: bool = False,
    workers: int | None = None
):
    """This function is to be called using params entered via the CLI

    src: /path/path2/dataset-{00-23}.tar
    dest: /path/path2/dataset-%02d.tar

    Args:
        src: The source directory to scan subdirs for tracks from
        dest: The output directory to write sliced tracks on
        spacing: The gap between segmentation lines
        flip: Whether to flip the track and optimal path
        workers: The number of worker processes to use
    """
    spacing = spacing or 10
    workers = workers or cpu_count()

    source_dataset = webdataset.WebDataset(src)

    # Scan all files to build the splicer inputs
    with webdataset.ShardWriter(dest, maxsize=500_000_000) as writer:
        with Pool(workers) as pool:
            func = partial(process_record, spacing=spacing, flip=flip)
            for i, outputs in enumerate(pool.imap(func, source_dataset)):
                if not outputs:
                    continue
                
                # Use the first output to get the ID for progress reporting
                item_id = outputs[0]['id']
                print(f"\r{i + 1} {item_id} " + " " * 20, end="")

                for output in outputs:
                    writer.write(output)

    print(f"\rSpliced items complete.")


if __name__ == "__main__":
    # from_cli(
    #     src="/Users/belle/Developer/MlLapSim/dataset/raw/lapsim-train-{00..16}.tar",
    #     dest="/Users/belle/Developer/MlLapSim/dataset/processed/lapsim-train-%02d.tar",
    #     flip=True
    # )
    from_cli(
        src="/Users/belle/Developer/MlLapSim/dataset/raw/lapsim-validation-{0..1}.tar",
        dest="/Users/belle/Developer/MlLapSim/dataset/processed/lapsim-validation-%01d.tar",
        flip=True
    )
    from_cli(
        src="/Users/belle/Developer/MlLapSim/dataset/raw/lapsim-test-{0..1}.tar",
        dest="/Users/belle/Developer/MlLapSim/dataset/processed/lapsim-test-%01d.tar",
        flip=True
    )
    # from_cli(
    #     src="/Users/belle/Developer/MlLapSim/dataset/lapsim-real-{00..02}.tar",
    #     dest="/Users/belle/Developer/MlLapSim/dataset/spliced-again-10/lapsim-real-%01d.tar",
    #     flip=False
    # )

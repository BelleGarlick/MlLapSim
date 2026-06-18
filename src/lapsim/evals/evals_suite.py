import json
from pathlib import Path

import lapsim
from lapsim.models.lapsim import LapSimModel
from toolkit.tracks.models import Track
from lapsim.evals.evaluate import evaluate
import webdataset

evaluations = []

def run_evals_suite(
    model: LapSimModel,
    dataset_path: Path
):
    source_dataset = webdataset.WebDataset(dataset_path)
    for record in source_dataset:
        print(record)
        print(f"\r{i} - {partition_name}" + " " * 20, end="")

        pred_pos, pred_vel = model.predict(partition)

        # Copy the track and set the predictions from the model
        track_copy = Track(**original_track.model_dump())
        for i in range(len(original_track.segmentations)):
            original_track.segmentations[i].pos = pred_pos[i]
            original_track.segmentations[i].vel = pred_vel[i]

        # Evaluate the model
        evaluation = evaluate(original_track, track_copy)
        evaluations.append((partition_name, evaluation, original_track, track_copy))

    return evaluations

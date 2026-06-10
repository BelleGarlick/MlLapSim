import json
import os
from pathlib import Path

from lapsim.encoder import EncoderInput
import lapsim
from lapsim.models.lapsim import LapSimModel
from toolkit.tracks.models import Track
from lapsim.evals.evaluate import evaluate

evaluations = []

def run_evals_suite(
    model: LapSimModel,
    test_data_path: Path,
    spliced_data_path: Path,
    flip=False
):
    for i, partition_name in enumerate([x for x in os.listdir(test_data_path) if x[0] != "."]):
        for flip in ([True, False] if flip else [False]):
            print(f"\r{i} - {partition_name}" + " " * 20, end="")

            # Load up the segmentation lines from the spliced track outputs
            spliced_path = spliced_data_path / partition_name
            with open(spliced_path, "r") as f:
                data = json.load(f)
                vehicle = data['vehicle']
                track = data['track']
                original_track = Track(**track)

            if flip:
                for line in original_track.segmentations:
                    line.y1, line.y2 = -line.y1, -line.y2

                    line.y1, line.y2 = line.y2, line.y1
                    line.x1, line.x2 = line.x2, line.x1

                    line.pos = 1 - line.pos

            partition = lapsim.encoder.encode(
                EncoderInput(
                    track=original_track,
                    vehicle=vehicle,
                    flip=False
                )
            )

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

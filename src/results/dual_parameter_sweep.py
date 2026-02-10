import dataclasses
import math
import os
from typing import Dict, List

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from neural_lapsim.ml import models
from neural_lapsim.preprocessing.encoding import Partition
from neural_lapsim.preprocessing.encoding.utils import input_shape
from neural_lapsim.preprocessing.normaliser import NormalisationBounds
from neural_lapsim.preprocessing import normaliser
from results.tools.evaluate import compare_normals

# matplotlib.use('Qt5Agg')


PATH = r"E:\encoded\parameter sweep\KBrake_KDrive"
X_AXIS_LABEL = "KBrake"
Y_AXIS_LABEL = "KDrive"


@dataclasses.dataclass
class OptimisationData:
    lap_times: List[float]
    pred_lap_times: List[float]
    param_1s: List[float]
    param_2s: List[float]
    param_1: float
    param_2: float
    pred_param_1: float
    pred_param_2: float
    lap_delta: float

    @staticmethod
    def new():
        return OptimisationData([], [], [], [], 0, 0, 0, 0, 0)


class ParameterSweepItem:
    def __init__(self, file_name):
        self.file = file_name
        self.path = os.path.join(PATH, file_name)

        tokens = file_name.split(" - ")
        self.vehicle_track_pair = " - ".join(tokens[:3])
        self.param_1 = float(tokens[3])
        self.param_2 = float(tokens[4]) if len(tokens) > 4 else -1


model = models.create_dense_model(input_shape(foresight=120))
model.load_weights('../../data/models/dense_model.h5')
bounds = NormalisationBounds().load("../../data/bounds.json")


def generate_predicted_normals(x, normals):
    # Predict data
    predicted_pos, predicted_vel = model.predict(x)
    predicted_vel = (predicted_vel * (bounds.max_velocity - bounds.min_velocity)) + bounds.min_velocity

    # Combine the sampling
    for s in range(9):
        predicted_pos[:, s] = np.roll(predicted_pos[:, s], s - 4)
        predicted_vel[:, s] = np.roll(predicted_vel[:, s], s - 4)
    predicted_pos = np.mean(predicted_pos, axis=1)
    predicted_vel = np.mean(predicted_vel, axis=1)

    # Copy the original normals to splice the prediction into.
    normals_copy = normals.copy()
    normals_copy[:, 4] = predicted_pos
    normals_copy[:, 5] = predicted_vel

    return normals_copy


def predicted_laptimes_and_errors_for_files(sweep_items: ParameterSweepItem):
    minimum_predicted = math.inf, 0.5, 0.5
    minimum_real = math.inf, 0.5, 0.5
    lap_time_map: Dict[float, Dict[float, float]] = {}

    results = OptimisationData.new()

    for i, sweep_item in enumerate(sweep_items):
        print(f"\r{i}/{len(sweep_items)}", end="")
        partition = Partition.load(os.path.join(PATH, sweep_item.file))
        normalised, _ = normaliser.normalise_partition(partition, bounds)

        results.param_1s.append(sweep_item.param_1)
        results.param_2s.append(sweep_item.param_2)

        predicted_normals = generate_predicted_normals(normalised, partition.normals)

        # Compare the copied track normals from the actual normals
        comparison = compare_normals(predicted_normals, partition.normals)

        if comparison['predicted-lap-time'] < minimum_predicted[0]:
            minimum_predicted = comparison['predicted-lap-time'], sweep_item.param_1, sweep_item.param_2
        if comparison['lap-time'] < minimum_real[0]:
            minimum_real = comparison['lap-time'], sweep_item.param_1, sweep_item.param_2

        if sweep_item.param_1 not in lap_time_map:
            lap_time_map[sweep_item.param_1] = {}
        lap_time_map[sweep_item.param_1][sweep_item.param_2] = comparison['lap-time']

        results.lap_times.append(comparison['lap-time'])
        results.pred_lap_times.append(comparison['predicted-lap-time'])

    results.param_1 = minimum_real[1]
    results.param_2 = minimum_real[2]
    results.pred_param_1 = minimum_predicted[1]
    results.pred_param_2 = minimum_predicted[2]
    results.lap_delta = round(abs(lap_time_map[minimum_real[1]][minimum_real[2]] - lap_time_map[minimum_predicted[1]][minimum_predicted[2]]), 3)

    return results


def create_range(start, end, steps):
    step_size = (end - start) / (steps - 1)
    steps = []

    c = start
    while c <= end + (step_size / 2):
        steps.append(c)
        c += step_size

    return steps


if __name__ == "__main__":
    files = [ParameterSweepItem(x) for x in os.listdir(PATH)]
    track_value_map = {}
    for file in files:
        if file.vehicle_track_pair not in track_value_map:
            track_value_map[file.vehicle_track_pair] = []
        track_value_map[file.vehicle_track_pair].append(file)

    for track_vehicle in track_value_map:
        files = sorted(track_value_map[track_vehicle], key=lambda x: x.param_1)

        result = predicted_laptimes_and_errors_for_files(files)
        print(f"\r{track_vehicle}: {result.param_1} {result.pred_param_1} {result.param_2} {result.pred_param_2} {result.lap_delta}")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(
            np.array(result.param_1s),
            np.array(result.param_2s),
            np.array(result.lap_times),
            label="True lap time"
        )
        ax.plot_trisurf(
            np.array(result.param_1s),
            np.array(result.param_2s),
            np.array(result.pred_lap_times),
            label="Predicted lap time"
        )
        ax.set_xlabel(X_AXIS_LABEL)
        ax.set_ylabel(Y_AXIS_LABEL)
        ax.set_zlabel("Lap Time")

        plt.title(track_vehicle)
        plt.show()

import math
import os
from typing import Dict, List

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import neural_lapsim
from neural_lapsim.preprocessing.encoding.utils import input_shape
from neural_lapsim.preprocessing.normaliser import NormalisationBounds
from results.tools.evaluate import compare_normals

# matplotlib.use('Qt5Agg')


REGULAR_FRICTION = r"E:\encoded\parameter sweep\KBrake"
MID_FRICTION = r"E:\encoded\parameter sweep\KBrake-mid-friction"
LOW_FRICTION = r"E:\encoded\parameter sweep\KBrake-low-friction"

TRACKS = [
    "UK Silverstone - GP Circuit",
    "BR Interlagos - GP Circuit",
    "AT Red Bull Ring - GP Circuit"
]

VEHICLES = [
    "Generic LMP2",
    "Generic GT3",
    "Generic 1982 Ground Effect F1"
]


model = neural_lapsim.ml.models.dense.create_model(input_shape(foresight=120))
model.load_weights('../../data/models/dense_model.h5')
bounds = NormalisationBounds().load("../../data/bounds.json")


class ParameterSweepItem:
    def __init__(self, path, file_name):
        self.file = file_name
        self.path = os.path.join(path, file_name)

        tokens = file_name.split(" - ")
        self.vehicle_track_pair = " - ".join(tokens[:3])
        self.param_1 = float(tokens[3])


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
    param1s = []
    true_lap_times = []
    predicted_lap_times = []

    minimum_predicted = math.inf, 0.5
    minimum_real = math.inf, 0.5
    lap_time_map: Dict[float, Dict[float, float]] = {}

    for i, sweep_item in enumerate(sweep_items):
        print(f"\r{i}/{len(sweep_items)}", end="")
        partition = neural_lapsim.preprocessing.encoding.Partition.load(sweep_item.path)
        normalised, _ = neural_lapsim.preprocessing.normaliser.normalise_partition(partition, bounds)

        param1s.append(sweep_item.param_1)
        predicted_normals = generate_predicted_normals(normalised, partition.normals)

        # Compare the copied track normals from the actual normals
        comparison = compare_normals(partition.normals, predicted_normals)

        if comparison['predicted-lap-time'] < minimum_predicted[0]:
            minimum_predicted = comparison['predicted-lap-time'], sweep_item.param_1
        if comparison['lap-time'] < minimum_real[0]:
            minimum_real = comparison['lap-time'], sweep_item.param_1

        if sweep_item.param_1 not in lap_time_map:
            lap_time_map[sweep_item.param_1] = {}
        lap_time_map[sweep_item.param_1] = comparison['lap-time']

        true_lap_times.append(comparison['lap-time'])
        predicted_lap_times.append(comparison['predicted-lap-time'])

    return {
        "lap-times": true_lap_times,
        "pred-lap-times": predicted_lap_times,
        "params": param1s,
        "param": minimum_real[1],
        "pred-param": minimum_predicted[1],
        "lap-delta": round(abs(lap_time_map[minimum_real[1]] - lap_time_map[minimum_predicted[1]]), 3)
    }


def get_files_for(vehicle, track) -> Dict[str, List[ParameterSweepItem]]:
    def filter_files(path):
        return [
            ParameterSweepItem(path, x)
            for x in
            filter(lambda x: f"{track} - {vehicle}" in x, os.listdir(path))
        ]

    return {
        "regular": filter_files(REGULAR_FRICTION),
        "medium": filter_files(MID_FRICTION),
        "low": filter_files(LOW_FRICTION)
    }


if __name__ == "__main__":
    for vehicle in VEHICLES:
        for track in TRACKS:
            filtered_files = get_files_for(vehicle, track)
            results = {}

            for key in filtered_files:
                files = sorted(filtered_files[key], key=lambda x: x.param_1)
                result = predicted_laptimes_and_errors_for_files(files)
                results[key] = result
                print(f"\r{track} {vehicle} {key}: {result['param']} {result['pred-param']} {result['lap-delta']} {min(result['lap-times'])} {min(result['pred-lap-times'])}")

                optim_laptime_index, pred_optim_laptime_index = np.argmin(result['lap-times']), np.argmin(result['pred-lap-times'])
                optimal_param = result['params'][optim_laptime_index]
                pred_optimal_param = result['params'][pred_optim_laptime_index]
                optim_laptime = result['lap-times'][optim_laptime_index]
                pred_optim_laptime = result['lap-times'][pred_optim_laptime_index]

                max_lin = max(result["lap-times"] + result["pred-lap-times"])
                min_lim = min(result["lap-times"] + result["pred-lap-times"])
                spread = max_lin - min_lim
                plt.ylim(min_lim - spread, max_lin + spread)

                plt.axvline(x=result['param'], color='green', linewidth=0.5)
                plt.axvline(x=result['pred-param'], color='red', linestyle='--', linewidth=0.5)
                plt.axhline(y=optim_laptime, color='green', linewidth=0.5)
                plt.axhline(y=pred_optim_laptime, color='red', linestyle='--', linewidth=0.5)

                plt.plot([x.param_1 for x in files], result["lap-times"], label="True lap time", color='green')
                plt.plot([x.param_1 for x in files], result["pred-lap-times"], label="Predicted lap time", color='red', linestyle='--')

                plt.ylabel("Lap Time")
                plt.xlabel("KBrake")

                plt.legend()

                title = f"{track} {vehicle} {key}"
                plt.title(title)
                plt.savefig(fr"C:\Users\SamGa\Documents\GitHub\ML-LapSim\data\results\images\parameter_sweep\{title}")
                # plt.show()

                plt.cla()


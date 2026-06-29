import random

import numpy as np

from lapsim.normalisation.transforms.sampling import get_target_output
from lapsim.normalisation.transforms.window.base import BaseWindowTransform


"""This module encodes data into windows as described in garlick and bradley 
2021, whereby the network trains on a series of windows to predict the vehicle
position and velocity within the center of that window.

This module transforms the data into a single vector allowing a dense NN to 
train upon."""


class FlatWindowTransform(BaseWindowTransform):

    def transform(self, record):
        window_length = self.foresight * 2 + 1

        track_length = len(record["widths"])

        # Select the indexes that will be sampled
        indexes = list(range(track_length))
        if self.single_sample:
            indexes = [random.randint(0, track_length - 1)]

        # Create the input vector
        x = np.zeros((len(indexes), window_length * len(self.inputs)))
        for out_index, index in enumerate(indexes):
            window_data = []
            for input_key in self.inputs:
                window_modality = np.zeros(self.foresight * 2 + 1)
                window_modality_input = record[input_key]
                for i, f in enumerate(range(index - self.foresight, index + self.foresight + 1)):
                    window_modality[i] = window_modality_input[f % track_length]

                window_data.append(window_modality)
            x[out_index] = np.concatenate(window_data)

        # Create the vehicles vector
        vehicles = np.array([record["vehicle"]] * len(indexes))

        # Create the targets vector
        outputs = [
            np.zeros((len(indexes), self.sampling * 2 + 1)),
            np.zeros((len(indexes), self.sampling * 2 + 1)),
        ]
        for idx, output_key in enumerate(["pos", "vel"]):
            for out_index, index in enumerate(indexes):
                window_modality = np.zeros(self.sampling * 2 + 1)
                window_modality_input = record[output_key]
                for i, s in enumerate(range(index - self.sampling, index + self.sampling + 1)):
                    window_modality[i] = window_modality_input[s % track_length]

                outputs[idx][out_index] = window_modality

        return x, vehicles, outputs[0], outputs[1]


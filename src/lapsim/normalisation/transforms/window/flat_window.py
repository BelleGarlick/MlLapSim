import numpy as np

from lapsim.normalisation.normalised_data import NormalisedData
from lapsim.normalisation.transforms.common import TransformMethod
from lapsim.normalisation.transforms.sampling import get_target_output
from lapsim.normalisation.transforms.window.base import BaseWindowTransform


"""This module encodes data into windows as described in garlick and bradley 
2021, whereby the network trains on a series of windows to predict the vehicle
position and velocity within the center of that window.

This module transforms the data into a single vector allowing a dense NN to 
train upon."""


class FlatWindowTransform(BaseWindowTransform):

    def transform(self, normalised: NormalisedData, cores: int):
        """Encode the data into a series of windows (as described in Garlick &
        Bradley 2021, but compress each window into a single vector containing
        widths, angles, offsets and vehicles.

        Args:
            normalised: The normalised partition from the normalisation step
            cores: Number of cores used to multiprocess the track using

        Returns:
            (x, vehicles), (y_pos, y_vel)
        """
        total_normals_count = sum([len(x) for x in normalised.angles])

        window_length = self.foresight * 2 + 1
        total_window_size = len(self.inputs) * window_length

        # Preallocate the memory, this makes it much faster and memory efficient as
        # the arrays don't need reallocating
        x = np.zeros((total_normals_count, total_window_size), dtype=np.float32)
        vehicles = np.zeros((total_normals_count, len(normalised.vehicles[0])), dtype=np.float32)

        # Encode tracks
        track_encodings = self.perform_parallel_transforms(_flat_window_transform, normalised, cores)

        global_index = 0
        for i in range(len(normalised)):
            track_length = normalised.track_length(i)

            x[global_index: global_index + track_length] = track_encodings[i]
            vehicles[global_index:global_index + track_length] = normalised["vehicles"][i]

            global_index += track_length

        return x, get_target_output(normalised, outputs=self.outputs, sampling=self.sampling), vehicles


def _flat_window_transform(normalised_data: NormalisedData, transform: TransformMethod, track_index: int):
    """The flat-window transform, called by `perform_parallel_transforms` to
     parellalize the transformation"""
    window_length = transform.foresight * 2 + 1

    track_length = normalised_data.track_length(track_index)
    x = np.zeros((track_length, window_length * len(transform.inputs)))

    # Extract inputs from the transform based on defined keys
    inputs = [normalised_data[_inp][track_index] for _inp in transform.inputs]

    for normal_index in range(track_length):
        for i, f in enumerate(range(normal_index - transform.foresight, normal_index + transform.foresight + 1)):
            index = f % track_length

            # Iterate through the inputs splicing in where needed
            for inp_idx in range(len(transform.inputs)):
                x[normal_index, i + (window_length * inp_idx)] = inputs[inp_idx][index]

    return x

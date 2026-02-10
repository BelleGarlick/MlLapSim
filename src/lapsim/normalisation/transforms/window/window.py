import numpy as np

from lapsim.normalisation.normalised_data import NormalisedData
from lapsim.normalisation.transforms.common import TransformMethod
from lapsim.normalisation.transforms.sampling import get_target_output
from lapsim.normalisation.transforms.window.base import BaseWindowTransform


"""This module encodes data into windows as described in garlick and bradley 
2021, whereby the network trains on a series of windows to predict the vehicle
position and velocity within the center of that window.

This method transforms the window into a 1d image with multiple channels 
allowing a CNN to train upon."""


class WindowTransform(BaseWindowTransform):

    def transform(self, normalised: NormalisedData, cores: int):
        """Encode the data into a series of windows (as described in Garlick &
        Bradley 2021, where each window is a 3 x (2f+1) matrix which can be trained
        with a CNN. Vehicles aren't included in the main input and need to be fed
        into the network in a sperate input.

        Args:
            normalised: The normalised partition from the normalisation step

        Returns:
            (x, vehicles), (y_pos, y_vel)
        """
        total_normals_count = sum([len(x) for x in normalised.angles])

        # Preallocate the memory, this makes it much faster and memory efficient as
        # the arrays don't need reallocating
        x = np.zeros((
            total_normals_count,
            len(self.inputs),
            self.foresight * 2 + 1
        ), dtype=np.float32)

        vehicles = np.ones((total_normals_count, len(normalised.vehicles[0])))

        # Encode tracks
        track_encodings = self.perform_parallel_transforms(_window_transform, normalised, cores)

        global_index = 0
        for i in range(len(normalised)):
            track_length = normalised.track_length(i)

            vehicles[global_index:global_index + track_length] = normalised["vehicles"][i]
            for ti, window in enumerate(track_encodings[i]):
                x[global_index + ti] = window

            global_index += track_length

        return x, get_target_output(normalised, outputs=self.outputs, sampling=self.sampling), vehicles


def _window_transform(normalised: NormalisedData, transform: TransformMethod, track_index: int):
    """The window transform, called by `perform_parallel_transforms` to
     parellalize the transformation"""
    track_length = len(normalised.widths[track_index])

    # Extract the inputs (eg widths, angles, offsets) for the current track
    inputs = [normalised[_inp][track_index] for _inp in transform.inputs]

    windows = []

    for normal_index in range(track_length):
        window = np.zeros((len(transform.inputs), 2 * transform.foresight + 1), dtype=np.float32)

        # Foresight
        for i, f in enumerate(range(normal_index - transform.foresight, normal_index + transform.foresight + 1)):
            index = f % track_length
            # Loop through the given inptus (eg widths, angles, offsets) and splice into the window
            for inp_idx, _inp in enumerate(transform.inputs):
                window[inp_idx][i] = inputs[inp_idx][index]

        windows.append(window)

    return windows

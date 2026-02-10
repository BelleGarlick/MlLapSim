import math
from typing import List

import numpy as np

from lapsim.normalisation.normalised_data import NormalisedData
from lapsim.normalisation.transforms.common import patchify, combine, TransformMethod
from lapsim.normalisation.transforms.sampling import loop_track_for_patching_sampling


"""This class transforms the data allowing for a stateful LSTM to predict from.
This method should not be used for training on.
"""


def apply_sampling(x, sampling=0):
    rolled_items = [np.roll(x, sampling, axis=0)]
    for i in range(sampling * 2):
        rolled_items.append(np.roll(rolled_items[-1], -1, axis=0))
    return np.concatenate(rolled_items, axis=1)


class StatefulLaggingTransformMethod(TransformMethod):

    def transform(self, normalised: NormalisedData, cores: int):
        """Stateful lagging history works akin to the normal lagging history in
        concept, however, instead of using a series of windows, this method just
        creates a single vector for the whole track which you would feed into the
        network. This method only works for stateful RNN/LSTMS but it allows for
        the prediction to run much faster since normals aren't repeated for each
        index in the window.

        Since this method requires feeding in the track to the network each line
        one-by-one, and started predicting immediently from the first line, then
        the NN would have no context of the previous corner. So instead the input
        sequence is repeated, meaning the track is fed through the network twice.
        This allows for the first segmentation line to have historical context.

        Args:
            normalised: The normalised track encoding.
            cores: Number of cores to spread transform across

        Returns:
            A tuple containing a list of the input track sequences and a list of
            the output track sequences
        """
        inputs = []
        outputs = []
        vehicles = []

        for track_idx in range(len(normalised)):
            track_length = normalised.track_length(track_idx)
            patched_track_size = math.ceil(track_length / self.patch_size)
            track_to_vec = np.arange(patched_track_size) / (max(1, patched_track_size - 1))

            track = combine(*[normalised[_inp][track_idx] for _inp in self.inputs])
            track = np.concatenate((track, track))
            track = patchify(track, self.patch_size)

            # Apply time to vec
            if self.time_to_vec:
                track = np.hstack((track, np.zeros((track.shape[0], 1))))
                track[:len(track_to_vec), -1] = track_to_vec
                track[-len(track_to_vec):, -1] = track_to_vec

            inputs.append(track)
            vehicles.append(np.array(normalised["vehicles"][track_idx]))

            # Loop and roll the tracks to accomodate patch_sizing/sampling larger than the track
            rolled_outputs = [np.roll(normalised[output][track_idx], self.lag) for output in self.outputs]
            patched_outputs = [
                loop_track_for_patching_sampling(output, sampling=self.sampling, patch_size=self.patch_size)
                for output in rolled_outputs]

            final_outputs = [
                np.zeros((patched_track_size, self.patch_size * (self.sampling * 2 + 1)))
                for _ in self.outputs
            ]
            for i, norm in enumerate(range(0, track_length, self.patch_size)):
                N, P, L = track_length, self.patch_size, (patched_track_size - i - 1)
                patch_start = (N - (P * (1 + L + self.sampling))) % N
                patch_end = patch_start + (self.patch_size * (self.sampling * 2 + 1))

                for out_idx in range(len(self.outputs)):
                    final_outputs[out_idx][i] = patched_outputs[out_idx][patch_start:patch_end]

            outputs.append(final_outputs)

        return inputs, outputs, vehicles

    def detransform(self, track_length: int, outputs: List[np.ndarray]) -> List[np.ndarray]:
        detransformed_outputs = []
        for output in outputs:
            detransformed = [[] for _ in range(track_length)]
            for index in range(len(output)):
                frame_output = output[index]
                for i in range(len(frame_output)):
                    # Calculate the original index to put the value in
                    N, P, L = track_length, self.patch_size, (len(output) - index - 1)
                    original_index = (N - (P * (1 + L + self.sampling)) + i) % N
                    detransformed[original_index].append(frame_output[i])

            detransformed = np.array([np.mean(x) for x in detransformed])
            detransformed = np.roll(detransformed, -self.lag)
            detransformed_outputs.append(detransformed)

        return detransformed_outputs

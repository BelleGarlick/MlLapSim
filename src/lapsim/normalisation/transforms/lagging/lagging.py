import math
from typing import List

import numpy as np

from lapsim.normalisation.normalised_data import NormalisedData
from lapsim.normalisation.transforms.common import patchify, combine, TransformMethod
from lapsim.normalisation.transforms.sampling import get_target_output


"""This method encodes the data allowing for a stateless LSTM to train and 
predict from.
"""


def apply_sampling(x, sampling=0):
    rolled_items = [np.roll(x, sampling, axis=0)]
    for i in range(sampling * 2):
        rolled_items.append(np.roll(rolled_items[-1], -1, axis=0))
    return np.concatenate(rolled_items, axis=1)


class LaggingTransformMethod(TransformMethod):

    def transform(self, normalised: NormalisedData, cores: int):
        """Lagging history transform creates a series track sequences where the
        last item in the sequence is the current point we're predicting the output
        of (assuming lag is 0). This method is designed to work in stateless RNN/
        LSTM networks hence the need for each seg. line having the entire track
        history included. This however, means it is far slower and less memory
        efficient than the normal window encoding and the stateful lag method.

        Each track is padded at the start with -1s at the start to keep everything
        regularised for training.

        Args:
            normalised: The normalised track encoding.
            cores: Number of cores to spread over

        Returns:
            A tuple containing a tuple of the input frames and input vehicles
            and the output tuple containing the y_pos and y_vel.
        """
        items_count = normalised.normals_count()
        max_track_length = normalised.longest_track_length()

        vector_length = math.ceil(max_track_length / self.patch_size) * 2
        x = np.zeros(
            (items_count, vector_length, len(self.inputs) * self.patch_size + int(self.time_to_vec)),
            dtype=np.float32)
        x.fill(-1)
        vehicles = np.zeros((items_count, len(normalised.vehicles[0])), dtype=np.float32)

        # Encode tracks
        track_encodings = self.perform_parallel_transforms(_lag_transform, normalised, cores)

        global_index = 0
        for i in range(len(normalised.vehicles)):
            track_length = len(normalised.widths[i])
            track_encoding = track_encodings[i]

            x[global_index:global_index + len(track_encoding), -track_encoding.shape[1]:] = track_encoding
            vehicles[global_index:global_index + track_length] = normalised.vehicles[i]

            global_index += track_length

        # Apply sampling patchification
        return (
            x,
            get_target_output(
                normalised,
                outputs=self.outputs,
                lag=self.lag,
                sampling=self.sampling,
                patch_size=self.patch_size
            ),
            vehicles
        )

    def detransform(self, track_length: int, outputs: List[np.ndarray]) -> List[np.ndarray]:
        detransformed_outputs = []
        for output in outputs:
            detransformed = [[] for _ in range(output.shape[0])]
            for index in range(len(output)):
                frame_output = output[index]
                for i in range(len(frame_output)):
                    original_index = (index + (i - self.patch_size * (self.sampling + 1) + 1)) % track_length
                    detransformed[original_index].append(frame_output[i])

            detransformed = np.array(np.mean(detransformed, axis=1))
            detransformed = np.roll(detransformed, -self.lag)
            detransformed_outputs.append(detransformed)

        return detransformed_outputs


def _lag_transform(normalised: NormalisedData, transform: TransformMethod, t_idx: int):
    """The lagging transform, called by `perform_parallel_transforms` to
     parellalize the transformation"""
    track_length = len(normalised.widths[t_idx])
    vector_length = math.ceil(track_length / transform.patch_size) * 2

    x = np.zeros((
        track_length,
        vector_length,
        len(transform.inputs) * transform.patch_size + int(transform.time_to_vec)
    ), dtype=np.float32)
    x.fill(-1)

    patched_track_size = math.ceil(track_length / transform.patch_size)

    track_to_vec = np.arange(patched_track_size) / (max(1, patched_track_size - 1))

    # Combine th defined inputs (e.g. position, angles, offsets)
    track = combine(*[normalised[_inp][t_idx] for _inp in transform.inputs])

    for normal_index in range(track_length):
        subtrack = np.concatenate((track, track[:normal_index + 1]))
        subtrack = patchify(subtrack, patch_size=transform.patch_size)

        if transform.time_to_vec:
            time_vec = np.zeros((len(subtrack), 1))
            time_vec[:len(track_to_vec), -1] = track_to_vec
            exp_patch = (normal_index // transform.patch_size) + 1
            time_vec[-exp_patch:, -1] = track_to_vec[:exp_patch]
            subtrack = np.hstack((subtrack, time_vec))

        x[normal_index, -len(subtrack):] = subtrack

    return x

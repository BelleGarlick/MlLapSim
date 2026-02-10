import math
from typing import List

import numpy as np

from lapsim.normalisation.normalised_data import NormalisedData
from lapsim.normalisation.transforms.common import patchify, combine, TransformMethod
from lapsim.normalisation.transforms.sampling import get_target_output


"""Bidirectional history transform allows for the full track to be passed into
the network such that it's both fed in forwards and in reverse. Since LSTMs and
RNN's have troubles remember long term information, we cannot rely on the 
network remember the whole track if fed in serially. 

The lagging history transform overcomes this by offseting the input window such 
that the output lags behind meaning it has a little window ahead of the current 
predition input line. This gives the network context for the upcoming corner 
but, too much lag may result in the network not remembering the current context
too well but too little lag may result in the network not able to correctly 
predict for the upcoming series of corners.

This method overcomes the lag tradeoff by passing in the whole upcoming track
and the whole historical track length for every segmentation line meaning it
has full context of past and future whilst keeping the current segmentation 
line as the most recent point in time.
"""


class BidirectionalTransformMethod(TransformMethod):

    def transform(self, normalised: NormalisedData, cores: int):
        """The bidirectional history transform. This works by iterating through the
        track length and inputing the history of the track up to this point in time
        and by passing in the future track doing the same but in reverse. This
        means you can a matrix with the following indexes:
         [t1, t2, ..., tn, t0]
         [tn, ..., t2, t1, t0]
        (for each data types (width, angles, offsets))

        Each track is padded with -1's at the start so that we have a regular
        shaped array. This helps speed up training since we can use stateful
        methods and regular batching.

        Args:
            normalised: The normalised track encoding.
            cores: Number of cores to spread the compute over

        Returns:
             A tuple of the inputs (a tuple of the input bidirectional array and
             vehicle array) and the outputs (sampled position and sampled velocity)
        """
        x = np.zeros((
            normalised.normals_count(),
            math.ceil(normalised.longest_track_length() / self.patch_size),
            2 * len(self.inputs) * self.patch_size
        ), dtype=np.float32)
        x.fill(-1)

        vehicles = np.zeros((normalised.normals_count(), normalised.vehicle_size()), dtype=np.float32)

        # Encode tracks
        track_encodings = self.perform_parallel_transforms(_bidirectional_transform, normalised, cores)

        global_index = 0
        for i in range(len(normalised.vehicles)):
            track_length = len(normalised.widths[i])
            track_encoding = track_encodings[i]

            x[global_index:global_index + len(track_encoding), -track_encoding.shape[1]:] = track_encoding
            vehicles[global_index:global_index + track_length] = normalised.vehicles[i]

            global_index += track_length

        return (
            x,
            get_target_output(
                normalised,
                outputs=self.outputs,
                sampling=self.sampling,
                patch_size=self.patch_size
            ),
            vehicles
        )

    # TODO Document
    def detransform(self, track_length: int, outputs: List[np.ndarray]):
        detransformed_outputs = []

        for output in outputs:
            detransformed = [[] for _ in range(output.shape[0])]
            for index in range(len(output)):
                frame_output = output[index]
                for i in range(len(frame_output)):
                    original_index = (index + (i - self.patch_size * (self.sampling + 1) + 1)) % track_length
                    detransformed[original_index].append(frame_output[i])

            detransformed_outputs.append(np.array([np.mean(x) for x in detransformed]))

        return detransformed_outputs


def _bidirectional_transform(normalised: NormalisedData, transform: TransformMethod, track_index: int):
    """The bidirectional transform, called by `perform_parallel_transforms` to
     parellalize the transformation"""
    track_length = len(normalised.widths[track_index])

    x = np.zeros((
        track_length,
        math.ceil(track_length / transform.patch_size),
        2 * len(transform.inputs) * transform.patch_size
    ), dtype=np.float32)
    x.fill(-1)

    track = combine(*[normalised[_inp][track_index] for _inp in transform.inputs])
    reversed_track = track[::-1]

    for normal_index in range(track_length):
        # Roll the track based on the normal index
        history = [np.roll(track, axis=0, shift=-normal_index - 1)]
        reversed_history = [np.roll(reversed_track, axis=0, shift=normal_index)]

        # Combine the history together
        window = np.concatenate(history + reversed_history, axis=1)
        window = patchify(window, patch_size=transform.patch_size)

        x[normal_index, -len(window):] = window

    return x

from typing import List

import numpy as np

from lapsim.normalisation.normalised_data import NormalisedData
from lapsim.normalisation.transforms.common import TransformMethod


"""This module stores the BaseWindow class which contains the detransformation."""


class BaseWindowTransform(TransformMethod):

    def transform(self, normalised: NormalisedData, cores: int):
        raise NotImplementedError

    def detransform(self, track_length: int, outputs: List[np.ndarray]) -> List[np.ndarray]:
        """This function detransforms the output sampling, combining it back to the
        original vector. This is useful for the combining the sampled output of the
        network.

        Args:
            track_length: Number of normals in the track
            outputs: The list of output data from the network. e.g. [y_pos, y_vel]

        Returns:
            The combined output, desampled output
        """
        detransformed_outputs = []
        for output in outputs:
            detransformed = [[] for _ in range(track_length)]
            for index in range(len(output)):
                for s in range(-self.sampling, self.sampling + 1):
                    detransformed[(index + s) % len(output)].append(output[index][s + self.sampling])

            detransformed_outputs.append(np.array([np.mean(x) for x in detransformed]))

        return detransformed_outputs

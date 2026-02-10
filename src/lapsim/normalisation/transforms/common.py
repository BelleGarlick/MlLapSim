import math
from abc import ABC
from multiprocessing import Pool
from typing import List, Callable, Tuple, Any

import numpy as np

from lapsim.normalisation.normalised_data import NormalisedData


def patchify(x, patch_size: int):
    """Patch up a matrix into sub patches. The inspiration for this comes from
    the vision transformers paper whereby an image is split into subpatches. We
    can do the same thing so that rather than a seg. line being passed to the
    network one-by-one, we can pass in a patch of seg. lines at once.

    This encoding has the following effect:
     - [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # Before
     - [[-1, -1, 0, 1, 2], [3, 4, 5, 6, 7], [8, 9, 10, 11, 12]]  # After
    ^ Before has been simplified, each item is a vector of that item (
    containing the width, angle and offset). This means the number of
    dimensions is preserved before and after this function.

    Args:
        x: The input matrix which is a list of encoded seg. lines.
        patch_size: The number of seg. lines to combine into one vector.

    Returns:
        The patched input.
    """
    if patch_size == 1:
        return x

    track_length = x.shape[0]
    start_padding = (patch_size - (track_length % patch_size)) % patch_size

    patched_x = np.zeros((math.ceil(track_length / patch_size), x.shape[1] * patch_size))

    for c, start in enumerate(range(-start_padding, track_length, patch_size)):
        end_index = start + patch_size

        patch = x[max(0, start):end_index].flatten()

        if c == 0:
            subset = np.zeros(x.shape[1] * patch_size)
            subset.fill(-1)
            subset[-patch.size:] = patch
            patch = subset

        patched_x[c] = patch

    return patched_x


def combine(*items):
    """Combine a list of vectors into a vector by stacking them."""
    track = np.zeros((len(items[0]), len(items)))
    for i, item in enumerate(items):
        track[:, i] = item

    return track


class TransformMethod(ABC):
    def __init__(self):
        self.inputs: List[str] = []
        self.outputs: List[str] = []
        self.sampling: int = 0

        self.foresight: int = 0
        self.patch_size: int = 1
        self.lag: int = 0
        self.time_to_vec: bool = False

    def transform(self, normalised: NormalisedData, cores: int):
        raise NotImplementedError

    def detransform(self, track_length: int, outputs: List[np.ndarray]) -> List[np.ndarray]:
        raise NotImplementedError

    def perform_parallel_transforms(
            self,
            function: Callable[[NormalisedData, 'TransformMethod', int], Any],
            normalised: NormalisedData,
            cores: int
    ):
        # Create inputs
        map_inputs = [(function, normalised, self, track_index) for track_index in range(len(normalised))]
        if cores == 1 or len(normalised) <= 1:
            for track_index in range(len(normalised.vehicles)):
                return list(map(parallel_wrapper, map_inputs))
        else:
            with Pool(cores) as p:
                return p.map(parallel_wrapper, map_inputs)


def parallel_wrapper(item: Tuple[Callable, NormalisedData, TransformMethod, int]):
    return item[0](item[1], item[2], item[3])

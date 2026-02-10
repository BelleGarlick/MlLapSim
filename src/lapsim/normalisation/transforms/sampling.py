import math
from typing import List

import numpy as np

from lapsim.normalisation.normalised_data import NormalisedData


"""This module handles sampling and desampling the outputs of the transformer &
networks."""


def get_target_output(
        normalised: NormalisedData,
        outputs: List[str],
        sampling: int = 0,
        lag: int = 0,
        patch_size: int = 1
):
    """This function gets the target output when normalising data. This is done
    by creating a multiple matrix described in garlick & bradley 2021

    Args:
        normalised: The normalised data class
        outputs: List of keys to encode in the given order, eg positions, velocities
        sampling: The output sampling
        lag: How much lag to apply to the output
        patch_size: The patch size

    Returns:
        A tuple of the output vectors
    """
    items_count = normalised.normals_count()

    outputs_vectors = [np.zeros((items_count, (sampling * 2 + 1) * patch_size)) for _ in outputs]

    global_item_index = 0
    for t_idx in range(len(normalised)):
        track_length = normalised.track_length(t_idx)

        # Compute output vectors by looping through output keys inserting them into the output vectors
        for i, output in enumerate(outputs):
            outputs_vectors[i][global_item_index:global_item_index + track_length] = compute_targets_for_track(
                np.array(normalised[output][t_idx]),
                lag=lag,
                sampling=sampling,
                patch_size=patch_size
            )

        global_item_index += track_length

    return outputs_vectors


def loop_track_for_patching_sampling(arr, sampling, patch_size):
    """Loop the track to encompass a large sampling/patch_size

    Returns:
        The looped track.
    """
    output_size = patch_size * (sampling * 2 + 1)
    repeats = math.ceil(output_size / len(arr)) + 1
    return np.concatenate([arr] * repeats)


def compute_targets_for_track(arr: np.ndarray, sampling=0, lag=0, patch_size=1):
    """Given the target vector and sampling, lag and patching, transform the
    input accordingly.

    Args:
        arr: The inputs to transform
        sampling: Number of seg lines to predict prior and
            post the central position
        lag: How much to offset the track so the output lags behind
        patch_size: Size of the patch to group lines by.

    Returns:
        The transformed output
    """
    output = np.zeros((len(arr), patch_size * (sampling * 2 + 1)))

    # Apply lag, then repeat the track so that the full target can be
    #  directly spliced from even for large sampling and patch sizes
    extended_positions = np.roll(arr, lag)
    extended_positions = loop_track_for_patching_sampling(extended_positions, sampling, patch_size)

    for normal_index in range(len(arr)):
        # Calculate here in the looped vector to splice from
        patch_start = (normal_index + 1 - patch_size - (sampling * patch_size)) % len(arr)
        patch_end = patch_start + (patch_size * (sampling * 2 + 1))

        output[normal_index] = extended_positions[patch_start:patch_end]

    return output

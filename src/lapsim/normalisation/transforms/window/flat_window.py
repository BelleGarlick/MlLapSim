import numpy as np

from lapsim.normalisation.transforms.window.base import BaseWindowTransform


"""This module encodes data into windows as described in garlick and bradley 
2021, whereby the network trains on a series of windows to predict the vehicle
position and velocity within the center of that window.

This module transforms the data into a single vector allowing a dense NN to 
train upon."""


class FlatWindowTransform(BaseWindowTransform):

    def transform(self, normalised_records: list[dict], cores: int):
        """Encode the data into a series of windows (as described in Garlick &
        Bradley 2021, but compress each window into a single vector containing
        widths, angles, offsets and vehicles.

        Args:
            normalised: The normalised partition from the normalisation step
            cores: Number of cores used to multiprocess the track using

        Returns:
            (x, vehicles), (y_pos, y_vel)
        """
        # Get the track window representations
        # track_encodings = self.perform_parallel_transforms(_flat_window, normalised, cores)
        normalised_records = list(normalised_records)

        args = [(record, self) for record in normalised_records]
        track_encodings = list(map(flat_window, args))

        window_length = self.foresight * 2 + 1
        total_window_size = 3 * window_length  # width,offsets,angles

        total_normals_count = sum([len(x[0]) for x in track_encodings])

        # Preallocate the memory, this makes it much faster and memory efficient as
        # the arrays don't need reallocating
        x = np.zeros((total_normals_count, total_window_size), dtype=np.float32)
        vehicles = np.zeros((total_normals_count, len(track_encodings[0][1])), dtype=np.float32)

        global_index = 0
        for track_encoding, vehicle_encoding in track_encodings:
            track_length = len(track_encoding)

            x[global_index: global_index + track_length] = track_encoding
            vehicles[global_index:global_index + track_length] = vehicle_encoding

            global_index += track_length

        return x, vehicles


def flat_window(args):
    record, transform = args

    window_length = transform.foresight * 2 + 1

    track_length = len(record["widths"])
    x = np.zeros((track_length, window_length * len(transform.inputs)))

    # Extract inputs from the transform based on defined keys
    inputs = [record[_inp] for _inp in transform.inputs]

    for normal_index in range(track_length):
        for i, f in enumerate(range(normal_index - transform.foresight, normal_index + transform.foresight + 1)):
            index = f % track_length

            # Iterate through the inputs splicing in where needed
            for inp_idx in range(len(transform.inputs)):
                x[normal_index, i + (window_length * inp_idx)] = inputs[inp_idx][index]

    return x, record["vehicle"]


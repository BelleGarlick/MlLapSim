import math

import numpy as np

from lapsim.normalisation.transforms.sampling import loop_track_for_patching_sampling
from test_lapsim.test_normalisation.test_transforms.test_transform_base import TestTransformBase


class TestStatefulLagNormalisationTransform(TestTransformBase):

    def test_lag_transform_toy(self):
        """Loop through the data and check it's formatted as it should be"""
        normalised_data, partition, transform, inp, output, v = self.load_toy_partition(
            "stateful-lag", lag=2, sampling=1)
        self.assertLagCorrect(normalised_data, partition, transform, inp, output, v, 2, 1)

        normalised_data, partition, transform, inp, output, v = self.load_toy_partition(
            "stateful-lag", lag=2, sampling=2, patch_size=3)
        self.assertLagCorrect(normalised_data, partition, transform, inp, output, v, 2, 2, patch_size=3)

        normalised_data, partition, transform, inp, output, v = self.load_toy_partition(
            "stateful-lag", lag=2, sampling=2, patch_size=3, time_to_vec=True)
        self.assertLagCorrect(
            normalised_data, partition, transform, inp, output, v, 2, 2, patch_size=3, time_to_vec=True)

        normalised_data, partition, transform, inp, output, v = self.load_toy_partition(
            "stateful-lag", lag=1, sampling=1, patch_size=10)
        self.assertLagCorrect(normalised_data, partition, transform, inp, output, v, 1, 1, patch_size=10)

        normalised_data, partition, transform, inp, output, v = self.load_toy_partition(
            "stateful-lag", lag=1, sampling=1, patch_size=10, time_to_vec=True)
        self.assertLagCorrect(normalised_data, partition, transform, inp, output, v, 1, 1, patch_size=10, time_to_vec=True)

    def test_lag_transform_real(self):
        """Test that the lag works with the correct lag and sampling
        mapping."""
        normalised_data, partition, transform, x, y, v = self.load_real_partition("stateful-lag", lag=20, sampling=4)
        self.assertLagCorrect(normalised_data, partition, transform, x, y, v, lag=20, sampling=4)

        normalised_data, partition, transform, x, y, v = self.load_real_partition("stateful-lag", lag=20, sampling=4, patch_size=10)
        self.assertLagCorrect(normalised_data, partition, transform, x, y, v, lag=20, sampling=4, patch_size=10)

        normalised_data, partition, transform, input, output, v = self.load_real_partition("stateful-lag", lag=10, sampling=2, patch_size=3, time_to_vec=True)
        self.assertLagCorrect(normalised_data, partition, transform, input, output, v, 10, 2, patch_size=3, time_to_vec=True)

    def assertLagCorrect(self, normalised_data, partition, transform, x, y, v, lag, sampling, patch_size: int = 1, time_to_vec: int = False):
        for track_index in range(len(normalised_data)):
            track_length = normalised_data.track_length(track_index)
            track = x[track_index]
            vehicle = v[track_index]
            y_pos, y_vel = y[track_index]

            # Loop the track enough times so the sampling and patching can be spliced from
            #  when large sampling or patching values are u sed
            extended_positions = loop_track_for_patching_sampling(
                normalised_data.positions[track_index], sampling=sampling, patch_size=patch_size)
            extended_velocities = loop_track_for_patching_sampling(
                normalised_data.velocities[track_index], sampling=sampling, patch_size=patch_size)

            self.assertTupleEqual((math.ceil((track_length * 2) / patch_size), 3 * patch_size + int(time_to_vec)), track.shape)
            self.assertTupleEqual((16, ), vehicle.shape)

            for normal_index in range(normalised_data.track_length(track_index) * 2):
                # Find the amount of subcells prior to the track
                #  starting due to padding (the cells filled with -1)
                prefix = (math.ceil((track_length * 2) / patch_size) * patch_size) - (track_length * 2)

                i = normal_index % track_length
                col = (prefix + normal_index) // patch_size
                row = (prefix + normal_index) % patch_size
                row_idx = row * 3
                self.assertFloatListEqual(track[col, row_idx:row_idx + 3], [
                    normalised_data.widths[track_index][i],
                    normalised_data.angles[track_index][i],
                    normalised_data.offsets[track_index][i]
                ])

            # Test sampling
            for normal_index in range(len(y_pos)):
                normal_target_pos = y_pos[normal_index]
                N, P, L = track_length, patch_size, (len(y_pos) - normal_index - 1)
                start = (N - L*P - sampling*P - P - lag) % N
                end = start + len(normal_target_pos)

                self.assertFloatListEqual(extended_positions[start:end], y_pos[normal_index])
                self.assertFloatListEqual(extended_velocities[start:end], y_vel[normal_index])

            # Test the time to vec is correct
            if time_to_vec:
                patched_track_len = math.ceil(track_length / patch_size)
                time_to_vec_arr = np.arange(patched_track_len) / (max(1, patched_track_len - 1))

                # Create the ground array
                ground_truth = np.zeros(track.shape[0])
                ground_truth[:len(time_to_vec_arr)] = time_to_vec_arr
                ground_truth[-len(time_to_vec_arr):] = time_to_vec_arr

                window_time_vec = track[:, -1]

                self.assertFloatListEqual(ground_truth, window_time_vec)

            self.assertFloatListEqual(normalised_data.vehicles[track_index], vehicle)

        # Testing detransform
        global_normal_index = 0
        for i in range(len(normalised_data)):
            track_length = len(normalised_data.angles[i])
            pred_pos, pred_vel = transform.detransform_and_denormalise(
                track_length=track_length,
                position=y[i][0],
                velocity=y[i][1]
            )
            self.assertFloatListEqual(pred_pos, partition.positions[i])
            self.assertFloatListEqual(pred_vel, partition.velocities[i])
            global_normal_index += track_length

import math

import numpy as np

from test_lapsim.test_normalisation.test_transforms.test_transform_base import TestTransformBase


class TestLagNormalisationTransform(TestTransformBase):

    def test_lag_transform_toy(self):
        """Loop through the data and check it's formatted as it should be"""
        normalised_data, partition, transform, inp, output, vehicles = self.load_toy_partition(
            "lag", lag=2, sampling=1)
        self.assertTupleEqual(inp.shape, (6, 12, 3))
        self.assertTupleEqual(vehicles.shape, (6, 16))
        self.assertLagCorrect(normalised_data, partition, transform, inp, output, vehicles, 2, 1)

        normalised_data, partition, transform, inp, output, vehicles = self.load_toy_partition(
            "lag", lag=2, sampling=2, patch_size=3)
        self.assertTupleEqual(inp.shape, (6, 4, 9))
        self.assertLagCorrect(normalised_data, partition, transform, inp, output, vehicles, 2, 2, patch_size=3)

        normalised_data, partition, transform, inp, output, vehicles = self.load_toy_partition(
            "lag", lag=2, sampling=2, patch_size=3, time_to_vec=True)
        self.assertTupleEqual(inp.shape, (6, 4, 10))
        self.assertLagCorrect(normalised_data, partition, transform, inp, output, vehicles, 2, 2, patch_size=3, time_to_vec=True)

        normalised_data, partition, transform, inp, output, vehicles = self.load_toy_partition(
            "lag", lag=1, sampling=1, patch_size=10)
        self.assertTupleEqual(inp.shape, (6, 2, 30))
        self.assertLagCorrect(normalised_data, partition, transform, inp, output, vehicles, 1, 1, patch_size=10)

        normalised_data, partition, transform, inp, output, vehicles = self.load_toy_partition(
            "lag", lag=1, sampling=1, patch_size=10, time_to_vec=True)
        self.assertTupleEqual(inp.shape, (6, 2, 31))
        self.assertLagCorrect(normalised_data, partition, transform, inp, output, vehicles, 1, 1, patch_size=10, time_to_vec=True)

    def test_lag_transform_real(self):
        """Test that the lag works with the correct lag and sampling mapping"""
        normalised_data, partition, transform, inp, output, vehicles = self.load_real_partition("lag", lag=20, sampling=4)
        self.assertTupleEqual(inp.shape, (2219, 2026, 3))
        self.assertTupleEqual(vehicles.shape, (2219, 16))
        self.assertLagCorrect(normalised_data, partition, transform, inp, output, vehicles, 20, 4)

        normalised_data, partition, transform, inp, output, vehicles = self.load_real_partition("lag", lag=20, sampling=4, patch_size=10)
        self.assertTupleEqual(inp.shape, (2219, 204, 30))  # 2 * math.ceil((2026 / 2) / 10)
        self.assertLagCorrect(normalised_data, partition, transform, inp, output, vehicles, 20, 4, patch_size=10)

        normalised_data, partition, transform, inp, output, vehicles = self.load_real_partition("lag", lag=10, sampling=2, patch_size=3, time_to_vec=True)
        self.assertTupleEqual(inp.shape, (2219, 676, 10))
        self.assertLagCorrect(normalised_data, partition, transform, inp, output, vehicles, 10, 2, patch_size=3, time_to_vec=True)

    def assertLagCorrect(self, normalised_data, partition, transform, x, output, vehicles, lag, sampling, patch_size=1, time_to_vec=False):
        """Check lag is correct"""
        y_pos, y_vel = output

        global_normal_index = 0
        for track_index in range(len(normalised_data)):
            track_length = len(normalised_data.angles[track_index])
            patched_track_len = math.ceil(track_length / patch_size)
            time_to_vec_arr = np.arange(patched_track_len) / (max(1, patched_track_len - 1))

            for normal_index in range(len(normalised_data.angles[track_index])):
                window = x[global_normal_index]

                expected_normals = track_length + normal_index + 1
                prefix = (window.shape[0] * patch_size) - expected_normals

                for n in range(prefix):
                    col = n // patch_size
                    row = n % patch_size
                    self.assertTrue(np.all(window[col, row:row+3] == -1))

                for n in range(expected_normals):
                    i = n % track_length
                    col = (prefix + n) // patch_size
                    row = (prefix + n) % patch_size
                    row_idx = row * 3
                    self.assertFloatListEqual(window[col, row_idx:row_idx + 3], [
                        normalised_data.widths[track_index][i],
                        normalised_data.angles[track_index][i],
                        normalised_data.offsets[track_index][i]
                    ])

                # Test the time to vec is correct
                if time_to_vec:
                    # Create the ground array
                    ground_truth = np.zeros(window.shape[0])
                    ground_truth.fill(-1)
                    # Create the first track loop time to vecs
                    padding_start = prefix // patch_size
                    ground_truth[padding_start:padding_start + len(time_to_vec_arr)] = time_to_vec_arr
                    # Splice in the second track's normals
                    exp_normals_indexes = (normal_index // patch_size) + 1
                    ground_truth[-exp_normals_indexes:] = time_to_vec_arr[:exp_normals_indexes]

                    window_time_vec = window[:, -1]

                    self.assertFloatListEqual(ground_truth, window_time_vec)

                # Test vehicle is correct
                self.assertFloatListEqual(normalised_data.vehicles[track_index], vehicles[global_normal_index])

                global_normal_index += 1

        self.assertSamplingCorrect(normalised_data, y_pos, y_vel, sampling, lag=lag, patch_size=patch_size)

        # Testing detransform
        global_normal_index = 0
        for i in range(len(normalised_data)):
            track_length = len(normalised_data.angles[i])
            pred_pos, pred_vel = transform.detransform_and_denormalise(
                track_length=track_length,
                position=output[0][global_normal_index:global_normal_index + track_length],
                velocity=output[1][global_normal_index:global_normal_index + track_length]
            )
            self.assertFloatListEqual(pred_pos, partition.positions[i])
            self.assertFloatListEqual(pred_vel, partition.velocities[i])
            global_normal_index += track_length

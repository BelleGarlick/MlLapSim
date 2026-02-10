import numpy as np

from lapsim.encoder.partition import Partition
from lapsim.normalisation import TransformNormalisation
from lapsim.normalisation.transforms.transformer import Transform
from test_lapsim.test_normalisation.test_transforms.test_transform_base import TestTransformBase


class TestBidirectionalTransform(TestTransformBase):

    def test_bidirectional_transform_toy(self):
        """Loop through the data and check it's formatted as it should be"""
        normalised_data, partition, transform, x, outputs, vehicles = self.load_toy_partition(
            "bidirectional", sampling=2, patch_size=1)
        self.assertTupleEqual(x.shape, (6, 6, 6))
        self.assertBidirectionalEqual(
            normalised_data, partition, transform, x, outputs, vehicles, 2, 1)

        normalised_data, partition, transform, x, outputs, vehicles = self.load_toy_partition(
            "bidirectional", sampling=1, patch_size=7)
        self.assertTupleEqual(x.shape, (6, 1, 42))
        self.assertBidirectionalEqual(
            normalised_data, partition, transform, x, outputs, vehicles, 1, 7)

        normalised_data, partition, transform, x, outputs, vehicles = self.load_toy_partition(
            "bidirectional", sampling=1, patch_size=3)
        self.assertTupleEqual(x.shape, (6, 2, 18))
        self.assertBidirectionalEqual(
            normalised_data, partition, transform, x, outputs, vehicles, sampling=1, patch_size=3)

        normalised_data, partition, transform, x, outputs, vehicles = self.load_toy_partition(
            "bidirectional", sampling=1, patch_size=5)
        self.assertTupleEqual(x.shape, (6, 2, 30))
        self.assertBidirectionalEqual(
            normalised_data, partition, transform, x, outputs, vehicles, 1, 5)

    def test_bidirectional_transform_real(self):
        """Test that the bidirectional mapping works."""
        normalised_data, partition, transform, inputs, outputs, vehicles = self.load_real_partition(
            "bidirectional", sampling=4)
        self.assertTupleEqual(inputs.shape, (2219, 1013, 6))
        self.assertTupleEqual(vehicles.shape, (2219, 16))
        self.assertBidirectionalEqual(
            normalised_data, partition, transform, inputs, outputs, vehicles, 4, 1)

        normalised_data, partition, transform, inputs, outputs, vehicles = self.load_real_partition(
            "bidirectional", sampling=4, patch_size=10)
        self.assertTupleEqual(inputs.shape, (2219, 102, 60))
        self.assertBidirectionalEqual(
            normalised_data, partition, transform, inputs, outputs, vehicles, 4, 10)

    # TODO Use this to test what's happening with subsplicing being incorrect
    # def test_bidirectional_partition_testing(self):
    #     partition = Partition.load("/Volumes/Main/training/partition-0.json")
    #
    #     bounds = TransformNormalisation(
    #         transform=Transform(method="bidirectional", sampling=31, patch_size=71)).extend(partition)
    #
    #     vehicle_vectors = bounds.transform.vectorise_vehicles(partition.vehicles)
    #     normalised_data = bounds.bounds.normalise(partition, vehicle_vectors)
    #     inp, output, vehicles = bounds.transform.transform(normalised_data, cores=1)

    def assertBidirectionalEqual(
            self, normalised_data, partition, transform, x, outputs, vehicles, sampling, patch_size):
        y_pos, y_vel = outputs

        global_index_count = 0
        for track_index in range(len(normalised_data.angles)):
            track_length = len(normalised_data.angles[track_index])

            widths, angles, offsets = (
                normalised_data.widths[track_index],
                normalised_data.angles[track_index],
                normalised_data.offsets[track_index])

            for normal_index in range(track_length):
                window = x[global_index_count]

                expected_normals = track_length
                prefix = (window.shape[0] * patch_size) - expected_normals

                for n in range(prefix):
                    col = n // patch_size
                    row = n % patch_size
                    self.assertTrue(np.all(window[col, row:row+3] == -1))

                for n in range(expected_normals):
                    i = (normal_index + n + 1) % track_length
                    reversed_index = (normal_index - n - 1) % track_length

                    col = (prefix + n) // patch_size
                    row = (prefix + n) % patch_size
                    row_idx = row * 6
                    self.assertFloatListEqual(window[col, row_idx:row_idx + 6], [
                        widths[i],
                        angles[i],
                        offsets[i],
                        widths[reversed_index],
                        angles[reversed_index],
                        offsets[reversed_index]
                    ])

                # Test vehicle
                self.assertFloatListEqual(normalised_data.vehicles[track_index], vehicles[global_index_count])

                global_index_count += 1

        self.assertSamplingCorrect(normalised_data, y_pos, y_vel, sampling, patch_size=patch_size)

        # Testing detransform
        global_normal_index = 0
        for i in range(len(normalised_data)):
            track_length = len(normalised_data.angles[i])
            pred_pos, pred_vel = transform.detransform_and_denormalise(
                track_length=track_length,
                position=outputs[0][global_normal_index:global_normal_index + track_length],
                velocity=outputs[1][global_normal_index:global_normal_index + track_length]
            )
            self.assertFloatListEqual(pred_pos, partition.positions[i])
            self.assertFloatListEqual(pred_vel, partition.velocities[i])
            global_normal_index += track_length
